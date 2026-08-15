import argparse
from contextlib import nullcontext
import copy
import datetime
import json
import os
import random
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torchinfo import summary
from tqdm.auto import tqdm


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_DIR)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from lib.utils import (  # noqa: E402
    CustomJSONEncoder,
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
)
from lib.lazy_data_prepare import get_dataloaders_from_index_data  # noqa: E402
from lib.spatial_patching import load_patch_index  # noqa: E402
from model.STAEformer import STAEformer  # noqa: E402


# X shape: (B, T, N, C)
DEVICE = torch.device("cpu")
SCALER = None
AMP_ENABLED = False
AMP_DTYPE = torch.float16
GRAD_SCALER = None

# --------------------------------------------------------------------------- #
# Распределённое обучение (DDP)
#
# Включается автоматически, когда процесс запущен через torchrun -- тот
# выставляет RANK / LOCAL_RANK / WORLD_SIZE. Обычный запуск через python
# ничего не меняет, все ветки ниже становятся тождественными.
#
# ВАЖНО про батч: batch_size из конфига трактуется как размер НА КАРТУ,
# как принято в DDP. При двух картах эффективный батч удваивается, а число
# шагов в эпохе вдвое падает -- значит milestones и max_epochs надо
# пересчитывать. Фактические числа печатаются при старте.
# --------------------------------------------------------------------------- #
RANK = 0
LOCAL_RANK = 0
WORLD_SIZE = 1
IS_DISTRIBUTED = False


def _is_main():
    return RANK == 0


def _setup_distributed():
    """Поднимает process group, если процесс запущен через torchrun."""
    global RANK, LOCAL_RANK, WORLD_SIZE, IS_DISTRIBUTED, print_log

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False

    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", RANK))
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    if WORLD_SIZE <= 1:
        return False

    torch.cuda.set_device(LOCAL_RANK)
    # Таймаут по умолчанию 10 минут, а первая компиляция torch.compile может
    # занять больше и разъехаться по рангам -- тогда коллектив упал бы, не
    # дождавшись отставшего. Час с запасом.
    # device_id указываем явно: без него NCCL угадывает карту по глобальному
    # рангу и предупреждает, что при неоднородной раскладке это может привести
    # к зависанию на барьере.
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
        device_id=torch.device(f"cuda:{LOCAL_RANK}"),
    )
    IS_DISTRIBUTED = True

    # Печатает только нулевой ранг: иначе каждая строка лога задвоится,
    # а файл лога откроют на запись сразу несколько процессов.
    if not _is_main():
        print_log = lambda *args, **kwargs: None  # noqa: E731

    return True


def _reduce_mean(value):
    """Среднее значение по всем рангам. Вне DDP возвращает как есть."""
    if not IS_DISTRIBUTED:
        return value
    tensor = torch.tensor([float(value)], device=DEVICE, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item() / WORLD_SIZE)

# Сколько батчей ПОДРЯД с нечисловым лоссом терпеть, прежде чем падать.
# Одиночный overflow в fp16 -- не повод ронять 30-часовой прогон, но серия
# подряд означает, что обучение разошлось и продолжать бессмысленно.
MAX_CONSECUTIVE_NONFINITE = 5


def _unwrap_model(model):
    """Достаёт исходный модуль из-под DataParallel и torch.compile.

    Критично для save/load: torch.compile оборачивает модель в OptimizedModule,
    и все ключи её state_dict() получают префикс `_orig_mod.`. Сохранённые так
    веса не загрузятся в обычную модель, а загруженные в скомпилированную --
    молча не найдут своих параметров. Разворачивать надо всегда.
    """
    wrappers = (nn.DataParallel, nn.parallel.DistributedDataParallel)
    while True:
        if isinstance(model, wrappers):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def _amp_context():
    """Enable autocast (bf16 on A100+, иначе fp16) only when CUDA AMP is active."""
    if AMP_ENABLED:
        return torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
        )
    return nullcontext()


def _atomic_torch_save(obj, path):
    """Запись через временный файл: обрыв в момент сохранения не портит старый."""
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _move_batch_to_device(x_batch, y_batch):
    """Move a batch to the selected device."""
    non_blocking = DEVICE.type == "cuda"
    return (
        x_batch.to(DEVICE, non_blocking=non_blocking),
        y_batch.to(DEVICE, non_blocking=non_blocking),
    )


class MetricAccumulator:
    """Потоковый счётчик маскированных RMSE / MAE / MAPE.

    Считает и суммарные метрики, и отдельные по каждому горизонту. Нулевые
    цели игнорируются -- так же, как в lib/metrics.py, иначе мёртвые датчики
    (их наплодил fillna(0)) занизили бы ошибку.

    Накопление в float64: за эпоху набегает порядка 2e9 слагаемых, и во
    float32 сумма потеряла бы значащие цифры задолго до конца прохода.
    Синхронизация с GPU одна, в самом конце -- поэтому счётчик практически
    ничего не стоит поверх уже идущего forward.
    """

    def __init__(self):
        self.count = None
        self.abs_error = None
        self.squared_error = None
        self.percentage_error = None

    @torch.no_grad()
    def update(self, y_pred, y_true):
        mask = y_true != 0
        abs_error = torch.abs(y_pred - y_true) * mask
        squared_error = torch.square(y_pred - y_true) * mask

        safe_target = torch.where(mask, y_true, torch.ones_like(y_true))
        percentage_error = abs_error / torch.abs(safe_target)

        # Складываем по всему, кроме оси горизонтов (dim=1).
        dims = (0, 2, 3)
        batch = (
            mask.sum(dim=dims).double(),
            abs_error.sum(dim=dims).double(),
            squared_error.sum(dim=dims).double(),
            percentage_error.sum(dim=dims).double(),
        )

        if self.count is None:
            self.count, self.abs_error = batch[0], batch[1]
            self.squared_error, self.percentage_error = batch[2], batch[3]
        else:
            self.count += batch[0]
            self.abs_error += batch[1]
            self.squared_error += batch[2]
            self.percentage_error += batch[3]

    def result(self):
        """Возвращает (суммарные метрики, список по горизонтам).

        Под DDP каждый ранг видел свою часть выборки, поэтому накопленные
        суммы складываются по всем рангам. Метрики считаются из сумм, а не
        усреднением средних -- иначе неравные размеры частей дали бы смещение.
        """
        if self.count is None:
            raise ValueError("Метрики не накоплены: пустой загрузчик.")

        if IS_DISTRIBUTED:
            for tensor in (
                self.count,
                self.abs_error,
                self.squared_error,
                self.percentage_error,
            ):
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.SUM
                )

        count = self.count.cpu()
        abs_error = self.abs_error.cpu()
        squared_error = self.squared_error.cpu()
        percentage_error = self.percentage_error.cpu()

        total = float(count.sum())
        if total == 0:
            raise ValueError("Ни одной ненулевой цели: метрики не определены.")
        if bool((count == 0).any()):
            raise ValueError("У какого-то горизонта нет ненулевых целей.")

        all_metrics = {
            "rmse": float(torch.sqrt(squared_error.sum() / total)),
            "mae": float(abs_error.sum() / total),
            "mape": float(100.0 * percentage_error.sum() / total),
        }

        step_metrics = [
            {
                "step": i + 1,
                "rmse": float(torch.sqrt(squared_error[i] / count[i])),
                "mae": float(abs_error[i] / count[i]),
                "mape": float(100.0 * percentage_error[i] / count[i]),
            }
            for i in range(len(count))
        ]

        return all_metrics, step_metrics


@torch.no_grad()
def eval_model(model, valset_loader, criterion, epoch, max_epochs):
    """Считает валидационный лосс И маскированные метрики за ОДИН проход.

    Отдельный проход ради метрик стоил бы столько же, сколько вся валидация
    (~7 минут на CA). Тяжёлое здесь -- forward, а накопление метрик поверх
    него практически бесплатно.
    """
    model.eval()
    batch_loss_list = []
    accumulator = MetricAccumulator()

    progress = tqdm(
        valset_loader,
        desc=f"Validation {epoch}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not _is_main(),
    )

    for batch_number, (x_batch, y_batch) in enumerate(progress, start=1):
        x_batch, y_batch = _move_batch_to_device(x_batch, y_batch)

        with _amp_context():
            out_batch = model(x_batch)

        # Keep inverse scaling and the loss in FP32 for numerical stability.
        out_batch = SCALER.inverse_transform(out_batch.float())
        y_batch = y_batch.float()

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        accumulator.update(out_batch, y_batch)

        if batch_number % 20 == 0 or batch_number == len(valset_loader):
            progress.set_postfix(
                val_loss=f"{np.mean(batch_loss_list):.4f}",
            )

    if not batch_loss_list:
        raise RuntimeError("Validation loader is empty.")

    all_metrics, step_metrics = accumulator.result()

    # Лосс усредняем по рангам: иначе early stopping и выбор лучшей эпохи
    # приняли бы РАЗНЫЕ решения на разных рангах, и обучение разъехалось бы.
    loss = _reduce_mean(float(np.mean(batch_loss_list)))

    return loss, all_metrics, step_metrics


def train_one_epoch(
    model,
    trainset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad,
    epoch,
    max_epochs,
):
    """Train for one epoch and show batch-level loss and learning rate."""
    model.train()
    batch_loss_list = []
    consecutive_nonfinite = 0
    skipped_batches = 0

    progress = tqdm(
        trainset_loader,
        desc=f"Training {epoch}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not _is_main(),
    )

    for batch_number, (x_batch, y_batch) in enumerate(progress, start=1):
        x_batch, y_batch = _move_batch_to_device(x_batch, y_batch)

        optimizer.zero_grad(set_to_none=True)

        with _amp_context():
            out_batch = model(x_batch)

        # Keep inverse scaling and the loss in FP32 for numerical stability.
        out_batch = SCALER.inverse_transform(out_batch.float())
        loss = criterion(out_batch, y_batch.float())

        # Одиночный inf/nan (overflow активаций в fp16) -- пропускаем батч,
        # а не роняем прогон: с --resume и восстановленным RNG тот же батч
        # воспроизвёлся бы снова, и падение стало бы детерминированным циклом.
        # Падаем только если лосс нечисловой несколько батчей ПОДРЯД.
        #
        # ПОД DDP решение обязано быть ОБЩИМ. Батчи у рангов разные, поэтому
        # лосс может оказаться нечисловым только на одном. Если он пропустит
        # backward, а остальные нет, all-reduce градиентов спарятся не с теми
        # -- градиенты молча перемешаются, а к концу эпохи ранги разойдутся по
        # числу коллективов и NCCL повиснет до таймаута. Скаляр через MAX
        # стоит копейки и делает ветвление симметричным по построению.
        nonfinite = not bool(torch.isfinite(loss))
        if IS_DISTRIBUTED:
            flag = torch.tensor(
                [1.0 if nonfinite else 0.0], device=DEVICE, dtype=torch.float32
            )
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            nonfinite = bool(flag.item() > 0)

        if nonfinite:
            consecutive_nonfinite += 1
            skipped_batches += 1
            if _is_main():
                tqdm.write(
                    f"WARNING: non-finite loss at epoch {epoch}, "
                    f"batch {batch_number}; "
                    f"skipping batch ({consecutive_nonfinite}/"
                    f"{MAX_CONSECUTIVE_NONFINITE} consecutive)"
                )
            if consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
                raise FloatingPointError(
                    f"Non-finite loss in {consecutive_nonfinite} consecutive "
                    f"batches at epoch {epoch}: training has diverged."
                )
            optimizer.zero_grad(set_to_none=True)
            continue

        consecutive_nonfinite = 0

        GRAD_SCALER.scale(loss).backward()

        if clip_grad:
            # Gradients must be unscaled before gradient clipping.
            GRAD_SCALER.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_grad,
            )

        GRAD_SCALER.step(optimizer)
        GRAD_SCALER.update()
        batch_loss_list.append(loss.item())

        if batch_number % 20 == 0 or batch_number == len(trainset_loader):
            recent_loss = np.mean(batch_loss_list[-20:])
            progress.set_postfix(
                loss=f"{recent_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    if not batch_loss_list:
        raise RuntimeError("Train loader is empty.")

    if skipped_batches:
        tqdm.write(
            f"Epoch {epoch}: skipped {skipped_batches} batch(es) "
            f"with non-finite loss."
        )

    epoch_loss = _reduce_mean(float(np.mean(batch_loss_list)))
    scheduler.step()

    return epoch_loss


@torch.no_grad()
def calculate_streaming_metrics(model, loader, description):
    """Маскированные RMSE / MAE / MAPE без хранения всех предсказаний в RAM.

    Нулевые цели игнорируются, как в lib/metrics.py. Возвращает суммарные
    метрики и разбивку по горизонтам. Вся арифметика -- в MetricAccumulator,
    чтобы она была ровно та же, что и при валидации в конце эпохи.
    """
    model.eval()
    accumulator = MetricAccumulator()

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not _is_main(),
    )

    for batch_number, (x_batch, y_batch) in enumerate(progress, start=1):
        x_batch, y_batch = _move_batch_to_device(x_batch, y_batch)

        with _amp_context():
            y_pred = model(x_batch)

        accumulator.update(
            SCALER.inverse_transform(y_pred.float()),
            y_batch.float(),
        )

        if batch_number % 100 == 0 or batch_number == len(loader):
            progress.set_postfix(processed=batch_number)

    return accumulator.result()


def save_training_artifacts(history, history_path, plot_path):
    """Persist the epoch history and a loss chart."""
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train Loss",
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("STAEformer training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Возобновление обучения после падения
# --------------------------------------------------------------------------- #


def _capture_rng():
    """Состояния генераторов, чтобы возобновлённый прогон шёл тем же путём."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state, log=None):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as err:  # noqa: BLE001
        # Не повод падать: обучение продолжится, просто порядок батчей и
        # маски dropout будут другими, чем в прерванном прогоне.
        print_log(f"RNG не восстановлен ({type(err).__name__}: {err})", log=log)


def save_resume_state(path, state):
    """Атомарная запись: падение во время сохранения не портит прошлый файл.

    torch.save пишет во временный файл, и только полностью записанный он
    подменяет собой предыдущий. Иначе прерывание ровно в момент записи
    оставило бы обрезанный чекпойнт вместо рабочего.
    """
    _atomic_torch_save(state, path)


def load_resume_state(path, dataset, log=None):
    """Читает состояние и проверяет, что оно от того же датасета."""
    state = torch.load(path, map_location="cpu", weights_only=False)

    saved_dataset = state.get("dataset")
    if saved_dataset != dataset:
        raise ValueError(
            f"Чекпойнт {path} от датасета {saved_dataset!r}, "
            f"запрошен {dataset!r}"
        )

    print_log(
        f"Возобновление из {path}: пройдено эпох {state['epoch']}, "
        f"лучшая эпоха {state['best_epoch']} "
        f"(val loss {state['min_val_loss']:.5f})",
        log=log,
    )
    return state


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    log=None,
    save=None,
    history_path=None,
    plot_path=None,
    resume_path=None,
    resume_state=None,
    dataset=None,
    config=None,
    steps_per_day=None,
):
    """Train with early stopping, progress bars, timing and persistent history."""
    # .to(DEVICE) здесь НЕ делаем: модель уже на нужной карте (main), а
    # перемещение уже обёрнутого DDP-модуля инвалидирует бакеты редьюсера.
    wait = 0
    min_val_loss = np.inf
    best_epoch = -1
    best_state_dict = None
    history = []
    start_epoch = 0
    total_training_start = time.time()

    # Для подписи горизонтов в минутах: 288 шагов в сутки -> 5 минут на шаг.
    minutes_per_step = 1440 / steps_per_day if steps_per_day else None

    if resume_state is not None:
        _unwrap_model(model).load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

        if (
            GRAD_SCALER is not None
            and GRAD_SCALER.is_enabled()
            and resume_state.get("grad_scaler")
        ):
            # У AMP-скейлера свой подобранный масштаб; без него первые шаги
            # после возобновления уйдут на его перекалибровку. В выключенный
            # скейлер (bf16-режим) состояние не грузим: чекпойнт мог быть
            # записан ещё в fp16-режиме, и его масштаб там не нужен.
            GRAD_SCALER.load_state_dict(resume_state["grad_scaler"])

        start_epoch = int(resume_state["epoch"])
        wait = int(resume_state["wait"])
        min_val_loss = float(resume_state["min_val_loss"])
        best_epoch = int(resume_state["best_epoch"])
        best_state_dict = resume_state["best_state_dict"]
        history = list(resume_state["history"])
        _restore_rng(resume_state.get("rng"), log=log)

        if start_epoch >= max_epochs:
            print_log(
                f"В чекпойнте уже {start_epoch} эпох при max_epochs="
                f"{max_epochs}: обучать нечего, перехожу к оценке.",
                log=log,
            )

    for epoch_index in range(start_epoch, max_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()

        # Без set_epoch DistributedSampler перемешивает выборку одинаково
        # каждую эпоху, и модель видит один и тот же порядок все 7 эпох.
        sampler = getattr(trainset_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model=model,
            trainset_loader=trainset_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            clip_grad=clip_grad,
            epoch=epoch,
            max_epochs=max_epochs,
        )

        val_loss, val_metrics, val_step_metrics = eval_model(
            model=model,
            valset_loader=valset_loader,
            criterion=criterion,
            epoch=epoch,
            max_epochs=max_epochs,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        epoch_seconds = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < min_val_loss

        if improved:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(_unwrap_model(model).state_dict())

            # Пишет только нулевой ранг: веса на всех рангах одинаковы (DDP
            # синхронизирует градиенты), а несколько процессов, пишущих в один
            # файл, гарантированно его испортят.
            if save and _is_main():
                # Атомарно: обрыв во время записи не портит прошлый файл.
                _atomic_torch_save(best_state_dict, save)
        else:
            wait += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_mape": val_metrics["mape"],
                # Разбивка по горизонтам прямо в CSV: val_mae_step1..12.
                **{
                    f"val_mae_step{item['step']}": item["mae"]
                    for item in val_step_metrics
                },
                "learning_rate": current_lr,
                "epoch_seconds": epoch_seconds,
                "best_so_far": improved,
            }
        )

        if history_path and plot_path and _is_main():
            save_training_artifacts(
                history=history,
                history_path=history_path,
                plot_path=plot_path,
            )

        # Полное состояние пишется КАЖДУЮ эпоху, а не только при улучшении:
        # возобновляться надо с того места, где прервались, а не с последнего
        # удачного. Веса лучшей эпохи лежат тут же отдельным ключом.
        # Под DDP пишет только нулевой ранг, остальные ждут его на барьере
        # ниже -- чтобы никто не ушёл в следующую эпоху раньше записи.
        if resume_path and _is_main():
            save_resume_state(
                resume_path,
                {
                    "epoch": epoch,
                    "dataset": dataset,
                    "config": config,
                    "model": _unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "grad_scaler": (
                        GRAD_SCALER.state_dict() if GRAD_SCALER is not None else None
                    ),
                    "best_state_dict": best_state_dict,
                    "best_epoch": best_epoch,
                    "min_val_loss": min_val_loss,
                    "wait": wait,
                    "history": history,
                    "rng": _capture_rng(),
                },
            )

        # Барьер после записи: остальные ранги не должны уйти в следующую
        # эпоху, пока нулевой не дописал чекпойнт. Иначе при обрыве ровно в
        # этот момент состояние на диске отстало бы от состояния в памяти.
        if IS_DISTRIBUTED:
            torch.distributed.barrier()

        if epoch % verbose == 0:
            marker = " *" if improved else ""
            print_log(
                datetime.datetime.now(),
                f"Epoch {epoch}/{max_epochs}{marker}",
                "Train Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                "LR = %.2e" % current_lr,
                "Time = %.2f min" % (epoch_seconds / 60),
                f"Early-stop wait = {wait}/{early_stop}",
                log=log,
            )
            # Маскированные метрики -- те же, что печатаются в конце по тесту,
            # и та же величина, что у других реализаций. В отличие от Val Loss,
            # который считается Huber'ом БЕЗ маски и потому с чужими таблицами
            # не сопоставим.
            print_log(
                "    Val MAE = %.4f, RMSE = %.4f, MAPE = %.2f%%"
                % (
                    val_metrics["mae"],
                    val_metrics["rmse"],
                    val_metrics["mape"],
                ),
                log=log,
            )
            if minutes_per_step:
                print_log(
                    "    MAE по горизонтам: "
                    + "  ".join(
                        "%g мин %.3f" % (item["step"] * minutes_per_step, item["mae"])
                        for item in val_step_metrics
                    ),
                    log=log,
                )

        if wait >= early_stop:
            print_log(
                f"Early stopping triggered after epoch {epoch}.",
                log=log,
            )
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without a valid model state.")

    _unwrap_model(model).load_state_dict(best_state_dict)

    # Гарантия существования файла с лучшими весами. Без этого после --resume
    # без единого улучшения (лучшая эпоха осталась в прошлом отрезке)
    # timestamped .pt этого прогона не был бы записан вовсе, а строка
    # "Saved Model: ..." указывала бы на несуществующий файл.
    # Только ранг 0: имя файла содержит отметку времени, вычисленную в каждом
    # процессе отдельно, и при совпадении секунды оба писали бы в один путь.
    if save and _is_main():
        _atomic_torch_save(best_state_dict, save)

    # Суммируем по истории, а не по часам этого процесса: после возобновления
    # time.time() отсчитывается заново и показал бы только последний отрезок.
    total_training_seconds = sum(
        float(record.get("epoch_seconds", 0.0)) for record in history
    )

    print_log(
        f"Training finished in {total_training_seconds / 3600:.2f} hours "
        f"({len(history)} эпох суммарно, включая возобновления).",
        log=log,
    )
    print_log(
        f"Best epoch: {best_epoch}; best validation loss: {min_val_loss:.5f}",
        log=log,
    )

    print_log("Calculating streaming train metrics...", log=log)
    train_metrics, _ = calculate_streaming_metrics(
        model,
        trainset_loader,
        description="Train metrics",
    )

    print_log("Calculating streaming validation metrics...", log=log)
    val_metrics, _ = calculate_streaming_metrics(
        model,
        valset_loader,
        description="Validation metrics",
    )

    out_str = (
        f"Best at epoch {best_epoch}:\n"
        f"Train RMSE = {train_metrics['rmse']:.5f}, "
        f"MAE = {train_metrics['mae']:.5f}, "
        f"MAPE = {train_metrics['mape']:.5f}\n"
        f"Val Loss = {min_val_loss:.5f}\n"
        f"Val RMSE = {val_metrics['rmse']:.5f}, "
        f"MAE = {val_metrics['mae']:.5f}, "
        f"MAPE = {val_metrics['mape']:.5f}"
    )
    print_log(out_str, log=log)

    return model, history


@torch.no_grad()
def test_model(model, testset_loader, log=None, steps_per_day=None):
    """Calculate test metrics in a streaming, memory-safe manner.

    steps_per_day нужен только для подписи горизонтов в минутах: при 288 шаг
    равен 5 минутам, при 96 -- 15. Без него шаги печатаются без подписи.
    """
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()

    all_metrics, step_metrics = calculate_streaming_metrics(
        model,
        testset_loader,
        description="Test metrics",
    )

    inference_seconds = time.time() - start

    out_str = (
        "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n"
        % (
            all_metrics["rmse"],
            all_metrics["mae"],
            all_metrics["mape"],
        )
    )

    minutes_per_step = (
        1440 / steps_per_day if steps_per_day else None
    )

    for item in step_metrics:
        horizon = (
            " (%g мин)" % (item["step"] * minutes_per_step)
            if minutes_per_step
            else ""
        )
        out_str += (
            "Step %d%s RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n"
            % (
                item["step"],
                horizon,
                item["rmse"],
                item["mae"],
                item["mape"],
            )
        )

    print_log(out_str, log=log, end="")
    print_log(
        "Inference time: %.2f s" % inference_seconds,
        log=log,
    )


def main():
    global DEVICE, SCALER, AMP_ENABLED, AMP_DTYPE, GRAD_SCALER

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument(
        "-g",
        "--gpu_num",
        type=str,
        default="0,1",
        help='Visible GPU IDs, for example "0,1".',
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение с последнего сохранённого состояния.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help=(
            "Путь к файлу состояния. По умолчанию "
            "saved_models/<модель>-<датасет>-resume.pt"
        ),
    )
    parser.add_argument(
        "--no-resume-save",
        action="store_true",
        help="Не писать состояние для возобновления (экономит запись на диск).",
    )
    args = parser.parse_args()

    # Под torchrun карты выбираются переменной CUDA_VISIBLE_DEVICES снаружи,
    # а внутри каждый ранг берёт свою по LOCAL_RANK. Флаг -g в этом случае
    # игнорируется -- иначе процессы передрались бы за одну карту.
    distributed = _setup_distributed()

    # Разный сид на ранг: с одинаковым оба применяли бы побитово одинаковые
    # маски dropout, и эффективный батч регуляризовался бы одной маской вместо
    # двух независимых. Веса при этом не разъедутся -- DDP броадкастит
    # параметры от нулевого ранга при конструировании.
    seed = 42
    seed_everything(seed + RANK)
    set_cpu_num(1)

    if distributed:
        DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        # setdefault, а не присваивание: заданный снаружи CUDA_VISIBLE_DEVICES
        # важнее дефолта флага -g ("0,1"). Иначе запуск вида
        #   CUDA_VISIBLE_DEVICES=2,3 python model/train.py -d CA
        # молча уехал бы на физическую карту 0.
        if args.gpu_num is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_num)
        DEVICE = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    AMP_ENABLED = DEVICE.type == "cuda"

    # bf16, если карта умеет НАТИВНО: тот же диапазон экспоненты, что у fp32,
    # поэтому overflow активаций невозможен в принципе и GradScaler не нужен.
    #
    # Проверяем поколение, а не is_bf16_supported(): на Turing (T4, SM 7.5)
    # он возвращает True через программную эмуляцию, но нативных ядер нет,
    # и часть операций просто падает. Нативный bf16 -- с Ampere, SM 8.0.
    compute_capability = (
        torch.cuda.get_device_capability(DEVICE) if AMP_ENABLED else (0, 0)
    )
    if AMP_ENABLED and compute_capability[0] >= 8:
        AMP_DTYPE = torch.bfloat16
    else:
        AMP_DTYPE = torch.float16

    # Скейлер нужен только fp16: у него градиенты уходят в underflow без
    # масштабирования. С enabled=False все его вызовы -- прозрачные no-op.
    GRAD_SCALER = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_ENABLED and AMP_DTYPE == torch.float16,
    )

    if DEVICE.type == "cuda":
        # Бэкенд `math` материализует полную матрицу (..., N, N) -- ровно ту
        # квадратичную память, от которой мы уходим, -- и PyTorch сваливается
        # на него МОЛЧА, когда быстрые ядра неприменимы. Поэтому на картах, где
        # flash доступен, мы его отключаем: неподходящая конфигурация тогда
        # падает сразу, а не сжигает десятки гигабайт незаметно.
        #
        # Flash требует Ampere (SM 8.0+). На Turing (T4) и старше его нет, и
        # без math не остаётся ни одного бэкенда -- SDPA падает с "Invalid
        # backend". Там math приходится оставить; на малых графах вроде SD
        # (716 узлов) квадратичная матрица всё равно копеечная.
        flash_available = compute_capability[0] >= 8

        torch.backends.cuda.enable_flash_sdp(flash_available)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(not flash_available)

    dataset = args.dataset.upper()
    data_path = os.path.join(REPO_DIR, "data", dataset)
    model_name = STAEformer.__name__
    config_path = os.path.join(MODEL_DIR, f"{model_name}.yaml")

    with open(config_path, "r", encoding="utf-8") as file:
        all_config = yaml.safe_load(file)

    if dataset not in all_config:
        raise KeyError(
            f"Dataset {dataset!r} was not found in {config_path}."
        )

    cfg = all_config[dataset]

    model_args = dict(cfg["model_args"])

    # Пространственный патчинг: плотное внимание по узлам заменяется на пару
    # depth/breadth по патчам leaf-KDTree. Индексы считаются один раз по
    # meta-файлу с координатами и кэшируются рядом с ним.
    # Убрать секцию patching из конфига -> плотное внимание, как в оригинале.
    patch_cfg = cfg.get("patching")
    if patch_cfg:
        meta_path = patch_cfg.get("meta")
        if meta_path is None:
            raise KeyError(f"В секции patching для {dataset} не указан meta-файл")
        if not os.path.isabs(meta_path):
            meta_path = os.path.join(data_path, meta_path)

        # Кэш индексов строит и пишет только ранг 0: иначе процессы полезли бы
        # в один .npz одновременно и оставили бы обрезанный файл. Остальные
        # ждут на барьере и читают уже готовый.
        if _is_main() or not distributed:
            patch_index = load_patch_index(
                meta_path,
                recur=patch_cfg["recur"],
                factors=patch_cfg["factors"],
                leaf_size=patch_cfg.get("leaf_size"),
            )
        if distributed:
            torch.distributed.barrier()
            if not _is_main():
                patch_index = load_patch_index(
                    meta_path,
                    recur=patch_cfg["recur"],
                    factors=patch_cfg["factors"],
                    leaf_size=patch_cfg.get("leaf_size"),
                )

        model_args["patch_index"] = patch_index

    model_args["use_checkpoint"] = bool(cfg.get("use_checkpoint", False))

    model = STAEformer(**model_args)
    model = model.to(DEVICE)

    if distributed:
        # DDP оборачивается ДО torch.compile: dynamo видит обёртку и сам
        # расставляет границы графа под all-reduce градиентов (DDPOptimizer).
        # В обратном порядке компиляция и синхронизация мешают друг другу.
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
        )
    elif DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(
            model,
            device_ids=list(range(torch.cuda.device_count())),
            output_device=0,
        )

    # torch.compile. Замеры на CA дали 9-11 TFLOP/s при пике A100 в 312, то есть
    # ~3%: время уходит не в матричные умножения, а в запуски ядер и в обход
    # памяти -- LayerNorm, dropout, residual, транспонирования, index_select.
    # Ровно этот профиль inductor и лечит, сливая цепочки поэлементных операций
    # в одно ядро.
    #
    # dynamic=False принципиально. Лоадеры идут без drop_last, так что кроме
    # основной формы батча появляются остатки (на CA при batch 8: 1 в train,
    # 5 в val/test). При настройках по умолчанию torch.compile на второй форме
    # уходит в динамический режим и теряет часть выигрыша на ВСЕХ батчах.
    # С dynamic=False получаем три отдельных статических графа, каждый быстрый.
    compile_mode = cfg.get("compile")

    if compile_mode is None or str(compile_mode).lower() in ("false", "none", "off"):
        compile_mode = None
    elif str(compile_mode).lower() in ("true", "on", "1"):
        compile_mode = "default"

    if compile_mode is not None:
        if isinstance(model, nn.DataParallel):
            raise RuntimeError(
                "torch.compile и DataParallel вместе не проверены: реплики "
                "ломают кэш компиляции. Для нескольких карт запускайте через "
                "torchrun -- это путь DDP, он с компиляцией дружит."
            )

        # Разрешаем TF32 для тех матричных умножений, что остаются в fp32
        # (под autocast это в основном LayerNorm и голова).
        torch.set_float32_matmul_precision("high")

        model = torch.compile(model, mode=compile_mode, dynamic=False)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    log_dir = os.path.join(REPO_DIR, "logs")
    saved_models_dir = os.path.join(REPO_DIR, "saved_models")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"{model_name}-{dataset}-{now}.log",
    )
    history_path = os.path.join(
        log_dir,
        f"{model_name}-{dataset}-{now}-history.csv",
    )
    plot_path = os.path.join(
        log_dir,
        f"{model_name}-{dataset}-{now}-loss.png",
    )
    save_path = os.path.join(
        saved_models_dir,
        f"{model_name}-{dataset}-{now}.pt",
    )

    # Имя без отметки времени: возобновляться надо по предсказуемому пути,
    # а не разыскивать последний файл среди прогонов.
    resume_path = args.resume_path or os.path.join(
        saved_models_dir,
        f"{model_name}-{dataset}-resume.pt",
    )

    # Файл лога открывает на запись только нулевой ранг: иначе несколько
    # процессов усекли бы его одновременно. На остальных print_log всё равно
    # заглушён, так что им достаточно пустышки.
    log = open(
        log_path if _is_main() else os.devnull,
        "w",
        encoding="utf-8",
    )

    try:
        print_log(f"Device: {DEVICE}", log=log)

        if DEVICE.type == "cuda":
            print_log(
                f"Visible GPUs: {torch.cuda.device_count()}",
                log=log,
            )
            for gpu_index in range(torch.cuda.device_count()):
                print_log(
                    f"GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}",
                    log=log,
                )

        print_log(
            "Multi-GPU: "
            + (
                f"DDP, {WORLD_SIZE} процессов (этот -- ранг {RANK}, "
                f"карта cuda:{LOCAL_RANK})"
                if IS_DISTRIBUTED
                else f"DataParallel on {torch.cuda.device_count()} GPUs"
                if isinstance(model, nn.DataParallel)
                else "disabled"
            ),
            log=log,
        )
        print_log(
            "AMP: "
            + (
                f"enabled ({'bf16' if AMP_DTYPE == torch.bfloat16 else 'fp16'}, "
                f"GradScaler {'on' if GRAD_SCALER.is_enabled() else 'off'})"
                if AMP_ENABLED
                else "disabled"
            ),
            log=log,
        )
        if AMP_ENABLED:
            print_log(
                f"SDPA: compute capability {compute_capability[0]}."
                f"{compute_capability[1]}, "
                + (
                    "flash доступен, math отключён"
                    if compute_capability[0] >= 8
                    else "flash НЕДОСТУПЕН (нужен Ampere), math оставлен "
                    "запасным -- на больших графах он даст квадратичную память"
                ),
                log=log,
            )
        print_log(dataset, log=log)

        (
            trainset_loader,
            valset_loader,
            testset_loader,
            SCALER,
        ) = get_dataloaders_from_index_data(
            data_path,
            tod=cfg.get("time_of_day"),
            dow=cfg.get("day_of_week"),
            batch_size=cfg.get("batch_size", 64),
            log=log,
            distributed=distributed,
        )
        print_log(log=log)

        if distributed:
            steps_per_epoch = len(trainset_loader)
            print_log(
                f"DDP: {WORLD_SIZE} карт, batch_size {cfg['batch_size']} НА КАРТУ "
                f"-> эффективный батч {cfg['batch_size'] * WORLD_SIZE}\n"
                f"     шагов оптимизатора в эпохе: {steps_per_epoch} "
                f"(на одной карте было бы {steps_per_epoch * WORLD_SIZE})\n"
                f"     ВНИМАНИЕ: milestones и max_epochs считаются в эпохах, "
                f"а эпоха стала короче в {WORLD_SIZE} раз -- проверьте расписание.",
                log=log,
            )

        if dataset in ("METRLA", "PEMSBAY"):
            criterion = MaskedMAELoss()
        elif dataset in (
            "PEMS03",
            "PEMS04",
            "PEMS07",
            "PEMS08",
            "SD",
            "GBA",
            "GLA",
            "CA",
        ):
            criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0),
            eps=cfg.get("eps", 1e-8),
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["milestones"],
            gamma=cfg.get("lr_decay_rate", 0.1),
        )

        print_log("---------", model_name, "---------", log=log)
        print_log(
            json.dumps(
                cfg,
                ensure_ascii=False,
                indent=4,
                cls=CustomJSONEncoder,
            ),
            log=log,
        )

        if patch_cfg:
            print_log(
                f"Patching: {patch_index['num_patches']} патчей по "
                f"{patch_index['patch_size']} слотов, всего "
                f"{patch_index['num_slots']} на {patch_index['num_nodes']} узлов "
                f"(паддинг {patch_index['num_slots'] / patch_index['num_nodes'] - 1:+.1%}), "
                f"лист {patch_index['leaf_size']}",
                log=log,
            )
        else:
            print_log("Patching: выключен (плотное внимание по узлам)", log=log)
        print_log(
            f"Gradient checkpointing: "
            f"{'включён' if model_args['use_checkpoint'] else 'выключен'}",
            log=log,
        )
        if compile_mode is None:
            print_log("torch.compile: выключен\n", log=log)
        else:
            print_log(
                f"torch.compile: режим {compile_mode}, dynamic=False.\n"
                "Первые батчи каждой новой формы уходят на компиляцию: обычно "
                "1-3 минуты, для max-autotune до 10-15. Прогресс-бар на это "
                "время замирает -- это норма, а не зависание. Форм будет "
                "несколько: основной батч плюс неполные остатки train/val/test.\n",
                log=log,
            )

        input_feature_count = next(iter(trainset_loader))[0].shape[-1]

        # summary() гоняет модель в fp32 вне autocast. Flash в fp32 не работает
        # вообще, math отключён выше, так что весь этот проход держится на
        # mem-efficient. Если он на конкретной сборке не подойдёт, падать будет
        # диагностическая печать -- ронять из-за неё многочасовой прогон нельзя.
        try:
            print_log(
                summary(
                    _unwrap_model(model),
                    [
                        cfg["batch_size"],
                        cfg["in_steps"],
                        cfg["num_nodes"],
                        input_feature_count,
                    ],
                    verbose=0,
                ),
                log=log,
            )
        except Exception as err:  # noqa: BLE001
            print_log(f"summary() пропущен: {type(err).__name__}: {err}", log=log)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print_log(f"Trainable params: {n_params:,}", log=log)
        print_log(log=log)

        print_log(f"Loss: {criterion._get_name()}", log=log)
        print_log(log=log)

        resume_state = None
        if args.resume:
            if os.path.exists(resume_path):
                resume_state = load_resume_state(resume_path, dataset, log=log)
            else:
                print_log(
                    f"Состояние {resume_path} не найдено, начинаю с нуля.",
                    log=log,
                )
        elif os.path.exists(resume_path):
            print_log(
                f"ВНИМАНИЕ: {resume_path} существует, но флаг --resume не задан "
                "-- обучение начнётся с нуля и перезапишет это состояние.",
                log=log,
            )

        if not args.no_resume_save:
            print_log(f"Состояние для возобновления: {resume_path}\n", log=log)

        model, history = train(
            model=model,
            trainset_loader=trainset_loader,
            valset_loader=valset_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            clip_grad=cfg.get("clip_grad"),
            max_epochs=cfg.get("max_epochs", 200),
            early_stop=cfg.get("early_stop", 10),
            verbose=1,
            log=log,
            save=save_path,
            history_path=history_path,
            plot_path=plot_path,
            resume_path=None if args.no_resume_save else resume_path,
            resume_state=resume_state,
            dataset=dataset,
            config=cfg,
            steps_per_day=cfg["model_args"].get("steps_per_day"),
        )

        print_log(f"Saved Model: {save_path}", log=log)
        print_log(f"History CSV: {history_path}", log=log)
        print_log(f"Loss plot: {plot_path}", log=log)

        test_model(
            model=model,
            testset_loader=testset_loader,
            log=log,
            steps_per_day=cfg["model_args"].get("steps_per_day"),
        )

    except KeyboardInterrupt:
        message = "Training interrupted by the user."
        print_log(message, log=log)
        raise

    except Exception:
        error_text = traceback.format_exc()
        print_log("--------- ERROR ---------", log=log)
        print_log(error_text, log=log)
        raise

    finally:
        log.close()
        if IS_DISTRIBUTED:
            # try/except: если один ранг падает, пока другие сидят в
            # коллективе, abort коммуникатора может подвиснуть сам. Ронять
            # процесс на выходе из-за этого незачем -- torchrun всё равно
            # прибьёт остальных.
            try:
                torch.distributed.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
