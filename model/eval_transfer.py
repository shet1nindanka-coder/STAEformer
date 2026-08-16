"""Прогон чекпойнта, обученного на одном датасете, по тестовой части другого.

Мотивация: SD, GBA и GLA -- подмножества CA (LargeST). Колонка ID2 в
<dataset>_meta.csv это позиция узла в полном массиве CA, поэтому модель,
обученная на CA, переносится на подмножество БЕЗ переобучения: единственный
параметр, зависящий от числа узлов, -- adaptive_embedding формы
(in_steps, num_nodes, dim) -- просто режется по нужным строкам.

Всё остальное (input_proj, обе стопки внимания, depth/breadth патчинга,
output_proj) по узлам общее и переносится как есть. Слои внимания не имеют
параметров, зависящих от длины последовательности, поэтому смена геометрии
патчинга (у CA R=128/P=72, у SD R=16/P=64) им безразлична.

Пример:
    python model/eval_transfer.py \
        --ckpt saved_models/STAEformer-CA-2026-08-01-12-00-00.pt \
        --src CA --dst SD \
        --src-mean 61.234567 --src-std 42.345678

Значения --src-mean/--src-std берутся из лога обучения CA, строка
"Scaler: mean=..., std=...". Они ОБЯЗАТЕЛЬНЫ при --scaler src: модель училась
на данных, нормированных статистиками CA, и подавать ей SD-нормировку -- это
менять вход под ней молча.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_DIR)

import model.train as train_mod  # noqa: E402
from lib.lazy_data_prepare import get_dataloaders_from_index_data  # noqa: E402
from lib.spatial_patching import load_patch_index  # noqa: E402
from lib.utils import StandardScaler, print_log  # noqa: E402
from model.STAEformer import STAEformer  # noqa: E402


def strip_prefixes(state):
    """Снимает обёртки torch.compile (_orig_mod.) и DDP (module.).

    CA обучается с compile: default, поэтому ключи в чекпойнте могут быть
    префиксованы. Без этого load_state_dict молча не найдёт ни одного веса.
    """
    out = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod."):
            while k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = v
    return out


def load_cfg(dataset):
    with open(os.path.join(MODEL_DIR, "STAEformer.yaml"), encoding="utf-8") as f:
        all_config = yaml.safe_load(f)
    if dataset not in all_config:
        raise KeyError(f"Датасет {dataset!r} не найден в STAEformer.yaml")
    return all_config[dataset]


def build_model(dataset, cfg, data_path, device):
    model_args = dict(cfg["model_args"])

    patch_cfg = cfg.get("patching")
    if patch_cfg:
        meta_path = patch_cfg["meta"]
        if not os.path.isabs(meta_path):
            meta_path = os.path.join(data_path, meta_path)
        model_args["patch_index"] = load_patch_index(
            meta_path,
            recur=patch_cfg["recur"],
            factors=patch_cfg["factors"],
            leaf_size=patch_cfg.get("leaf_size"),
        )

    model_args["use_checkpoint"] = False
    return STAEformer(**model_args).to(device), bool(patch_cfg)


def node_rows(dst, dst_cfg, src_num_nodes, id2_base, log=None):
    """Строки исходного (большого) датасета, соответствующие узлам целевого.

    Порядок строк ОБЯЗАН совпадать с порядком узлов в data.npz целевого
    датасета -- то есть с порядком строк meta-файла. Тот же файл используется
    для построения патчей, так что согласованность проверяется автоматически:
    длина обязана совпасть с num_nodes.
    """
    meta_name = (dst_cfg.get("patching") or {}).get("meta") or f"{dst.lower()}_meta.csv"
    meta_path = os.path.join(REPO_DIR, "data", dst, meta_name)
    meta = pd.read_csv(meta_path)

    if "ID2" not in meta.columns:
        raise KeyError(
            f"В {meta_path} нет колонки ID2 -- без неё соответствие узлов "
            f"{dst} строкам исходного датасета не восстановить."
        )

    rows = meta["ID2"].to_numpy(np.int64) - id2_base

    n = dst_cfg["model_args"]["num_nodes"]
    if len(rows) != n:
        raise ValueError(f"{meta_path}: {len(rows)} строк, а узлов у {dst} {n}")
    if rows.min() < 0 or rows.max() >= src_num_nodes:
        raise ValueError(
            f"ID2 выходит за границы исходного датасета: диапазон "
            f"[{rows.min()}, {rows.max()}] при {src_num_nodes} узлах. "
            f"Похоже, --id2-base подобран неверно."
        )
    if len(np.unique(rows)) != len(rows):
        raise ValueError("ID2 содержит повторы")

    print_log(
        f"Узлы {dst} -> строки {os.path.basename(meta_path)}: "
        f"[{rows.min()}, {rows.max()}], "
        f"{'непрерывный блок' if np.array_equal(rows, np.arange(rows.min(), rows.min() + len(rows))) else 'разрозненные'}",
        log=log,
    )
    return rows


def transfer_state(ckpt_state, model, rows, log=None):
    """Кладёт веса исходной модели в целевую, срезая adaptive_embedding."""
    state = strip_prefixes(ckpt_state)
    target = model.state_dict()

    # Буферы патчинга принадлежат геометрии ЦЕЛЕВОГО датасета, а не весам.
    # В state_dict они лежат как обычные записи, поэтому их надо выкинуть,
    # иначе load_state_dict уронит прогон на несовпадении форм.
    for key in ("gather_idx", "unpad_idx"):
        state.pop(key, None)

    if "adaptive_embedding" in state:
        src = state["adaptive_embedding"]
        want = target["adaptive_embedding"].shape
        if src.shape[1] < rows.max() + 1:
            raise ValueError(
                f"adaptive_embedding в чекпойнте на {src.shape[1]} узлов, "
                f"а максимальный индекс среза {rows.max()}"
            )
        state["adaptive_embedding"] = src[:, torch.as_tensor(rows), :].clone()
        print_log(
            f"adaptive_embedding: {tuple(src.shape)} -> "
            f"{tuple(state['adaptive_embedding'].shape)} (нужно {tuple(want)})",
            log=log,
        )

    if "node_emb" in state:
        state["node_emb"] = state["node_emb"][torch.as_tensor(rows)].clone()

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in ("gather_idx", "unpad_idx")]

    if missing:
        raise RuntimeError(
            "В чекпойнте нет весов для: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
            + "\nСкорее всего, исходная модель обучена без патчинга, а целевая "
            "конфигурация его включает (или наоборот)."
        )
    if unexpected:
        raise RuntimeError("Лишние ключи в чекпойнте: " + ", ".join(unexpected[:10]))

    print_log("Все веса перенесены, несовпадений нет.", log=log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="чекпойнт исходной модели")
    p.add_argument("--src", default="CA", help="датасет, на котором обучались")
    p.add_argument("--dst", default="SD", help="датасет, на котором меряем")
    p.add_argument(
        "--scaler",
        choices=("src", "dst"),
        default="src",
        help=(
            "чьей нормировкой кормить модель. src -- статистики обучения "
            "(честный перенос). dst -- статистики целевого датасета "
            "(скрытая адаптация, метрики будут лучше и несопоставимы)."
        ),
    )
    p.add_argument("--src-mean", type=float, help="mean скейлера исходного датасета")
    p.add_argument("--src-std", type=float, help="std скейлера исходного датасета")
    p.add_argument(
        "--id2-base",
        type=int,
        default=0,
        choices=(0, 1),
        help=(
            "с нуля или с единицы нумерует ID2. Ошибка здесь сдвигает ВСЕ узлы "
            "на один и портит метрики молча -- сверьтесь с --verify-data."
        ),
    )
    p.add_argument(
        "--verify-data",
        metavar="CA_DATA_NPZ",
        help=(
            "путь к data.npz исходного датасета. Если задан, скрипт сверит "
            "срез строк с data.npz целевого и упадёт при расхождении. "
            "Прогоните один раз на сервере, где лежит CA."
        ),
    )
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log", help="файл лога")
    args = p.parse_args()

    src, dst = args.src.upper(), args.dst.upper()
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    device = torch.device(args.device)

    src_cfg, dst_cfg = load_cfg(src), load_cfg(dst)
    data_path = os.path.join(REPO_DIR, "data", dst)

    print_log(f"Перенос {src} -> {dst}, устройство {device}", log=log)

    rows = node_rows(dst, dst_cfg, src_cfg["model_args"]["num_nodes"], args.id2_base, log)

    if args.verify_data:
        verify_rows(args.verify_data, os.path.join(data_path, "data.npz"), rows, log)

    model, patched = build_model(dst, dst_cfg, data_path, device)
    print_log(f"Целевая модель: патчинг {'включён' if patched else 'выключен'}", log=log)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and "optimizer" in state:
        state = state["model"]  # файл состояния для --resume, а не голые веса
    transfer_state(state, model, rows, log)

    batch_size = args.batch_size or dst_cfg.get("batch_size", 64)
    _, _, testset_loader, dst_scaler = get_dataloaders_from_index_data(
        data_path,
        tod=dst_cfg.get("time_of_day"),
        dow=dst_cfg.get("day_of_week"),
        batch_size=batch_size,
        log=log,
    )

    if args.scaler == "src":
        if args.src_mean is None or args.src_std is None:
            raise SystemExit(
                "При --scaler src нужны --src-mean и --src-std. Возьмите их из "
                f"лога обучения {src}: строка 'Scaler: mean=..., std=...'."
            )
        scaler = StandardScaler(mean=args.src_mean, std=args.src_std)
        # WindowDataset нормирует вход своим scaler, метрики разнормируются
        # глобальным train_mod.SCALER. Подменять надо ОБА, иначе прямое и
        # обратное преобразования разъедутся и метрики будут бессмысленны.
        testset_loader.dataset.scaler = scaler
        print_log(
            f"Скейлер {src}: mean={scaler.mean:.6f}, std={scaler.std:.6f} "
            f"(у {dst} было mean={dst_scaler.mean:.6f}, std={dst_scaler.std:.6f})",
            log=log,
        )
    else:
        scaler = dst_scaler
        print_log(
            f"ВНИМАНИЕ: используется скейлер {dst}. Это не чистый перенос -- "
            "модель получает вход в другой нормировке, чем при обучении.",
            log=log,
        )

    train_mod.SCALER = scaler
    train_mod.DEVICE = device
    train_mod.AMP_ENABLED = False  # инференс, экономить память незачем

    train_mod.test_model(
        model,
        testset_loader,
        log=log,
        steps_per_day=dst_cfg["model_args"].get("steps_per_day"),
    )

    if log:
        log.close()


def verify_rows(src_npz, dst_npz, rows, log=None):
    """Проверяет, что срез строк исходного датасета -- это ровно целевой.

    Единственный способ поймать ошибку в --id2-base до того, как она молча
    испортит метрики: сравнить сами данные, а не индексы.
    """
    src = np.load(src_npz, mmap_mode="r")["data"]
    dst = np.load(dst_npz, mmap_mode="r")["data"]

    # CA собран на три года (2017-2019), а SD -- на один. Совпадать обязаны
    # не длины, а ХВОСТЫ: оба ряда заканчиваются одной датой, поэтому целевой
    # датасет -- это последние len(dst) отсчётов исходного. Если предположение
    # неверно, сверка ниже это и поймает.
    offset = src.shape[0] - dst.shape[0]
    if offset < 0:
        raise ValueError(
            f"Исходный ряд короче целевого: {src.shape[0]} против {dst.shape[0]}"
        )
    if offset:
        print_log(
            f"Длины по времени разные ({src.shape[0]} против {dst.shape[0]}), "
            f"выравниваю по концу: сдвиг {offset} отсчётов "
            f"(~{offset / 288 / 365:.2f} года).",
            log=log,
        )

    # Сравниваем на выборке шагов: полный массив CA это гигабайты.
    steps = np.linspace(0, dst.shape[0] - 1, 200, dtype=np.int64)
    a = np.asarray(src[steps + offset][:, rows, 0], dtype=np.float64)
    b = np.asarray(dst[steps][:, :, 0], dtype=np.float64)

    if not np.allclose(a, b, rtol=0, atol=1e-6):
        bad = int((~np.isclose(a, b, rtol=0, atol=1e-6)).sum())
        raise ValueError(
            f"Срез не совпадает с целевым датасетом: {bad} расхождений из "
            f"{a.size}. Проверьте --id2-base и порядок строк meta-файла."
        )

    print_log(f"Сверка данных пройдена: {a.size} значений совпали.", log=log)


if __name__ == "__main__":
    main()
