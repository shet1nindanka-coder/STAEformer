import argparse
from contextlib import nullcontext
import copy
import datetime
import json
import os
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
from model.STAEformer import STAEformer  # noqa: E402


# X shape: (B, T, N, C)
DEVICE = torch.device("cpu")
SCALER = None
AMP_ENABLED = False
GRAD_SCALER = None


def _unwrap_model(model):
    """Return the underlying model when DataParallel is enabled."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _amp_context():
    """Enable FP16 autocast only when CUDA AMP is active."""
    if AMP_ENABLED:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


def _move_batch_to_device(x_batch, y_batch):
    """Move a batch to the selected device."""
    non_blocking = DEVICE.type == "cuda"
    return (
        x_batch.to(DEVICE, non_blocking=non_blocking),
        y_batch.to(DEVICE, non_blocking=non_blocking),
    )


@torch.no_grad()
def eval_model(model, valset_loader, criterion, epoch, max_epochs):
    """Calculate validation loss and show batch-level progress."""
    model.eval()
    batch_loss_list = []

    progress = tqdm(
        valset_loader,
        desc=f"Validation {epoch}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
    )

    for batch_number, (x_batch, y_batch) in enumerate(progress, start=1):
        x_batch, y_batch = _move_batch_to_device(x_batch, y_batch)

        with _amp_context():
            out_batch = model(x_batch)

        # Keep inverse scaling and the loss in FP32 for numerical stability.
        out_batch = SCALER.inverse_transform(out_batch.float())
        loss = criterion(out_batch, y_batch.float())
        batch_loss_list.append(loss.item())

        if batch_number % 20 == 0 or batch_number == len(valset_loader):
            progress.set_postfix(
                val_loss=f"{np.mean(batch_loss_list):.4f}",
            )

    if not batch_loss_list:
        raise RuntimeError("Validation loader is empty.")

    return float(np.mean(batch_loss_list))


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

    progress = tqdm(
        trainset_loader,
        desc=f"Training {epoch}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
    )

    for batch_number, (x_batch, y_batch) in enumerate(progress, start=1):
        x_batch, y_batch = _move_batch_to_device(x_batch, y_batch)

        optimizer.zero_grad(set_to_none=True)

        with _amp_context():
            out_batch = model(x_batch)

        # Keep inverse scaling and the loss in FP32 for numerical stability.
        out_batch = SCALER.inverse_transform(out_batch.float())
        loss = criterion(out_batch, y_batch.float())

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}, batch {batch_number}: "
                f"{loss.item()}"
            )

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

    epoch_loss = float(np.mean(batch_loss_list))
    scheduler.step()

    return epoch_loss


@torch.no_grad()
def calculate_streaming_metrics(model, loader, description):
    """
    Calculate masked RMSE, MAE and MAPE without storing all predictions in RAM.

    This matches the repository metrics: target values equal to zero are ignored.
    Metrics are returned for all horizons together and separately for every step.
    """
    model.eval()

    total_count = 0
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_absolute_percentage_error = 0.0

    step_count = None
    step_abs_error = None
    step_squared_error = None
    step_absolute_percentage_error = None

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
    )

    for batch_number, (x_batch, y_batch) in enumerate(progress, start=1):
        x_batch, y_batch = _move_batch_to_device(x_batch, y_batch)

        with _amp_context():
            y_pred = model(x_batch)

        y_pred = SCALER.inverse_transform(y_pred.float())
        y_batch = y_batch.float()

        mask = y_batch != 0
        abs_error = torch.abs(y_pred - y_batch)
        squared_error = torch.square(y_pred - y_batch)

        safe_target = torch.where(
            mask,
            y_batch,
            torch.ones_like(y_batch),
        )
        absolute_percentage_error = abs_error / torch.abs(safe_target)

        mask_float = mask.to(abs_error.dtype)

        total_count += int(mask.sum().item())
        total_abs_error += float((abs_error * mask_float).sum().item())
        total_squared_error += float((squared_error * mask_float).sum().item())
        total_absolute_percentage_error += float(
            (absolute_percentage_error * mask_float).sum().item()
        )

        reduce_dims = (0, 2, 3)

        batch_step_count = mask.sum(dim=reduce_dims).detach().cpu().double()
        batch_step_abs_error = (
            (abs_error * mask_float).sum(dim=reduce_dims).detach().cpu().double()
        )
        batch_step_squared_error = (
            (squared_error * mask_float)
            .sum(dim=reduce_dims)
            .detach()
            .cpu()
            .double()
        )
        batch_step_absolute_percentage_error = (
            (absolute_percentage_error * mask_float)
            .sum(dim=reduce_dims)
            .detach()
            .cpu()
            .double()
        )

        if step_count is None:
            step_count = torch.zeros_like(batch_step_count)
            step_abs_error = torch.zeros_like(batch_step_abs_error)
            step_squared_error = torch.zeros_like(batch_step_squared_error)
            step_absolute_percentage_error = torch.zeros_like(
                batch_step_absolute_percentage_error
            )

        step_count += batch_step_count
        step_abs_error += batch_step_abs_error
        step_squared_error += batch_step_squared_error
        step_absolute_percentage_error += batch_step_absolute_percentage_error

        if batch_number % 100 == 0 or batch_number == len(loader):
            progress.set_postfix(processed=batch_number)

    if total_count == 0:
        raise ValueError("No non-zero targets were found while calculating metrics.")

    all_metrics = {
        "rmse": float(np.sqrt(total_squared_error / total_count)),
        "mae": float(total_abs_error / total_count),
        "mape": float(
            100.0 * total_absolute_percentage_error / total_count
        ),
    }

    if step_count is None or torch.any(step_count == 0):
        raise ValueError(
            "At least one forecast horizon has no non-zero targets."
        )

    step_metrics = []

    for step in range(len(step_count)):
        count = float(step_count[step].item())

        step_metrics.append(
            {
                "step": step + 1,
                "rmse": float(
                    torch.sqrt(step_squared_error[step] / count).item()
                ),
                "mae": float((step_abs_error[step] / count).item()),
                "mape": float(
                    100.0
                    * (
                        step_absolute_percentage_error[step]
                        / count
                    ).item()
                ),
            }
        )

    return all_metrics, step_metrics


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
):
    """Train with early stopping, progress bars, timing and persistent history."""
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf
    best_epoch = -1
    best_state_dict = None
    history = []
    total_training_start = time.time()

    for epoch_index in range(max_epochs):
        epoch = epoch_index + 1
        epoch_start = time.time()

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

        val_loss = eval_model(
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

            if save:
                torch.save(best_state_dict, save)
        else:
            wait += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "epoch_seconds": epoch_seconds,
                "best_so_far": improved,
            }
        )

        if history_path and plot_path:
            save_training_artifacts(
                history=history,
                history_path=history_path,
                plot_path=plot_path,
            )

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

        if wait >= early_stop:
            print_log(
                f"Early stopping triggered after epoch {epoch}.",
                log=log,
            )
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without a valid model state.")

    _unwrap_model(model).load_state_dict(best_state_dict)

    total_training_seconds = time.time() - total_training_start

    print_log(
        f"Training finished in {total_training_seconds / 3600:.2f} hours.",
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
def test_model(model, testset_loader, log=None):
    """Calculate test metrics in a streaming, memory-safe manner."""
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

    for item in step_metrics:
        out_str += (
            "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n"
            % (
                item["step"],
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
    global DEVICE, SCALER, AMP_ENABLED, GRAD_SCALER

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument(
        "-g",
        "--gpu_num",
        type=str,
        default="0,1",
        help='Visible GPU IDs, for example "0,1".',
    )
    args = parser.parse_args()

    seed = 42
    seed_everything(seed)
    set_cpu_num(1)

    gpu_ids = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    DEVICE = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    AMP_ENABLED = DEVICE.type == "cuda"
    GRAD_SCALER = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_ENABLED,
    )

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

    model = STAEformer(**cfg["model_args"])
    model = model.to(DEVICE)

    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(
            model,
            device_ids=list(range(torch.cuda.device_count())),
            output_device=0,
        )

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

    log = open(log_path, "w", encoding="utf-8")

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
                f"DataParallel on {torch.cuda.device_count()} GPUs"
                if isinstance(model, nn.DataParallel)
                else "disabled"
            ),
            log=log,
        )
        print_log(
            f"AMP FP16: {'enabled' if AMP_ENABLED else 'disabled'}",
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
        )
        print_log(log=log)

        if dataset in ("METRLA", "PEMSBAY"):
            criterion = MaskedMAELoss()
        elif dataset in (
            "PEMS03",
            "PEMS04",
            "PEMS07",
            "PEMS08",
            "SD",
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

        input_feature_count = next(iter(trainset_loader))[0].shape[-1]

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
        print_log(log=log)

        print_log(f"Loss: {criterion._get_name()}", log=log)
        print_log(log=log)

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
        )

        print_log(f"Saved Model: {save_path}", log=log)
        print_log(f"History CSV: {history_path}", log=log)
        print_log(f"Loss plot: {plot_path}", log=log)

        test_model(
            model=model,
            testset_loader=testset_loader,
            log=log,
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


if __name__ == "__main__":
    main()
