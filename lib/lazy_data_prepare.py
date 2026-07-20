import os

import numpy as np
import torch

from .utils import print_log, StandardScaler


class WindowDataset(torch.utils.data.Dataset):
    """
    Ленивый датасет.

    Не создаёт все временные окна заранее.
    Каждое окно формируется только при обращении к нему.
    """

    def __init__(self, data, indices, features, scaler):
        self.data = data
        self.indices = indices.astype(np.int64, copy=False)
        self.features = features
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start, x_end, y_end = self.indices[index]

        x = self.data[start:x_end][..., self.features].copy()
        y = self.data[x_end:y_end, ..., :1].copy()

        # Нормализуем только транспортный поток.
        # Время суток и день недели не меняем.
        x[..., 0] = self.scaler.transform(x[..., 0])

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
        )


def calculate_train_scaler(
    data,
    train_index,
    chunk_size=512,
):
    """
    Считает mean и std по обучающим входным окнам,
    не загружая все окна в RAM одновременно.
    """

    total_sum = 0.0
    total_squared_sum = 0.0
    total_count = 0

    for chunk_start in range(
        0,
        len(train_index),
        chunk_size,
    ):
        chunk = train_index[
            chunk_start:chunk_start + chunk_size
        ]

        input_length = int(
            chunk[0, 1] - chunk[0, 0]
        )

        positions = (
            chunk[:, 0, None]
            + np.arange(input_length)[None, :]
        )

        values = data[positions, ..., 0]

        total_sum += values.sum(dtype=np.float64)

        total_squared_sum += np.square(
            values,
            dtype=np.float64,
        ).sum(dtype=np.float64)

        total_count += values.size

    mean = total_sum / total_count

    variance = (
        total_squared_sum / total_count
        - mean**2
    )

    std = np.sqrt(max(variance, 0.0))

    if not np.isfinite(mean):
        raise ValueError(
            f"Некорректное среднее: {mean}"
        )

    if not np.isfinite(std):
        raise ValueError(
            f"Некорректное стандартное отклонение: {std}"
        )

    if std == 0:
        raise ValueError(
            "Стандартное отклонение равно нулю"
        )

    return StandardScaler(
        mean=mean,
        std=std,
    )


def get_dataloaders_from_index_data(
    data_dir,
    tod=False,
    dow=False,
    dom=False,
    batch_size=64,
    log=None,
):
    data_path = os.path.join(
        data_dir,
        "data.npz",
    )

    index_path = os.path.join(
        data_dir,
        "index.npz",
    )

    data = np.load(data_path)["data"].astype(
        np.float32,
        copy=False,
    )

    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"В data.npz найдены "
            f"NaN={nan_count}, Inf={inf_count}"
        )

    # Первый канал — traffic flow.
    features = [0]

    if tod:
        features.append(1)

    if dow:
        features.append(2)

    index_data = np.load(index_path)

    train_index = index_data["train"]
    val_index = index_data["val"]
    test_index = index_data["test"]

    scaler = calculate_train_scaler(
        data=data,
        train_index=train_index,
    )

    trainset = WindowDataset(
        data=data,
        indices=train_index,
        features=features,
        scaler=scaler,
    )

    valset = WindowDataset(
        data=data,
        indices=val_index,
        features=features,
        scaler=scaler,
    )

    testset = WindowDataset(
        data=data,
        indices=test_index,
        features=features,
        scaler=scaler,
    )

    print_log(
        f"Trainset: {len(trainset)} samples",
        log=log,
    )

    print_log(
        f"Valset:   {len(valset)} samples",
        log=log,
    )

    print_log(
        f"Testset:  {len(testset)} samples",
        log=log,
    )

    print_log(
        f"Scaler: mean={scaler.mean:.6f}, "
        f"std={scaler.std:.6f}",
        log=log,
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }

    trainset_loader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        **loader_args,
    )

    valset_loader = torch.utils.data.DataLoader(
        valset,
        shuffle=False,
        **loader_args,
    )

    testset_loader = torch.utils.data.DataLoader(
        testset,
        shuffle=False,
        **loader_args,
    )

    return (
        trainset_loader,
        valset_loader,
        testset_loader,
        scaler,
    )
