"""
Сборка data.npz и index.npz из сырых файлов LargeST.

Заменяет ноутбук: на сервере он неудобен (ядро отваливается вместе с ssh,
а сборка CA идёт десятки минут и требует десятков гигабайт RAM).

Пример:
    python -u lib/prepare_data.py --dataset CA --years 2017 2018 2019

Что делает:
  1. берёт ca_meta.csv, фильтрует датчики по дистриктам (для CA -- все);
  2. читает ca_his_raw_<год>.h5 за каждый год и склеивает по времени;
  3. приводит к сетке 5-минутных отсчётов, пропуски -> 0;
  4. добавляет каналы времени суток и дня недели;
  5. режет на окна и раскладывает индексы по train/val/test хронологически;
  6. пишет data.npz, index.npz и <dataset>_meta.csv в data/<dataset>/.

ВАЖНО про нули: датчик, которого в каком-то году ещё не было, превращается
в колонку нулей за целый год. Метрики нули в цели маскируют, но на входе они
остаются и попадают в scaler. Скрипт печатает долю мёртвых датчиков по годам
-- если она велика, ранние годы лучше не брать.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_STEPS = 12
OUT_STEPS = 12
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
FREQ = "5min"


def find_largest_files(explicit_path=None):
    """Находит файлы LargeST: по явному пути или через kagglehub."""
    if explicit_path:
        root = explicit_path
    else:
        import kagglehub

        root = kagglehub.dataset_download("liuxu77/largest")
        print(f"Датасет скачан в: {root}")

    files = {}
    for current_dir, _, names in os.walk(root):
        for name in names:
            files[name] = os.path.join(current_dir, name)

    if "ca_meta.csv" not in files:
        raise FileNotFoundError(f"ca_meta.csv не найден в {root}")

    return files


def load_year(files, year, sensor_ids):
    """Читает год и приводит к полной 5-минутной сетке с нужными колонками."""
    key = f"ca_his_raw_{year}.h5"
    if key not in files:
        raise FileNotFoundError(f"{key} не найден; есть: {sorted(files)[:10]}")

    frame = pd.read_hdf(files[key])
    frame.columns = frame.columns.astype(str)
    frame.index = pd.to_datetime(frame.index)

    frame = frame.loc[~frame.index.duplicated(keep="first")].sort_index()

    full_index = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year + 1}-01-01",
        freq=FREQ,
        inclusive="left",
    )

    frame = frame.reindex(index=full_index, columns=sensor_ids)
    return frame.astype(np.float32)


def build_indices(num_steps):
    """Хронологический сплит окон на train / val / test."""
    val_start = int(num_steps * TRAIN_RATIO)
    test_start = int(num_steps * (TRAIN_RATIO + VAL_RATIO))

    starts = np.arange(num_steps - IN_STEPS - OUT_STEPS + 1)
    indices = np.column_stack(
        [starts, starts + IN_STEPS, starts + IN_STEPS + OUT_STEPS]
    )

    target_start, target_end = indices[:, 1], indices[:, 2]

    return (
        indices[target_end <= val_start],
        indices[(target_start >= val_start) & (target_end <= test_start)],
        indices[target_start >= test_start],
        val_start,
        test_start,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CA", help="CA, GLA, GBA, SD")
    parser.add_argument("--years", type=int, nargs="+", default=[2019])
    parser.add_argument(
        "--districts",
        type=int,
        nargs="+",
        default=None,
        help="Дистрикты; по умолчанию все (то есть полный CA).",
    )
    parser.add_argument(
        "--largest-path",
        default=None,
        help="Путь к распакованному LargeST; иначе качается через kagglehub.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только посчитать мёртвые датчики по годам и выйти.",
    )
    args = parser.parse_args()

    files = find_largest_files(args.largest_path)

    meta = pd.read_csv(files["ca_meta.csv"])
    if args.districts:
        meta = meta[meta["District"].isin(args.districts)].reset_index(drop=True)

    sensor_ids = meta["ID"].astype(str).tolist()
    num_nodes = len(sensor_ids)
    print(f"Датасет {args.dataset}: {num_nodes} датчиков, годы {args.years}")

    # --- чтение годов + отчёт по мёртвым датчикам ------------------------- #
    frames = []
    for year in sorted(args.years):
        frame = load_year(files, year, sensor_ids)
        nan_share = float(frame.isna().to_numpy().mean())
        dead = int((frame.fillna(0) == 0).all().sum())
        print(
            f"  {year}: {frame.shape[0]} отсчётов, "
            f"мёртвых датчиков {dead}/{num_nodes} ({dead / num_nodes:.1%}), "
            f"NaN {nan_share:.1%}"
        )
        frames.append(frame)

    if args.check_only:
        print("\n--check-only: сборка пропущена.")
        return

    history = pd.concat(frames, axis=0).sort_index()
    del frames

    history = history.fillna(0)
    num_steps = len(history)
    print(f"\nСклеено: {num_steps} отсчётов x {num_nodes} датчиков")

    # --- три канала: поток, время суток, день недели ---------------------- #
    data = np.empty((num_steps, num_nodes, 3), dtype=np.float32)

    data[:, :, 0] = history.to_numpy(dtype=np.float32)

    minutes = history.index.hour * 60 + history.index.minute
    # Broadcast вместо np.tile: тот же результат без лишней копии на узлы.
    data[:, :, 1] = (minutes / (24 * 60)).to_numpy(np.float32)[:, None]
    data[:, :, 2] = history.index.dayofweek.to_numpy(np.float32)[:, None]

    index_start = history.index[0]
    index_end = history.index[-1]
    del history

    if not np.isfinite(data).all():
        raise ValueError("В собранных данных остались NaN или Inf")

    # --- сплит ------------------------------------------------------------ #
    train_idx, val_idx, test_idx, val_start, test_start = build_indices(num_steps)

    print(f"Период: {index_start} .. {index_end}")
    print(f"  train: {len(train_idx):>8} окон")
    print(f"  val:   {len(val_idx):>8} окон")
    print(f"  test:  {len(test_idx):>8} окон")

    # --- запись ------------------------------------------------------------ #
    out_dir = os.path.join(REPO_DIR, "data", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, f"{args.dataset.lower()}_meta.csv")
    meta.to_csv(meta_path, index=False)

    # Без сжатия: 30 ГБ жмутся десятки минут ради экономии, которой не видно
    # на фоне 280 ГБ свободного диска, а читать несжатое заметно быстрее.
    np.savez(os.path.join(out_dir, "data.npz"), data=data)
    np.savez(
        os.path.join(out_dir, "index.npz"),
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )

    print(f"\nЗаписано в {out_dir}:")
    for name in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, name)) / 2**30
        print(f"  {name:20} {size:7.2f} ГБ")

    print(
        f"\nПроверьте, что в yaml для {args.dataset} стоит "
        f"num_nodes: {num_nodes} и patching.meta: {os.path.basename(meta_path)}"
    )


if __name__ == "__main__":
    sys.exit(main())
