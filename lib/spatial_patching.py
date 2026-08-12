"""
Пространственный патчинг по leaf-KDTree.

Идея из PatchSTG (Fang et al., "Efficient Large-Scale Traffic Forecasting with
Transformers: A Spatial Data Management Perspective", KDD 2025, arXiv:2412.09972).

Зачем. Плотное внимание по узлам стоит O(N^2). На CA это 8600^2 = 74M элементов
матрицы скоров на каждый слой и каждый шаг времени. Патчинг режет узлы на R
непересекающихся патчей по P точек и заменяет одно внимание по N двумя дешёвыми:
depth (внутри патча) и breadth (между патчами при фиксированном индексе).
Стоимость становится O(M * (P + R)) вместо O(N^2), где M = R * P >= N.

Ключевое требование: в одном патче должны лежать географически близкие датчики,
иначе depth-внимание перемешивает несвязанные точки. Для этого строится KDTree
по широте и долготе, а его BFS-обход даёт нужную перестановку узлов.

Отличие leaf-KDTree от обычного: в обычном точка-гиперплоскость остаётся во
внутреннем узле и в лист не попадает, из-за чего в порядке обхода рядом
оказываются несвязанные точки. Здесь дерево всегда делит множество ровно
пополам, и каждая точка гарантированно доходит до листа.

Все индексы считаются один раз офлайн по meta-файлу с колонками Lat/Lng и
кэшируются в .npy рядом с ним.
"""

import os

import numpy as np
import pandas as pd

__all__ = ["build_patch_index", "load_patch_index", "suggest_leaf_size"]


# --------------------------------------------------------------------------- #
# Построение дерева
# --------------------------------------------------------------------------- #


def _build_leaves(index, coords, depth, axis=0):
    """Рекурсивно делит множество точек пополам по медиане, чередуя оси.

    Возвращает список из 2**depth массивов с индексами узлов, слева направо.
    Соседние листья гарантированно принадлежат общему поддереву, что и нужно
    для последующей склейки в патчи.

    Деление ровно пополам (mid = len // 2) означает, что точка-гиперплоскость
    не задерживается во внутреннем узле, а уходит в правого потомка. Это и есть
    "leaf" в leaf-KDTree. Размеры листьев при этом отличаются не более чем на 1.
    """
    if depth == 0:
        return [index]

    order = np.argsort(coords[index, axis], kind="stable")
    index = index[order]
    mid = len(index) // 2
    next_axis = 1 - axis

    return _build_leaves(index[:mid], coords, depth - 1, next_axis) + _build_leaves(
        index[mid:], coords, depth - 1, next_axis
    )


def suggest_leaf_size(num_nodes, recur):
    """Минимальная вместимость листа при заданной глубине дерева.

    При делении пополам размеры листьев отличаются максимум на единицу, поэтому
    достаточно ceil(N / 2**recur).
    """
    n_leaves = 2**recur
    return int(np.ceil(num_nodes / n_leaves))


# --------------------------------------------------------------------------- #
# Паддинг неполных листьев
# --------------------------------------------------------------------------- #


def _validate(index):
    """Проверяет, что индексы образуют корректную раскладку.

    Вызывается и после построения, и после чтения кэша: устаревший или битый
    .npz иначе прошёл бы дальше молча, а обнаружилось бы это только по
    необъяснимо просевшему качеству.
    """
    gather, unpad = index["gather_idx"], index["unpad_idx"]
    num_nodes, num_slots = int(index["num_nodes"]), int(index["num_slots"])
    rows, cols = int(index["num_patches"]), int(index["patch_size"])

    if len(unpad) != num_nodes or len(gather) != num_slots:
        raise RuntimeError("длины индексов не совпадают с числом узлов/слотов")
    if rows * cols != num_slots:
        raise RuntimeError(f"R*P = {rows * cols} != числа слотов {num_slots}")
    if num_slots < num_nodes:
        raise RuntimeError(f"слотов {num_slots} меньше, чем узлов {num_nodes}")
    if not np.array_equal(gather[unpad], np.arange(num_nodes)):
        raise RuntimeError("gather/unpad не образуют перестановку узлов")
    if len(np.unique(gather)) != num_nodes:
        raise RuntimeError("не каждый узел присутствует в слотах")
    return index


def _pick_donors(scores, exclude, count):
    """count лучших кандидатов по scores (больше = лучше), исключая exclude."""
    scores = scores.copy()
    scores[exclude] = -np.inf
    available = np.isfinite(scores).sum()
    if available < count:
        raise ValueError(f"нужно {count} доноров, доступно {available}")
    return np.argpartition(-scores, count)[:count]


def _fill_patches(leaves, factors, leaf_size, profile_of, score_against):
    """Добивает неполные листья донорами и раскладывает всё по слотам.

    Донор нужен, чтобы патч был заполнен целиком. Его собственный выход на
    unpad отбрасывается, но во время depth-внимания он служит контекстом для
    настоящих точек патча -- поэтому донор должен быть близким.

    Исключаем кандидатов по всему патчу, а не только по листу. Depth-внимание
    работает сразу по всем P слотам патча, так что попади узел в один патч
    дважды -- он получил бы двойной вес в softmax просто из-за паддинга.

    profile_of(members)      -> вектор-профиль патча
    score_against(profile)   -> (num_nodes,) оценка близости, больше = лучше
    """
    n_leaves = len(leaves)
    slots = np.full((n_leaves, leaf_size), -1, np.int64)

    for i, leaf in enumerate(leaves):
        slots[i, : len(leaf)] = leaf

    for lo in range(0, n_leaves, factors):
        hi = lo + factors
        holes = [
            (i, j)
            for i in range(lo, hi)
            for j in range(len(leaves[i]), leaf_size)
        ]
        if not holes:
            continue

        members = np.concatenate([leaves[i] for i in range(lo, hi)])
        donors = _pick_donors(score_against(profile_of(members)), members, len(holes))
        for (i, j), donor in zip(holes, donors):
            slots[i, j] = donor

    if (slots < 0).any():
        raise RuntimeError("остались незаполненные слоты")
    return slots


# --------------------------------------------------------------------------- #
# Публичный интерфейс
# --------------------------------------------------------------------------- #


def build_patch_index(
    meta_path,
    recur,
    factors,
    leaf_size=None,
    series=None,
    lat_col="Lat",
    lng_col="Lng",
):
    """Считает индексы патчинга по meta-файлу с координатами датчиков.

    Параметры
    ---------
    recur : глубина KDTree. Число листьев равно 2**recur.
    factors : сколько соседних листьев склеивается в один патч. Только степень
        двойки: общее поддерево есть только у 2**k подряд идущих листьев.
    leaf_size : вместимость листа. По умолчанию минимально возможная.
    series : (num_nodes, length), опционально. Если передана, доноры для
        паддинга выбираются по косинусной близости рядов, как в статье.
        Иначе -- по географическому расстоянию.

    Возвращает словарь:
        gather_idx : (M,)  какой узел лежит в каком слоте
        unpad_idx  : (N,)  из какого слота забирать узел обратно
        num_patches R, patch_size P, M = R * P >= N
    """
    if factors & (factors - 1) != 0:
        raise ValueError(f"factors должен быть степенью двойки, получено {factors}")

    meta = pd.read_csv(meta_path)
    for col in (lat_col, lng_col):
        if col not in meta.columns:
            raise ValueError(f"В {meta_path} нет колонки {col!r}")

    coords = meta[[lat_col, lng_col]].to_numpy(np.float64)
    num_nodes = len(coords)

    n_leaves = 2**recur
    if n_leaves > num_nodes:
        raise ValueError(
            f"2**recur = {n_leaves} листьев на {num_nodes} узлов: пустые листья"
        )
    if factors > n_leaves:
        raise ValueError(f"factors={factors} больше числа листьев {n_leaves}")

    leaves = _build_leaves(np.arange(num_nodes), coords, recur)

    required = max(len(leaf) for leaf in leaves)
    if leaf_size is None:
        leaf_size = required
    elif leaf_size < required:
        raise ValueError(
            f"leaf_size={leaf_size} мал: самый полный лист содержит {required} точек"
        )

    # Обратное отображение строим до паддинга: узел владеет тем слотом, в
    # который его положило дерево, а не своими донорскими копиями.
    unpad_idx = np.full(num_nodes, -1, np.int64)
    for i, leaf in enumerate(leaves):
        unpad_idx[leaf] = i * leaf_size + np.arange(len(leaf))
    if (unpad_idx < 0).any():
        raise RuntimeError("не все узлы получили слот")

    if series is not None and len(series) != num_nodes:
        raise ValueError(
            f"series на {len(series)} узлов, meta на {num_nodes}"
        )

    if series is not None:
        series = np.asarray(series, dtype=np.float64)
        norm = series / (np.linalg.norm(series, axis=1, keepdims=True) + 1e-8)
        profile_of = lambda members: norm[members].mean(axis=0)
        score_against = lambda profile: norm @ profile
    else:
        profile_of = lambda members: coords[members].mean(axis=0)
        score_against = lambda profile: -np.sum((coords - profile) ** 2, axis=1)

    # Слоты в порядке обхода дерева. Плоский вид (M,) при reshape в (R, P)
    # склеивает factors подряд идущих листьев в один патч -- ровно то, чего
    # добивается backtracking в статье.
    slots = _fill_patches(leaves, factors, leaf_size, profile_of, score_against)
    gather_idx = slots.reshape(-1)

    patch_size = leaf_size * factors
    num_patches = n_leaves // factors

    return _validate({
        "gather_idx": gather_idx,
        "unpad_idx": unpad_idx,
        "num_patches": num_patches,
        "patch_size": patch_size,
        "num_slots": len(gather_idx),
        "num_nodes": num_nodes,
        "leaf_size": leaf_size,
        "recur": recur,
        "factors": factors,
    })


def load_patch_index(meta_path, recur, factors, leaf_size=None, cache_dir=None, **kw):
    """build_patch_index с кэшированием в .npz.

    Дерево строится за секунды, но индексы должны быть побитово одинаковыми
    между прогонами, поэтому кэш заодно служит и фиксацией.
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(os.path.abspath(meta_path))

    # Имя meta-файла обязано входить в тег: в одной директории могут лежать
    # разные meta с одинаковым числом узлов, и без этого смена meta в конфиге
    # молча подхватила бы старую перестановку от прежнего файла.
    stem = os.path.splitext(os.path.basename(meta_path))[0]
    tag = f"patchidx_{stem}_r{recur}_f{factors}_l{leaf_size or 'auto'}"
    if kw.get("series") is not None:
        tag += "_sim"
    cache_path = os.path.join(cache_dir, tag + ".npz")

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        index = {k: (v.item() if v.ndim == 0 else v) for k, v in cached.items()}
        _validate(index)
        return index

    index = build_patch_index(meta_path, recur, factors, leaf_size, **kw)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, **index)
    return index
