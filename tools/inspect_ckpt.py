"""Опись чекпойнтов: какой датасет, патчинг или плотное внимание, размер.

Зачем: имя файла содержит только дату и датасет из конфига, а понять,
на скольких узлах модель и обучалась ли она с патчингом, можно лишь заглянув
внутрь. Ошибиться тут дорого -- перенос молча загрузит не те веса.

Torch не нужен: скрипт разбирает pickle внутри .pt своим Unpickler'ом и
читает только МЕТАДАННЫЕ тензоров (форму и dtype), не трогая сами данные.
Поэтому опись 40-гигабайтной директории занимает секунды и работает где
угодно -- на сервере, на Kaggle, на ноутбуке без CUDA.

Пример:
    python tools/inspect_ckpt.py saved_models/
"""

import argparse
import collections
import os
import pickle
import zipfile


class _Stub:
    """Заглушка вместо любого класса из torch."""

    def __init__(self, *a, **kw):
        pass


def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    """Подмена torch._utils._rebuild_tensor_v2: возвращает только форму."""
    return {"shape": tuple(size), "dtype": storage}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        # OrderedDict нужен настоящий: pickle восстанавливает его через
        # __setstate__, а у голого dict нет __dict__, и разбор падает.
        if name == "OrderedDict":
            return collections.OrderedDict
        return _Stub

    def persistent_load(self, pid):
        # ('storage', <dtype class>, key, location, numel)
        try:
            return getattr(pid[1], "__name__", "?").replace("Storage", "").lower()
        except Exception:
            return "?"


def read_state(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        with z.open(name) as f:
            return _Unpickler(f).load()


def describe(path):
    obj = read_state(path)

    # Файл возобновления (--resume) хранит веса под ключом "model".
    kind = "веса"
    if isinstance(obj, dict) and "optimizer" in obj and "model" in obj:
        kind = "resume"
        obj = obj["model"]

    if not isinstance(obj, dict):
        return {"файл": os.path.basename(path), "ошибка": "не похоже на state_dict"}

    keys = list(obj)
    ae = obj.get("adaptive_embedding")

    nodes = ae["shape"][1] if isinstance(ae, dict) and len(ae["shape"]) == 3 else None
    steps = ae["shape"][0] if isinstance(ae, dict) and len(ae["shape"]) == 3 else None

    # Число узлов однозначно определяет датасет LargeST.
    known = {716: "SD", 2352: "GBA", 3834: "GLA", 8600: "CA",
             207: "METRLA", 325: "PEMSBAY", 307: "PEMS04", 883: "PEMS07", 170: "PEMS08"}

    return {
        "файл": os.path.basename(path),
        "тип": kind,
        "узлов": nodes,
        "датасет": known.get(nodes, "?"),
        "in_steps": steps,
        "внимание": "патчинг" if any("breadth" in k for k in keys) else "плотное",
        "compile": "да" if any("_orig_mod" in k for k in keys) else "нет",
        "слоёв": sum(1 for k in keys if k.endswith("attn.FC_Q.weight")) // 2 or None,
        "МБ": round(os.path.getsize(path) / 2**20, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="файлы .pt или директории с ними")
    args = p.parse_args()

    files = []
    for path in args.paths:
        if os.path.isdir(path):
            files += [
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if f.endswith((".pt", ".pth", ".ckpt"))
            ]
        else:
            files.append(path)

    if not files:
        raise SystemExit("Чекпойнтов не найдено")

    rows = []
    for f in files:
        try:
            rows.append(describe(f))
        except Exception as exc:
            rows.append({"файл": os.path.basename(f), "ошибка": str(exc)[:60]})

    cols = ["файл", "датасет", "узлов", "внимание", "compile", "тип", "МБ", "ошибка"]
    cols = [c for c in cols if any(r.get(c) is not None for r in rows)]
    width = {c: max(len(c), max(len(str(r.get(c, "-"))) for r in rows)) for c in cols}

    print("  ".join(c.ljust(width[c]) for c in cols))
    print("  ".join("-" * width[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "-")).ljust(width[c]) for c in cols))


if __name__ == "__main__":
    main()
