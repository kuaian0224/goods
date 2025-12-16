import json
import os
from typing import Any, Dict

import pandas as pd
import yaml


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_dir(path: str) -> None:
    if path == "":
        return
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        ensure_dir(dirpath)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
