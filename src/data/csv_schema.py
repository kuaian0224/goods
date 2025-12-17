from typing import Dict, List

import pandas as pd


def _match_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return ""


def infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    id_col = _match_column(df, ["id"])
    label_col = _match_column(df, ["categories", "label"])
    text_cols: List[str] = []
    for name in ["title", "description", "text"]:
        matched = _match_column(df, [name])
        if matched and matched not in text_cols:
            text_cols.append(matched)
    return {"id_col": id_col, "label_col": label_col, "text_cols": text_cols}


def build_text(row: pd.Series, text_cols: List[str]) -> str:
    parts: List[str] = []
    for col in text_cols:
        if col not in row:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        text = str(val).strip()
        if text:
            parts.append(text)
    return " ".join(parts)
