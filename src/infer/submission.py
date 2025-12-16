import os
from typing import List

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


def write_submission(path: str, ids: List[str], preds: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame({"id": ids, "categories": preds})
    df.to_csv(path, index=False)
