import pandas as pd

from src.config import CATEGORICAL, DATA_PATH, NUMERICAL, TARGET


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]

    for column in CATEGORICAL:
        frame[column] = frame[column].astype(str).str.strip().str.lower()

    for column in NUMERICAL:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)
    return frame
