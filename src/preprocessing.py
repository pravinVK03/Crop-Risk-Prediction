import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORICAL, FEATURES, NUMERICAL, TARGET


class TabularPreprocessor:
    def __init__(self):
        self.category_maps: dict[str, dict[str, int]] = {}
        self.target_to_index: dict[int, int] = {}
        self.index_to_target: dict[int, int] = {}
        self.numeric_medians: dict[str, float] = {}
        self.location_profiles: dict[tuple[str, str], dict[str, float]] = {}
        self.state_profiles: dict[str, dict[str, float]] = {}
        self.season_by_loc_crop: dict[tuple[str, str, str], str] = {}
        self.season_by_state_crop: dict[tuple[str, str], str] = {}
        self.season_by_crop: dict[str, str] = {}
        self.global_season_mode: str = "unknown"

    def fit(self, frame: pd.DataFrame) -> None:
        for column in CATEGORICAL:
            values = sorted(frame[column].astype(str).str.lower().unique().tolist())
            # Keep 0 for unknown categories from API payloads.
            self.category_maps[column] = {value: idx + 1 for idx, value in enumerate(values)}

        self.numeric_medians = {column: float(frame[column].median()) for column in NUMERICAL}

        grouped = frame.groupby(["state", "district"], dropna=False)[NUMERICAL].median()
        for (state, district), row in grouped.iterrows():
            self.location_profiles[(str(state).lower(), str(district).lower())] = {
                col: float(row[col]) for col in NUMERICAL
            }

        state_grouped = frame.groupby("state", dropna=False)[NUMERICAL].median()
        for state, row in state_grouped.iterrows():
            self.state_profiles[str(state).lower()] = {col: float(row[col]) for col in NUMERICAL}

        loc_crop_mode = (
            frame.groupby(["state", "district", "crop"], dropna=False)["season"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
        )
        for (state, district, crop), season in loc_crop_mode.items():
            self.season_by_loc_crop[
                (str(state).lower(), str(district).lower(), str(crop).lower())
            ] = str(season).lower()

        state_crop_mode = (
            frame.groupby(["state", "crop"], dropna=False)["season"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
        )
        for (state, crop), season in state_crop_mode.items():
            self.season_by_state_crop[(str(state).lower(), str(crop).lower())] = str(season).lower()

        crop_mode = frame.groupby("crop", dropna=False)["season"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"
        )
        for crop, season in crop_mode.items():
            self.season_by_crop[str(crop).lower()] = str(season).lower()

        global_mode = frame["season"].mode()
        self.global_season_mode = str(global_mode.iloc[0]).lower() if not global_mode.empty else "unknown"

        target_values = sorted(frame[TARGET].astype(int).unique().tolist())
        self.target_to_index = {value: idx for idx, value in enumerate(target_values)}
        self.index_to_target = {idx: value for value, idx in self.target_to_index.items()}

    def transform(self, frame: pd.DataFrame, with_target: bool = True):
        encoded = frame.copy()

        for column in CATEGORICAL:
            mapping = self.category_maps[column]
            encoded[column] = (
                encoded[column].astype(str).str.lower().map(mapping).fillna(0).astype(np.int64)
            )

        for column in NUMERICAL:
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(self.numeric_medians[column])

        x = encoded[FEATURES].values.astype(np.float32)

        if not with_target:
            return x, None

        y = encoded[TARGET].astype(int).map(self.target_to_index).values.astype(np.int64)
        return x, y

    def get_cat_dims(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in CATEGORICAL]

    def _fallback_profile(self, state: str, district: str) -> dict[str, float]:
        key = (state, district)
        if key in self.location_profiles:
            return self.location_profiles[key]
        if state in self.state_profiles:
            return self.state_profiles[state]
        return self.numeric_medians

    def infer_season(self, state: str, district: str, crop: str) -> str:
        key_1 = (state, district, crop)
        key_2 = (state, crop)
        if key_1 in self.season_by_loc_crop:
            return self.season_by_loc_crop[key_1]
        if key_2 in self.season_by_state_crop:
            return self.season_by_state_crop[key_2]
        if crop in self.season_by_crop:
            return self.season_by_crop[crop]
        return self.global_season_mode

    def api_payload_to_frame(self, payload: dict) -> pd.DataFrame:
        location = payload.get("location", {})
        state = str(location.get("state", payload.get("state", "unknown"))).strip().lower()
        district = str(location.get("district", payload.get("district", "unknown"))).strip().lower()
        crop = str(payload.get("crop_type", payload.get("crop", "unknown"))).strip().lower()
        season = str(payload.get("season", "")).strip().lower()
        if not season:
            season = self.infer_season(state=state, district=district, crop=crop)
        soil = payload.get("soil", {})

        profile = self._fallback_profile(state=state, district=district)

        row = {
            "state": state,
            "district": district,
            "season": season,
            "crop": crop,
            "year": payload.get("year", datetime.now().year),
            "area": payload.get("area", profile["area"]),
            "production": payload.get("production", profile["production"]),
            "yield_log": payload.get("yield_log", profile["yield_log"]),
            "n": soil.get("n", payload.get("n", profile["n"])),
            "p": soil.get("p", payload.get("p", profile["p"])),
            "k": soil.get("k", payload.get("k", profile["k"])),
            "ph": soil.get("ph", payload.get("ph", profile["ph"])),
            "soil_fertility": soil.get(
                "soil_fertility",
                payload.get("soil_fertility", profile["soil_fertility"]),
            ),
        }

        return pd.DataFrame([row])

    def save(self, path: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str) -> "TabularPreprocessor":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def encode_data(frame: pd.DataFrame):
    preprocessor = TabularPreprocessor()
    preprocessor.fit(frame)
    encoded_x, encoded_y = preprocessor.transform(frame, with_target=True)
    return encoded_x, encoded_y, preprocessor
