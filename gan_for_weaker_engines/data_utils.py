from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_CSV = PROJECT_ROOT / "weaker_engine_latency_baseline.csv"
FEATURES = ["latency_ms"]


def resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_baseline_data(path: Path | str = BASELINE_CSV) -> pd.DataFrame:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).copy()
    for col in ["source", "device_key", "hw_name", "model_key", "measurement_type", "source_url", "notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df[df["latency_ms"].notna()].copy()
    df["combo_key"] = df["device_key"] + "::" + df["model_key"]
    combos = sorted(df["combo_key"].unique().tolist())
    combo_to_idx = {combo: idx for idx, combo in enumerate(combos)}
    df["combo_idx"] = df["combo_key"].map(combo_to_idx).astype(int)
    return df


def _sample_positive(mean: float, std: float, n: int, rng: np.random.Generator) -> np.ndarray:
    floor = max(mean * 0.05, 1e-6)
    return np.maximum(floor, rng.normal(mean, std, n))


def build_training_dataframe(
    csv_path: Path | str = BASELINE_CSV,
    rows_per_combo: int = 64,
    random_seed: int = 42,
    default_std_pct: float = 0.08,
) -> pd.DataFrame:
    baseline = load_baseline_data(csv_path)
    rng = np.random.default_rng(random_seed)
    rows = []

    for combo_key, combo_df in baseline.groupby("combo_key", sort=True):
        latency = combo_df["latency_ms"].astype(float)
        mean = float(latency.mean())
        if len(latency) >= 2:
            std = float(latency.std(ddof=1))
        else:
            std = 0.0
        std = max(std, mean * default_std_pct, 0.05)
        samples = _sample_positive(mean, std, rows_per_combo, rng)
        combo_idx = int(combo_df["combo_idx"].iloc[0])
        device_key = str(combo_df["device_key"].iloc[0])
        model_key = str(combo_df["model_key"].iloc[0])
        hw_name = str(combo_df["hw_name"].iloc[0])

        for latency_ms in samples:
            rows.append({
                "combo_key": combo_key,
                "combo_idx": combo_idx,
                "device_key": device_key,
                "model_key": model_key,
                "hw_name": hw_name,
                "latency_ms": float(latency_ms),
                "latency_seed_mean_ms": mean,
                "latency_seed_std_ms": std,
                "source_row_count": int(len(combo_df)),
            })

    out = pd.DataFrame(rows)
    print(
        f"[data] Grounded weaker-engine dataset: {len(out):,} rows "
        f"({out['combo_key'].nunique()} combos × {rows_per_combo} rows)"
    )
    return out


class LatencyScaler:
    Z_CLIP = 3.0

    def __init__(self):
        self._scaler = MinMaxScaler()
        self._combo_means: dict[int, float] = {}
        self._combo_stds: dict[int, float] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        grouped = df.groupby("combo_idx")["latency_ms"].agg(["mean", "std"]).reset_index()
        for row in grouped.itertuples(index=False):
            combo_idx = int(row.combo_idx)
            mean = float(row.mean)
            std = float(row.std) if pd.notna(row.std) else 0.0
            self._combo_means[combo_idx] = mean
            self._combo_stds[combo_idx] = max(std, mean * 0.05, 1e-6)

        z = self._residualise(
            df["latency_ms"].values.astype(np.float64),
            df["combo_idx"].values.astype(int),
        )
        self._scaler.fit(self._to_unit_interval(z).reshape(-1, 1))
        self.fitted = True
        return self

    def _residualise(self, values: np.ndarray, combo_idx: np.ndarray) -> np.ndarray:
        means = np.array([self._combo_means[int(idx)] for idx in combo_idx], dtype=np.float64)
        stds = np.array([self._combo_stds[int(idx)] for idx in combo_idx], dtype=np.float64)
        z = (values - means) / stds
        return np.clip(z, -self.Z_CLIP, self.Z_CLIP)

    def _to_unit_interval(self, z: np.ndarray) -> np.ndarray:
        return (z + self.Z_CLIP) / (2.0 * self.Z_CLIP)

    def _from_unit_interval(self, values: np.ndarray) -> np.ndarray:
        return values * (2.0 * self.Z_CLIP) - self.Z_CLIP

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        z = self._residualise(
            df["latency_ms"].values.astype(np.float64),
            df["combo_idx"].values.astype(int),
        )
        return self._scaler.transform(self._to_unit_interval(z).reshape(-1, 1)).astype(np.float32)

    def inverse_transform(self, arr: np.ndarray, combo_idx: np.ndarray) -> np.ndarray:
        unit = self._scaler.inverse_transform(np.asarray(arr, dtype=np.float64).reshape(-1, 1)).reshape(-1)
        z = self._from_unit_interval(unit)
        means = np.array([self._combo_means[int(idx)] for idx in combo_idx], dtype=np.float64)
        stds = np.array([self._combo_stds[int(idx)] for idx in combo_idx], dtype=np.float64)
        return np.maximum(means + z * stds, 1e-6)


def one_hot_condition(combo_idx: np.ndarray, num_combos: int) -> np.ndarray:
    cond = np.zeros((len(combo_idx), num_combos), dtype=np.float32)
    cond[np.arange(len(combo_idx)), combo_idx] = 1.0
    return cond


class LatencyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: LatencyScaler, num_combos: int):
        self.x = torch.from_numpy(scaler.transform(df))
        self.cond = torch.from_numpy(
            one_hot_condition(df["combo_idx"].values.astype(int), num_combos)
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.cond[idx]
