"""
Edge AI Energy GAN - Data Utilities

Builds the training dataset, handles feature scaling, and provides a PyTorch
Dataset for the GAN. The preferred path uses the real benchmark CSV plus the
ingested real-device profiles to estimate missing device/model combinations.

If the real benchmark CSV is unavailable, the module falls back to the legacy
synthetic seed generator so the project still runs end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from gan_model import DEVICES, MODELS, NUM_DEVICES, NUM_MODELS


PROJECT_ROOT = Path(__file__).resolve().parent
REAL_BENCHMARK_CSV = PROJECT_ROOT / "real_benchmark_data.csv"
REAL_DEVICE_PROFILES_JSON = PROJECT_ROOT / "real_device_profiles.json"


# ─── Hardware Profiles ────────────────────────────────────────────────────────

# These defaults are only used when real-device profile data is missing.
DEVICE_PROFILES = {
    #                         TDP   E-scale  Lat-scale
    "apple_silicon": dict(tdp=5.30, es=1.00, ls=1.00),
    "raspberry_pi4": dict(tdp=3.40, es=2.10, ls=3.80),
    "jetson_nano": dict(tdp=5.00, es=1.60, ls=2.20),
    "coral_tpu": dict(tdp=2.00, es=0.30, ls=0.40),
    "stm32": dict(tdp=0.05, es=0.80, ls=45.0),
    "snapdragon888": dict(tdp=4.50, es=0.85, ls=0.70),
}

# Paper-measured baselines on Apple Silicon
MODEL_BASELINES = {
    #                          E(J)   W(W)  lat(ms)  params(M)  GFLOPs
    "mobilenetv3_small": dict(e=0.080, w=5.3, l=15.14, p=2.54, f=0.22),
    "mobilenetv2": dict(e=0.106, w=5.3, l=20.07, p=3.50, f=0.30),
    "resnet18": dict(e=0.110, w=5.3, l=20.68, p=11.69, f=1.82),
    "tiny_vit_5m": dict(e=0.219, w=5.3, l=41.33, p=12.08, f=1.30),
    "efficientnet_b0": dict(e=0.292, w=5.3, l=55.08, p=5.29, f=0.39),
}

DEFAULT_NOISE_STD_PCT = {
    "energy_J": 0.07,
    "power_W": 0.04,
    "latency_ms": 0.08,
}
ESTIMATED_NOISE_STD_PCT = {
    "energy_J": 0.12,
    "power_W": 0.08,
    "latency_ms": 0.15,
}
LEGACY_NOISE_STD_PCT = 0.05
LEGACY_TRIALS_PER_COMBO = 100
TRAINING_ROWS_PER_COMBO = 64

FEATURE_MODES = {
    "raw": ["energy_J", "power_W", "latency_ms", "energy_std"],
    # Paper-aligned workflow treats power and repeated-trial spread as derived
    # support quantities rather than independent per-sample GAN targets.
    "paper_aligned": ["energy_J", "latency_ms"],
}


# ─── Legacy Synthetic Seed Generator ─────────────────────────────────────────

def _sample_legacy(mean: float, n: int, rng: np.random.Generator) -> np.ndarray:
    std = mean * LEGACY_NOISE_STD_PCT
    return np.maximum(mean * 0.05, rng.normal(mean, std, n))


def generate_legacy_seed_data(
    trials_per_combo: int = LEGACY_TRIALS_PER_COMBO,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Legacy fallback: generate synthetic data from static device scales plus
    Gaussian noise. This path is retained only for environments without the
    real benchmark CSV.
    """
    rng = np.random.default_rng(random_seed)
    rows = []

    for dev_name, dev in DEVICE_PROFILES.items():
        for mdl_name, mdl in MODEL_BASELINES.items():
            base_e = mdl["e"] * dev["es"]
            base_l = mdl["l"] * dev["ls"]

            e_samples = _sample_legacy(base_e, trials_per_combo, rng)
            l_samples = _sample_legacy(base_l, trials_per_combo, rng)
            w_samples = _sample_legacy(dev["tdp"], trials_per_combo, rng)
            std_e = float(np.std(e_samples))

            for e, l, w in zip(e_samples, l_samples, w_samples):
                rows.append({
                    "device": dev_name,
                    "model": mdl_name,
                    "energy_J": float(e),
                    "power_W": float(w),
                    "latency_ms": float(l),
                    "energy_std": std_e,
                    "params_M": mdl["p"],
                    "flops_G": mdl["f"],
                    "device_idx": DEVICES[dev_name],
                    "model_idx": MODELS[mdl_name],
                    "combo_source": "legacy_estimated",
                    "measurement_count": 0,
                    "source_count": 0,
                })

    df = pd.DataFrame(rows)
    print(
        f"[data] Legacy seed dataset: {len(df):,} rows "
        f"({len(DEVICES)} devices × {len(MODELS)} models × {trials_per_combo} trials)"
    )
    return df


# ─── Real Data Loading / Estimation ──────────────────────────────────────────

def load_real_device_profiles(path: Path = REAL_DEVICE_PROFILES_JSON) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("profiles", {})


def resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def parse_requested_keys(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",") if part.strip()]
        return parts or None
    return [str(item).strip().lower() for item in value if str(item).strip()]


def resolve_feature_columns(feature_mode: str) -> list[str]:
    try:
        return list(FEATURE_MODES[feature_mode])
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature_mode={feature_mode!r}. "
            f"Expected one of {sorted(FEATURE_MODES)}."
        ) from exc


def _select_keys(
    requested: list[str] | None,
    universe: dict,
    label: str,
) -> list[str]:
    if not requested:
        return list(universe.keys())
    invalid = [key for key in requested if key not in universe]
    if invalid:
        raise ValueError(f"Unknown {label}: {invalid}")
    return [key for key in universe if key in requested]


def load_real_benchmark_data(path: Path = REAL_BENCHMARK_CSV) -> pd.DataFrame:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).copy()
    for col in ["device", "model", "source"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    for col in ["hw_name", "notes", "source_url", "measurement_type"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["latency_ms", "power_W", "energy_J"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive the missing third metric when the other two are available.
    energy_missing = (
        df["energy_J"].isna() & df["power_W"].notna() & df["latency_ms"].notna()
    )
    df.loc[energy_missing, "energy_J"] = (
        df.loc[energy_missing, "power_W"] * df.loc[energy_missing, "latency_ms"] / 1000.0
    )

    power_missing = (
        df["power_W"].isna() & df["energy_J"].notna() & df["latency_ms"].notna()
    )
    df.loc[power_missing, "power_W"] = (
        df.loc[power_missing, "energy_J"] * 1000.0 / df.loc[power_missing, "latency_ms"]
    )

    latency_missing = (
        df["latency_ms"].isna() & df["energy_J"].notna() & df["power_W"].notna()
    )
    df.loc[latency_missing, "latency_ms"] = (
        df.loc[latency_missing, "energy_J"] * 1000.0 / df.loc[latency_missing, "power_W"]
    )

    df = df[df["device"].isin(DEVICES) & df["model"].isin(MODELS)].copy()
    df = df[
        df[["energy_J", "power_W", "latency_ms"]].notna().any(axis=1)
    ].copy()

    # Collapse exact duplicate measurements copied from multiple sources.
    grouped = (
        df.groupby(
            ["device", "model", "latency_ms", "power_W", "energy_J"],
            as_index=False,
            dropna=False,
        )
        .agg(
            hw_name=("hw_name", "first"),
            notes=("notes", lambda s: "; ".join(sorted({str(x) for x in s if str(x) != "nan" and str(x)}))),
            source=("source", lambda s: ";".join(sorted(set(s)))),
            source_count=("source", "nunique"),
            source_url=("source_url", lambda s: ";".join(sorted({str(x) for x in s if str(x) != "nan" and str(x)}))),
            measurement_type=("measurement_type", lambda s: ";".join(sorted({str(x) for x in s if str(x) != "nan" and str(x)}))),
        )
    )
    return grouped


def _series_std(series: pd.Series, default_pct: float) -> float:
    mean = float(series.mean())
    if len(series) >= 2:
        std = float(series.std(ddof=1))
        if np.isfinite(std) and std > 0:
            return std
    return max(mean * default_pct, 1e-6)


def _profile_entry(device: str, model: str, real_profiles: dict) -> dict:
    model_profile = (
        real_profiles.get(device, {})
        .get("per_model", {})
        .get(model, {})
    )
    device_profile = real_profiles.get(device, {})
    defaults = DEVICE_PROFILES[device]

    return {
        "energy_scale": model_profile.get(
            "energy_scale", device_profile.get("energy_scale_mean", defaults["es"])
        ),
        "latency_scale": model_profile.get(
            "latency_scale", device_profile.get("latency_scale_mean", defaults["ls"])
        ),
        "tdp_W": model_profile.get("tdp_W", device_profile.get("tdp_W", defaults["tdp"])),
    }


def _combo_distribution(
    device: str,
    model: str,
    real_rows: pd.DataFrame,
    real_profiles: dict,
) -> dict:
    combo_rows = real_rows[
        (real_rows["device"] == device) & (real_rows["model"] == model)
    ].copy()
    baseline = MODEL_BASELINES[model]

    if not combo_rows.empty:
        means = {
            "energy_J": None,
            "power_W": None,
            "latency_ms": None,
        }
        stds = {}
        profile = _profile_entry(device, model, real_profiles)
        fallback_means = {
            "energy_J": float(baseline["e"] * profile["energy_scale"]),
            "power_W": float(profile["tdp_W"]),
            "latency_ms": float(baseline["l"] * profile["latency_scale"]),
        }

        for feature in ["energy_J", "power_W", "latency_ms"]:
            feature_rows = combo_rows[feature].dropna()
            if not feature_rows.empty:
                means[feature] = float(feature_rows.mean())
                stds[feature] = _series_std(feature_rows, DEFAULT_NOISE_STD_PCT[feature])
            else:
                means[feature] = fallback_means[feature]
                stds[feature] = max(
                    means[feature] * ESTIMATED_NOISE_STD_PCT[feature], 1e-6
                )
        measurement_count = int(len(combo_rows))
        source_count = int(combo_rows["source_count"].sum()) if "source_count" in combo_rows else measurement_count
        combo_source = "real"
    else:
        profile = _profile_entry(device, model, real_profiles)
        means = {
            "energy_J": float(baseline["e"] * profile["energy_scale"]),
            "power_W": float(profile["tdp_W"]),
            "latency_ms": float(baseline["l"] * profile["latency_scale"]),
        }
        stds = {
            feature: max(means[feature] * ESTIMATED_NOISE_STD_PCT[feature], 1e-6)
            for feature in ["energy_J", "power_W", "latency_ms"]
        }
        measurement_count = 0
        source_count = 0
        combo_source = "estimated"

    return {
        "means": means,
        "stds": stds,
        "measurement_count": measurement_count,
        "source_count": source_count,
        "combo_source": combo_source,
        "params_M": baseline["p"],
        "flops_G": baseline["f"],
    }


def _sample_positive(
    mean: float,
    std: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    floor = max(mean * 0.05, 1e-6)
    return np.maximum(floor, rng.normal(mean, std, n))


def build_training_dataframe(
    csv_path: Path | str = REAL_BENCHMARK_CSV,
    rows_per_combo: int = TRAINING_ROWS_PER_COMBO,
    random_seed: int = 42,
    include_devices: list[str] | None = None,
    include_models: list[str] | None = None,
    observed_only: bool = False,
) -> pd.DataFrame:
    """
    Build the dataset used by train/generate/evaluate.

    Preferred path:
    - Load deduplicated real benchmark rows.
    - Use those rows to define means/stds for observed combos.
    - Estimate missing combos from ingested real-device profiles, falling back
      to static defaults only when no real profile exists.

    Fallback path:
    - Legacy synthetic seed generator.
    """
    csv_path = Path(csv_path)
    csv_path = resolve_project_path(csv_path)
    selected_devices = _select_keys(include_devices, DEVICES, "devices")
    selected_models = _select_keys(include_models, MODELS, "models")
    if not csv_path.exists():
        df = generate_legacy_seed_data(random_seed=random_seed)
        df = df[
            df["device"].isin(selected_devices) & df["model"].isin(selected_models)
        ].copy()
        return df

    real_rows = load_real_benchmark_data(csv_path)
    real_rows = real_rows[
        real_rows["device"].isin(selected_devices) & real_rows["model"].isin(selected_models)
    ].copy()
    real_profiles = load_real_device_profiles()
    rng = np.random.default_rng(random_seed)
    rows = []

    if observed_only:
        combos = [
            (device, model)
            for device in selected_devices
            for model in selected_models
            if not real_rows[
                (real_rows["device"] == device) & (real_rows["model"] == model)
            ].empty
        ]
    else:
        combos = [
            (device, model)
            for device in selected_devices
            for model in selected_models
        ]

    for device, model in combos:
            combo = _combo_distribution(device, model, real_rows, real_profiles)
            e_samples = _sample_positive(
                combo["means"]["energy_J"], combo["stds"]["energy_J"], rows_per_combo, rng
            )
            p_samples = _sample_positive(
                combo["means"]["power_W"], combo["stds"]["power_W"], rows_per_combo, rng
            )
            l_samples = _sample_positive(
                combo["means"]["latency_ms"], combo["stds"]["latency_ms"], rows_per_combo, rng
            )
            energy_std = float(np.std(e_samples))

            for e, p, l in zip(e_samples, p_samples, l_samples):
                rows.append({
                    "device": device,
                    "model": model,
                    "energy_J": float(e),
                    "power_W": float(p),
                    "latency_ms": float(l),
                    "energy_std": energy_std,
                    "params_M": combo["params_M"],
                    "flops_G": combo["flops_G"],
                    "device_idx": DEVICES[device],
                    "model_idx": MODELS[model],
                    "combo_source": combo["combo_source"],
                    "measurement_count": combo["measurement_count"],
                    "source_count": combo["source_count"],
                })

    df = pd.DataFrame(rows)
    real_combo_count = int((df.groupby(["device", "model"])["combo_source"].first() == "real").sum())
    total_combo_count = int(df.groupby(["device", "model"]).ngroups)
    print(
        f"[data] Grounded training dataset: {len(df):,} rows "
        f"({df['device'].nunique()} devices × {df['model'].nunique()} models × {rows_per_combo} rows)"
    )
    print(
        f"[data] Combo coverage: {real_combo_count} real combos, "
        f"{total_combo_count - real_combo_count} estimated combos"
    )
    return df


def generate_seed_data(
    trials_per_combo: int = TRAINING_ROWS_PER_COMBO,
    random_seed: int = 42,
    csv_path: Path | str = REAL_BENCHMARK_CSV,
) -> pd.DataFrame:
    """
    Compatibility wrapper for older code paths.

    The preferred behavior now is to build the grounded training dataset from
    the real benchmark CSV. The legacy synthetic generator remains available via
    `generate_legacy_seed_data`.
    """
    return build_training_dataframe(
        csv_path=csv_path,
        rows_per_combo=trials_per_combo,
        random_seed=random_seed,
    )


# ─── Scaler ───────────────────────────────────────────────────────────────────

class DataScaler:
    """
    Combo-aware scaler for the 4 continuous features.

    The GAN trains on per-combo residuals rather than raw physical values.
    This removes the large across-combo scale differences that otherwise cause
    the generator to collapse toward a single global average.

    For each (device, model) combo:
    - compute mean and std for each feature
    - convert rows into z-scores relative to that combo
    - clip z-scores to a bounded range and map them into [0, 1]

    During generation, inverse_transform restores physical units using the
    combo-specific mean/std implied by the requested condition vector.
    """

    Z_CLIP = 3.0

    def __init__(self, features: list[str] | None = None):
        self._scaler = MinMaxScaler()
        self._combo_means: dict[tuple[int, int], np.ndarray] = {}
        self._combo_stds: dict[tuple[int, int], np.ndarray] = {}
        self.feature_names = list(features or FEATURE_MODES["raw"])
        self.fitted = False

    def _safe_std(self, mean: float, std: float) -> float:
        if np.isfinite(std) and std > 0:
            return float(std)
        return max(abs(mean) * 0.05, 1e-6)

    def _lookup_combo_stats(
        self,
        device_idx: np.ndarray,
        model_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        means = np.zeros((len(device_idx), len(self.feature_names)), dtype=np.float64)
        stds = np.zeros((len(device_idx), len(self.feature_names)), dtype=np.float64)
        for i, (d_idx, m_idx) in enumerate(zip(device_idx, model_idx)):
            key = (int(d_idx), int(m_idx))
            means[i] = self._combo_means[key]
            stds[i] = self._combo_stds[key]
        return means, stds

    def _residualise(
        self,
        values: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        z = (values - means) / stds
        z = np.clip(z, -self.Z_CLIP, self.Z_CLIP)
        return z

    def _to_unit_interval(self, z: np.ndarray) -> np.ndarray:
        return (z + self.Z_CLIP) / (2.0 * self.Z_CLIP)

    def _from_unit_interval(self, arr: np.ndarray) -> np.ndarray:
        return arr * (2.0 * self.Z_CLIP) - self.Z_CLIP

    def fit(self, df: pd.DataFrame):
        grouped = (
            df.groupby(["device_idx", "model_idx"])[self.feature_names]
            .agg(["mean", "std"])
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            key = (int(row[0]), int(row[1]))
            means = []
            stds = []
            for offset in range(len(self.feature_names)):
                mean = float(row[2 + offset * 2])
                std = row[3 + offset * 2]
                means.append(mean)
                stds.append(self._safe_std(mean, float(std) if pd.notna(std) else np.nan))
            self._combo_means[key] = np.asarray(means, dtype=np.float64)
            self._combo_stds[key] = np.asarray(stds, dtype=np.float64)

        values = df[self.feature_names].values.astype(np.float64)
        means, stds = self._lookup_combo_stats(
            df["device_idx"].values.astype(int),
            df["model_idx"].values.astype(int),
        )
        z = self._residualise(values, means, stds)
        self._scaler.fit(self._to_unit_interval(z))
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.fitted
        values = df[self.feature_names].values.astype(np.float64)
        means, stds = self._lookup_combo_stats(
            df["device_idx"].values.astype(int),
            df["model_idx"].values.astype(int),
        )
        z = self._residualise(values, means, stds)
        transformed = self._scaler.transform(self._to_unit_interval(z))
        return transformed.astype(np.float32)

    def inverse_transform(
        self,
        arr: np.ndarray,
        device_idx: np.ndarray,
        model_idx: np.ndarray,
    ) -> np.ndarray:
        assert self.fitted
        unit = self._scaler.inverse_transform(np.asarray(arr, dtype=np.float64))
        z = self._from_unit_interval(unit)
        means, stds = self._lookup_combo_stats(
            np.asarray(device_idx, dtype=int),
            np.asarray(model_idx, dtype=int),
        )
        values = means + z * stds
        return np.maximum(values, 1e-6)

    def inverse_df(
        self,
        arr: np.ndarray,
        device_idx: np.ndarray,
        model_idx: np.ndarray,
    ) -> pd.DataFrame:
        inv = self.inverse_transform(arr, device_idx, model_idx)
        return pd.DataFrame(inv, columns=self.feature_names)


# ─── One-Hot Encoding ─────────────────────────────────────────────────────────

def one_hot_condition(device_idx: np.ndarray, model_idx: np.ndarray) -> np.ndarray:
    """Return concatenated one-hot vectors [device | model]."""
    n = len(device_idx)
    d_oh = np.zeros((n, NUM_DEVICES), dtype=np.float32)
    m_oh = np.zeros((n, NUM_MODELS), dtype=np.float32)
    d_oh[np.arange(n), device_idx] = 1.0
    m_oh[np.arange(n), model_idx] = 1.0
    return np.concatenate([d_oh, m_oh], axis=1)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class EnergyDataset(Dataset):
    """
    Wrap the training DataFrame for use with DataLoader.

    Each item: (x_scaled, cond_onehot)
        x_scaled    : torch.float32, shape (n_features,)
        cond_onehot : torch.float32, shape (11,)
    """

    def __init__(self, df: pd.DataFrame, scaler: DataScaler):
        self.x = torch.from_numpy(scaler.transform(df))
        self.cond = torch.from_numpy(
            one_hot_condition(
                df["device_idx"].values.astype(int),
                df["model_idx"].values.astype(int),
            )
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.cond[idx]


# ─── Condition Sampler ────────────────────────────────────────────────────────

def sample_conditions(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample n random (device, model) condition vectors uniformly.
    Used during generation to produce balanced synthetic output.
    """
    if rng is None:
        rng = np.random.default_rng()
    d_idx = rng.integers(0, NUM_DEVICES, size=n)
    m_idx = rng.integers(0, NUM_MODELS, size=n)
    return one_hot_condition(d_idx, m_idx), d_idx, m_idx
