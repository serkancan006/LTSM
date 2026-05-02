import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

SYMBOL = os.getenv("LTSM_SYMBOL", "SPY")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATA_CACHE_DIR = OUTPUT_ROOT / "_cache"
FORCE_DOWNLOAD = os.getenv("LTSM_FORCE_DOWNLOAD", "false").strip().lower() in {"1", "true", "yes", "y"}
RANDOM_SEED = 42


@dataclass(frozen=True)
class HorizonConfig:
    name: str
    label: str
    resample_rule: str
    lookback: int
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_learning_rate: float = 1e-6
    forecast_steps: int = 1
    verbose: int = 1

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.name


HORIZON_CONFIGS = {
    "daily": HorizonConfig(
        name="daily",
        label="Gunluk",
        resample_rule="1D",
        lookback=120,
        epochs=150,
        patience=35,
        lr_patience=10,
        batch_size=64,
    ),
    "weekly": HorizonConfig(
        name="weekly",
        label="Haftalik",
        resample_rule="W-FRI",
        lookback=104,
        epochs=190,
        patience=45,
        lr_patience=12,
        batch_size=32,
    ),
    "monthly": HorizonConfig(
        name="monthly",
        label="Aylik",
        resample_rule="ME",
        lookback=48,
        epochs=260,
        patience=60,
        lr_patience=16,
        batch_size=24,
    ),
    "quarterly": HorizonConfig(
        name="quarterly",
        label="Uc_Aylik",
        resample_rule="QE",
        lookback=16,
        epochs=340,
        patience=80,
        lr_patience=20,
        batch_size=12,
    ),
}


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1",
    "log_return_1",
    "range_pct",
    "volume_change",
    "ma_7",
    "ma_21",
    "volatility_21",
    "rsi_14",
    "macd",
    "macd_signal",
]
