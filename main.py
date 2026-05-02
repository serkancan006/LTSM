import argparse

from experiment import run_experiment
from ltsm_config import HORIZON_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPY fiyat tahmini icin LSTM ve Large Time-Series Transformer karsilastirmasi."
    )
    parser.add_argument(
        "horizon",
        choices=sorted(HORIZON_CONFIGS.keys()),
        help="Calistirilacak tahmin periyodu.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.horizon)
