from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ltsm_config import FEATURE_COLUMNS, RANDOM_SEED, SYMBOL, HorizonConfig
from models import count_parameters


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dates: np.ndarray,
        target_indices: list[int],
        lookback: int,
    ) -> None:
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.dates = dates
        self.target_indices = target_indices
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        target_index = self.target_indices[index]
        start = target_index - self.lookback
        x = self.features[start:target_index]
        y = self.targets[target_index]
        date = str(pd.Timestamp(self.dates[target_index]).date())
        return torch.from_numpy(x), torch.tensor(y), date


def _make_split_indices(n_rows: int, lookback: int) -> tuple[list[int], list[int], list[int], int, int]:
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    train_indices = list(range(lookback, train_end))
    val_indices = list(range(max(train_end, lookback), val_end))
    test_indices = list(range(max(val_end, lookback), n_rows))
    if min(len(train_indices), len(val_indices), len(test_indices)) <= 0:
        raise RuntimeError(
            "Train/validation/test bolunmesi icin veri yetersiz. "
            f"Satir={n_rows}, lookback={lookback}, train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )
    return train_indices, val_indices, test_indices, train_end, val_end


def build_dataloaders(
    data: pd.DataFrame,
    config: HorizonConfig,
    paths: dict[str, Path],
) -> tuple[dict[str, DataLoader], dict[str, StandardScaler], dict[str, object]]:
    model_frame = data[FEATURE_COLUMNS].copy()
    model_frame["target_close"] = model_frame["close"].shift(-config.forecast_steps)
    model_frame = model_frame.dropna()

    train_indices, val_indices, test_indices, train_end, val_end = _make_split_indices(
        len(model_frame), config.lookback
    )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(model_frame.iloc[:train_end][FEATURE_COLUMNS])
    target_scaler.fit(model_frame.iloc[:train_end][["target_close"]])

    scaled_features = feature_scaler.transform(model_frame[FEATURE_COLUMNS])
    scaled_targets = target_scaler.transform(model_frame[["target_close"]]).reshape(-1)
    dates = model_frame.index.to_numpy()

    datasets = {
        "train": SequenceDataset(scaled_features, scaled_targets, dates, train_indices, config.lookback),
        "val": SequenceDataset(scaled_features, scaled_targets, dates, val_indices, config.lookback),
        "test": SequenceDataset(scaled_features, scaled_targets, dates, test_indices, config.lookback),
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(datasets["val"], batch_size=config.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False, num_workers=0),
    }

    joblib.dump(feature_scaler, paths["models"] / f"feature_scaler_{config.name}.joblib")
    joblib.dump(target_scaler, paths["models"] / f"target_scaler_{config.name}.joblib")

    metadata = {
        "model_frame": model_frame,
        "split": {
            "train_rows": len(train_indices),
            "val_rows": len(val_indices),
            "test_rows": len(test_indices),
            "train_end_date": str(model_frame.index[train_end - 1].date()),
            "val_end_date": str(model_frame.index[val_end - 1].date()),
        },
    }
    return loaders, {"feature": feature_scaler, "target": target_scaler}, metadata


def train_model(
    model: nn.Module,
    model_name: str,
    loaders: dict[str, DataLoader],
    config: HorizonConfig,
    paths: dict[str, Path],
    device: torch.device,
) -> dict[str, object]:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_learning_rate,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for x, y, _ in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _ in loaders["val"]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                predictions = model(x)
                loss = criterion(predictions, y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-6
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "config": asdict(config),
                    "model_name": model_name,
                    "symbol": SYMBOL,
                    "feature_columns": FEATURE_COLUMNS,
                    "parameter_count": count_parameters(model),
                    "best_val_loss": best_val_loss,
                },
                paths["models"] / f"{model_name}_{config.name}.pt",
            )
        else:
            wait += 1

        if config.verbose >= 2 or (
            config.verbose == 1 and (epoch == 1 or epoch % 10 == 0 or improved or wait >= config.patience)
        ):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{config.name}] {model_name} epoch {epoch:03d}/{config.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"lr={current_lr:.2e} early_stop_wait={wait}/{config.patience}"
            )

        if wait >= config.patience:
            if config.verbose >= 1:
                print(
                    f"[{config.name}] {model_name} early stopping: "
                    f"best_val_loss={best_val_loss:.6f}, epochs_ran={epoch}"
                )
            break

    model.load_state_dict(best_state)
    history_path = paths["reports"] / f"history_{model_name}_{config.name}.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history["train_loss"]),
        "parameters": count_parameters(model),
    }


def predict(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: StandardScaler,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    all_predictions = []
    all_targets = []
    all_dates = []
    with torch.no_grad():
        for x, y, dates in loader:
            x = x.to(device, non_blocking=True)
            predictions = model(x).detach().cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(y.numpy())
            all_dates.extend(list(dates))

    pred_scaled = np.concatenate(all_predictions).reshape(-1, 1)
    target_scaled = np.concatenate(all_targets).reshape(-1, 1)
    pred = target_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual = target_scaler.inverse_transform(target_scaled).reshape(-1)
    return pd.DataFrame({"date": all_dates, "actual": actual, "prediction": pred})


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["actual"].to_numpy()
    pred = predictions["prediction"].to_numpy()
    mse = mean_squared_error(actual, pred)
    return {
        "r2": float(r2_score(actual, pred)),
        "mae": float(mean_absolute_error(actual, pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def plot_training_history(results: dict[str, dict[str, object]], config: HorizonConfig, paths: dict[str, Path]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for model_name, result in results.items():
        history = result["history"]
        axes[0].plot(history["train_loss"], label=f"{model_name} train")
        axes[0].plot(history["val_loss"], label=f"{model_name} val")
        axes[1].plot(history["learning_rate"], label=f"{model_name} lr")
    axes[0].set_title(f"{config.label} Egitim ve Validation Kaybi")
    axes[0].set_ylabel("Scaled MSE")
    axes[0].legend()
    axes[1].set_title("Learning Rate Scheduler")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_yscale("log")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"training_history_{config.name}.png", dpi=160)
    plt.close(fig)


def plot_predictions(prediction_frames: dict[str, pd.DataFrame], config: HorizonConfig, paths: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    first_frame = next(iter(prediction_frames.values()))
    dates = pd.to_datetime(first_frame["date"])
    ax.plot(dates, first_frame["actual"], label="Gercek", color="#202020", linewidth=2)
    colors = {"lstm": "#1f6f8b", "large_ts_transformer": "#c44900"}
    for model_name, frame in prediction_frames.items():
        ax.plot(pd.to_datetime(frame["date"]), frame["prediction"], label=model_name, linewidth=1.6, color=colors[model_name])
    ax.set_title(f"{SYMBOL} {config.label} Test Tahminleri")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Kapanis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"test_predictions_{config.name}.png", dpi=160)
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, config: HorizonConfig, paths: dict[str, Path]) -> None:
    metric_columns = ["r2", "mae", "mse", "rmse"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.ravel(), metric_columns):
        ax.bar(metrics["model"], metrics[metric], color=["#1f6f8b", "#c44900"])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(f"{config.label} LSTM vs Large Time-Series Transformer")
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"metric_comparison_{config.name}.png", dpi=160)
    plt.close(fig)
