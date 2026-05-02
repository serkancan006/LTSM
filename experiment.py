from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import torch

from data_pipeline import ensure_output_dirs, make_eda_plots, prepare_market_frame, write_eda_report
from ltsm_config import FEATURE_COLUMNS, HORIZON_CONFIGS, SYMBOL
from models import LSTMPriceModel, LargeTimeSeriesTransformer, count_parameters
from reporting import (
    plot_architecture_diagrams,
    plot_layer_parameter_breakdown,
    plot_parameter_totals,
    save_model_summaries,
)
from training import (
    build_dataloaders,
    evaluate_predictions,
    get_device,
    plot_metric_comparison,
    plot_predictions,
    plot_training_history,
    predict,
    seed_everything,
    train_model,
)


def build_models(input_size: int) -> dict[str, torch.nn.Module]:
    return {
        "lstm": LSTMPriceModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.20,
        ),
        "large_ts_transformer": LargeTimeSeriesTransformer(
            input_size=input_size,
            d_model=64,
            num_layers=2,
            nhead=4,
            dim_feedforward=128,
            dropout=0.20,
        ),
    }


def run_experiment(horizon: str) -> None:
    if horizon not in HORIZON_CONFIGS:
        valid = ", ".join(HORIZON_CONFIGS)
        raise ValueError(f"Bilinmeyen horizon: {horizon}. Gecerli degerler: {valid}")

    seed_everything()
    config = HORIZON_CONFIGS[horizon]
    paths = ensure_output_dirs(config.output_dir)
    device = get_device()

    print(f"[{config.name}] Sembol: {SYMBOL}")
    print(f"[{config.name}] Cihaz: {device}")
    print(f"[{config.name}] Veri indiriliyor ve on isleme basliyor...")
    data = prepare_market_frame(config, paths)
    write_eda_report(data, config, paths)
    make_eda_plots(data, config, paths)

    print(f"[{config.name}] Sequence dataset hazirlaniyor...")
    loaders, scalers, metadata = build_dataloaders(data, config, paths)
    models = build_models(input_size=len(FEATURE_COLUMNS))

    model_frame = save_model_summaries(models, config, paths, input_size=len(FEATURE_COLUMNS))
    plot_parameter_totals(model_frame, config, paths)
    plot_layer_parameter_breakdown(models, config, paths, input_size=len(FEATURE_COLUMNS))
    plot_architecture_diagrams(models, config, paths, input_size=len(FEATURE_COLUMNS))

    parameter_report = {
        model_name: count_parameters(model)
        for model_name, model in models.items()
    }
    with (paths["reports"] / f"model_parameters_{config.name}.json").open("w", encoding="utf-8") as file:
        json.dump(parameter_report, file, indent=2)
    print(f"[{config.name}] Parametre sayilari: {parameter_report}")

    train_results = {}
    prediction_frames = {}
    validation_prediction_frames = {}
    metric_rows = []
    validation_metric_rows = []

    for model_name, model in models.items():
        print(f"[{config.name}] {model_name} egitiliyor...")
        try:
            result = train_model(model, model_name, loaders, config, paths, device)
            train_results[model_name] = result

            validation_predictions = predict(result["model"], loaders["val"], scalers["target"], device)
            validation_predictions["model"] = model_name
            validation_prediction_frames[model_name] = validation_predictions

            predictions = predict(result["model"], loaders["test"], scalers["target"], device)
            predictions["model"] = model_name
            prediction_frames[model_name] = predictions

            validation_metrics = evaluate_predictions(validation_predictions)
            validation_metrics.update(
                {
                    "model": model_name,
                    "split": "validation",
                    "parameters": result["parameters"],
                    "epochs_ran": result["epochs_ran"],
                    "best_val_loss": result["best_val_loss"],
                }
            )
            validation_metric_rows.append(validation_metrics)

            metrics = evaluate_predictions(predictions)
            metrics.update(
                {
                    "model": model_name,
                    "split": "test",
                    "parameters": result["parameters"],
                    "epochs_ran": result["epochs_ran"],
                    "best_val_loss": result["best_val_loss"],
                }
            )
            metric_rows.append(metrics)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            error_path = paths["reports"] / f"error_{model_name}_{config.name}.txt"
            error_path.write_text(str(exc), encoding="utf-8")
            raise

    validation_metrics_frame = pd.DataFrame(validation_metric_rows)
    validation_metrics_frame = validation_metrics_frame[
        ["model", "split", "parameters", "epochs_ran", "best_val_loss", "r2", "mae", "mse", "rmse"]
    ]
    validation_metrics_frame.to_csv(paths["reports"] / f"validation_metrics_{config.name}.csv", index=False)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame = metrics_frame[
        ["model", "split", "parameters", "epochs_ran", "best_val_loss", "r2", "mae", "mse", "rmse"]
    ]
    metrics_frame.to_csv(paths["reports"] / f"metrics_{config.name}.csv", index=False)

    all_metrics_frame = pd.concat([validation_metrics_frame, metrics_frame], ignore_index=True)
    all_metrics_frame.to_csv(paths["reports"] / f"all_metrics_{config.name}.csv", index=False)

    combined_validation_predictions = pd.concat(validation_prediction_frames.values(), ignore_index=True)
    combined_validation_predictions.to_csv(
        paths["reports"] / f"validation_predictions_{config.name}.csv", index=False
    )

    combined_predictions = pd.concat(prediction_frames.values(), ignore_index=True)
    combined_predictions.to_csv(paths["reports"] / f"test_predictions_{config.name}.csv", index=False)

    plot_training_history(train_results, config, paths)
    plot_predictions(prediction_frames, config, paths)
    plot_metric_comparison(metrics_frame, config, paths)

    summary = {
        "symbol": SYMBOL,
        "horizon": config.name,
        "device": str(device),
        "config": asdict(config),
        "split": metadata["split"],
        "validation_metrics": validation_metrics_frame.to_dict(orient="records"),
        "test_metrics": metrics_frame.to_dict(orient="records"),
        "all_metrics": all_metrics_frame.to_dict(orient="records"),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    with (paths["reports"] / f"summary_{config.name}.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"[{config.name}] Tamamlandi. Ciktilar: {paths['root']}")
