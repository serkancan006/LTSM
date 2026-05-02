from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from ltsm_config import FEATURE_COLUMNS, SYMBOL, HorizonConfig
from models import LSTMPriceModel, LargeTimeSeriesTransformer, count_parameters


def _shape_of(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape_of(item) for item in value]
    return str(type(value).__name__)


def _module_parameter_count(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters(recurse=False))
    trainable = sum(parameter.numel() for parameter in module.parameters(recurse=False) if parameter.requires_grad)
    return total, trainable


def layer_summary(model: nn.Module, input_shape: tuple[int, int, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hooks = []

    def register_hook(module_name: str, module: nn.Module) -> None:
        if list(module.children()):
            return

        def hook(layer: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            total_params, trainable_params = _module_parameter_count(layer)
            rows.append(
                {
                    "layer_name": module_name or layer.__class__.__name__,
                    "layer_type": layer.__class__.__name__,
                    "input_shape": json.dumps(_shape_of(inputs[0] if inputs else None)),
                    "output_shape": json.dumps(_shape_of(output)),
                    "parameters": total_params,
                    "trainable_parameters": trainable_params,
                }
            )

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        register_hook(name, module)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(input_shape, dtype=torch.float32)
        model(dummy_input)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    return pd.DataFrame(rows)


def model_architecture_config(model: nn.Module) -> dict[str, object]:
    if isinstance(model, LSTMPriceModel):
        return {
            "architecture": "LSTM",
            "input_size": model.lstm.input_size,
            "hidden_size": model.lstm.hidden_size,
            "num_layers": model.lstm.num_layers,
            "dropout": model.lstm.dropout,
            "bidirectional": model.lstm.bidirectional,
            "head": "LayerNorm -> Dropout -> Linear(1)",
        }

    if isinstance(model, LargeTimeSeriesTransformer):
        first_layer = model.encoder.layers[0]
        return {
            "architecture": "Transformer Encoder",
            "input_size": model.input_projection.in_features,
            "d_model": model.input_projection.out_features,
            "num_layers": len(model.encoder.layers),
            "num_attention_heads": first_layer.self_attn.num_heads,
            "dim_feedforward": first_layer.linear1.out_features,
            "activation": "gelu",
            "norm_first": first_layer.norm_first,
            "head": "LayerNorm -> Dropout -> Linear(1)",
        }

    return {"architecture": model.__class__.__name__}


def save_model_summaries(
    models: dict[str, nn.Module],
    config: HorizonConfig,
    paths: dict[str, Path],
    input_size: int,
) -> pd.DataFrame:
    model_rows = []
    all_layer_frames = []
    input_shape = (1, config.lookback, input_size)

    for model_name, model in models.items():
        config_summary = model_architecture_config(model)
        layers = layer_summary(model, input_shape)
        layers.insert(0, "model", model_name)
        layers.to_csv(paths["reports"] / f"layer_summary_{model_name}_{config.name}.csv", index=False)
        all_layer_frames.append(layers)

        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = count_parameters(model)
        non_trainable_params = total_params - trainable_params
        model_summary = {
            "symbol": SYMBOL,
            "horizon": config.name,
            "model": model_name,
            "input_shape": list(input_shape),
            "target": "next_period_close",
            "feature_columns": FEATURE_COLUMNS,
            "training_config": asdict(config),
            "architecture_config": config_summary,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "torch_module": str(model),
            "layers": layers.drop(columns=["model"]).to_dict(orient="records"),
        }
        with (paths["reports"] / f"model_summary_{model_name}_{config.name}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(model_summary, file, indent=2)

        text = [
            f"Model: {model_name}",
            f"Symbol: {SYMBOL}",
            f"Horizon: {config.name}",
            f"Input shape: {input_shape}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {non_trainable_params:,}",
            "",
            "Training configuration:",
            json.dumps(asdict(config), indent=2),
            "",
            "Architecture configuration:",
            json.dumps(config_summary, indent=2),
            "",
            "PyTorch module:",
            str(model),
        ]
        (paths["reports"] / f"model_summary_{model_name}_{config.name}.txt").write_text(
            "\n".join(text), encoding="utf-8"
        )

        model_rows.append(
            {
                "model": model_name,
                "architecture": config_summary["architecture"],
                "input_shape": str(input_shape),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                **{f"training_{key}": value for key, value in asdict(config).items()},
            }
        )

    model_frame = pd.DataFrame(model_rows)
    model_frame.to_csv(paths["reports"] / f"model_comparison_parameters_{config.name}.csv", index=False)

    layer_frame = pd.concat(all_layer_frames, ignore_index=True)
    layer_frame.to_csv(paths["reports"] / f"all_layer_summaries_{config.name}.csv", index=False)
    return model_frame


def plot_parameter_totals(model_frame: pd.DataFrame, config: HorizonConfig, paths: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(model_frame["model"], model_frame["trainable_parameters"], color=["#1f6f8b", "#c44900"])
    ax.set_title(f"{config.label} Model Parametre Karsilastirmasi")
    ax.set_ylabel("Trainable Parameters")
    ax.tick_params(axis="x", rotation=10)
    for index, value in enumerate(model_frame["trainable_parameters"]):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"parameter_totals_{config.name}.png", dpi=180)
    plt.close(fig)


def plot_layer_parameter_breakdown(
    models: dict[str, nn.Module],
    config: HorizonConfig,
    paths: dict[str, Path],
    input_size: int,
) -> None:
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4.5 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models.items()):
        layers = layer_summary(model, (1, config.lookback, input_size))
        layer_params = layers[layers["trainable_parameters"] > 0].copy()
        layer_params = layer_params.sort_values("trainable_parameters", ascending=True)
        ax.barh(layer_params["layer_name"], layer_params["trainable_parameters"], color="#4d7c8a")
        ax.set_title(f"{model_name} Katman Bazli Parametre Dagilimi")
        ax.set_xlabel("Trainable Parameters")
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(paths["plots"] / f"layer_parameter_breakdown_{config.name}.png", dpi=180)
    plt.close(fig)


def _architecture_steps(model_name: str, model: nn.Module, config: HorizonConfig, input_size: int) -> list[str]:
    if isinstance(model, LSTMPriceModel):
        return [
            f"Input\n{config.lookback} x {input_size} features",
            f"LSTM\nlayers={model.lstm.num_layers}\nhidden={model.lstm.hidden_size}",
            "Last time step\nsequence pooling",
            "LayerNorm\nDropout",
            "Linear Head\nnext close",
        ]

    if isinstance(model, LargeTimeSeriesTransformer):
        first_layer = model.encoder.layers[0]
        return [
            f"Input\n{config.lookback} x {input_size} features",
            f"Linear Projection\nd_model={model.input_projection.out_features}",
            "Sinusoidal\nPosition Encoding",
            f"Transformer Encoder\nlayers={len(model.encoder.layers)}\nheads={first_layer.self_attn.num_heads}",
            "LayerNorm\nDropout\nLinear Head",
        ]

    return [model_name, model.__class__.__name__]


def plot_architecture_diagrams(
    models: dict[str, nn.Module],
    config: HorizonConfig,
    paths: dict[str, Path],
    input_size: int,
) -> None:
    for model_name, model in models.items():
        steps = _architecture_steps(model_name, model, config, input_size)
        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.axis("off")
        xs = [0.08 + index * (0.84 / max(1, len(steps) - 1)) for index in range(len(steps))]
        for index, (x_pos, label) in enumerate(zip(xs, steps)):
            ax.text(
                x_pos,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#f4f1de" if model_name == "lstm" else "#edf6f9",
                    "edgecolor": "#3d405b",
                    "linewidth": 1.4,
                },
                transform=ax.transAxes,
            )
            if index < len(steps) - 1:
                ax.annotate(
                    "",
                    xy=(xs[index + 1] - 0.07, 0.5),
                    xytext=(x_pos + 0.07, 0.5),
                    arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#333333"},
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )
        ax.set_title(f"{config.label} {model_name} Mimari Diyagrami", fontsize=13)
        fig.tight_layout()
        fig.savefig(paths["plots"] / f"architecture_{model_name}_{config.name}.png", dpi=180)
        plt.close(fig)
