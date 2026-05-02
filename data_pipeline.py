from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from ltsm_config import DATA_CACHE_DIR, FEATURE_COLUMNS, FORCE_DOWNLOAD, SYMBOL, HorizonConfig

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "cache": DATA_CACHE_DIR,
        "data": output_dir / "data",
        "models": output_dir / "models",
        "plots": output_dir / "plots",
        "reports": output_dir / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _cache_path(symbol: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace("-", "_").upper()
    return DATA_CACHE_DIR / f"{safe_symbol}_daily_yfinance_max.csv"


def _normalise_downloaded_data(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.lower)
    if "adj close" in data.columns:
        data = data.rename(columns={"adj close": "adj_close"})
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    return data.sort_index()


def _read_cached_daily_data(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if FORCE_DOWNLOAD or not path.exists():
        return None
    data = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if data.empty:
        return None
    return _normalise_downloaded_data(data)


def _write_cached_daily_data(symbol: str, data: pd.DataFrame) -> Path:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol)
    data.to_csv(path, index_label="date")
    return path


def download_daily_market_data(symbol: str = SYMBOL, retries: int = 3) -> pd.DataFrame:
    cached = _read_cached_daily_data(symbol)
    if cached is not None:
        print(f"[cache] Ham veri bulundu, tekrar indirilmiyor: {_cache_path(symbol)}")
        return cached

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                symbol,
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if data.empty:
                raise RuntimeError(f"{symbol} icin veri indirilemedi.")
            data = _normalise_downloaded_data(data)
            cache_path = _write_cached_daily_data(symbol, data)
            print(f"[cache] Ham veri indirildi ve kaydedildi: {cache_path}")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"{symbol} verisi {retries} denemede indirilemedi: {last_error}")


def clean_daily_ohlcv(raw_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    df = raw_daily.copy().sort_index()
    duplicate_rows = int(df.index.duplicated(keep="last").sum())
    df = df[~df.index.duplicated(keep="last")]

    missing_before = df.isna().sum().astype(int).to_dict()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    missing_after_numeric = df.isna().sum().astype(int).to_dict()

    missing_columns = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"Eksik zorunlu OHLCV kolonlari: {missing_columns}")

    price_columns = ["open", "high", "low", "close"]
    df["close"] = df["close"].ffill().bfill()
    for column in ["open", "high", "low"]:
        df[column] = df[column].fillna(df["close"]).ffill().bfill()
    df["volume"] = df["volume"].fillna(0)
    if "adj_close" in df.columns:
        df["adj_close"] = df["adj_close"].ffill().bfill()

    invalid_price_rows = int((df[price_columns] <= 0).any(axis=1).sum())
    negative_volume_rows = int((df["volume"] < 0).sum())
    df = df[(df[price_columns] > 0).all(axis=1)]
    df.loc[df["volume"] < 0, "volume"] = 0
    df = df.dropna(subset=OHLCV_COLUMNS)

    missing_after_cleaning = df.isna().sum().astype(int).to_dict()
    report = {
        "original_rows": int(len(raw_daily)),
        "cleaned_rows": int(len(df)),
        "duplicate_rows_removed": duplicate_rows,
        "invalid_price_rows_removed": invalid_price_rows,
        "negative_volume_rows_set_to_zero": negative_volume_rows,
        "missing_before_cleaning": missing_before,
        "missing_after_numeric_conversion": missing_after_numeric,
        "missing_after_cleaning": missing_after_cleaning,
        "fill_strategy": {
            "close": "forward fill, then backward fill",
            "open_high_low": "fill from close, then forward fill, then backward fill",
            "volume": "fill missing values with 0",
            "adj_close": "forward fill, then backward fill when present",
            "technical_indicator_nan_rows": "dropped after feature engineering to avoid leakage",
        },
    }
    return df, report


def write_dataset_report(
    raw_daily: pd.DataFrame,
    cleaned_daily: pd.DataFrame,
    resampled: pd.DataFrame,
    featured: pd.DataFrame,
    config: HorizonConfig,
    paths: dict[str, Path],
    cleaning_report: dict[str, object],
) -> None:
    technical_rows_dropped = int(len(resampled) - len(featured))
    report = {
        "dataset_name": f"{SYMBOL} Yahoo Finance OHLCV",
        "symbol": SYMBOL,
        "source": "Yahoo Finance via yfinance",
        "source_library": "yfinance",
        "token_required": False,
        "cache_enabled": True,
        "force_download": FORCE_DOWNLOAD,
        "cache_file": str(_cache_path(SYMBOL)),
        "raw_interval": "1d",
        "raw_period": "max",
        "target_column": "close shifted by forecast_steps",
        "horizon": config.name,
        "resample_rule": config.resample_rule,
        "forecast_steps": config.forecast_steps,
        "lookback_periods": config.lookback,
        "feature_columns": FEATURE_COLUMNS,
        "raw_rows": int(len(raw_daily)),
        "cleaned_rows": int(len(cleaned_daily)),
        "raw_columns": list(raw_daily.columns),
        "raw_start_date": str(raw_daily.index.min().date()),
        "raw_end_date": str(raw_daily.index.max().date()),
        "cleaned_start_date": str(cleaned_daily.index.min().date()),
        "cleaned_end_date": str(cleaned_daily.index.max().date()),
        "resampled_rows": int(len(resampled)),
        "resampled_start_date": str(resampled.index.min().date()),
        "resampled_end_date": str(resampled.index.max().date()),
        "processed_rows": int(len(featured)),
        "technical_indicator_rows_dropped": technical_rows_dropped,
        "processed_start_date": str(featured.index.min().date()),
        "processed_end_date": str(featured.index.max().date()),
        "missing_values_raw": raw_daily.isna().sum().astype(int).to_dict(),
        "missing_values_cleaned": cleaned_daily.isna().sum().astype(int).to_dict(),
        "missing_values_processed": featured.isna().sum().astype(int).to_dict(),
        "missing_value_treatment": cleaning_report,
        "processed_close_summary": featured["close"].describe().to_dict(),
        "processed_volume_summary": featured["volume"].describe().to_dict(),
    }
    with (paths["reports"] / f"dataset_report_{config.name}.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    text = [
        f"Dataset: {report['dataset_name']}",
        f"Source: {report['source']}",
        "Token required: false",
        f"Cache file: {report['cache_file']}",
        f"Raw interval: {report['raw_interval']}",
        f"Horizon: {config.name}",
        f"Resample rule: {config.resample_rule}",
        f"Raw rows: {report['raw_rows']:,}",
        f"Cleaned rows: {report['cleaned_rows']:,}",
        f"Raw date range: {report['raw_start_date']} - {report['raw_end_date']}",
        f"Cleaned date range: {report['cleaned_start_date']} - {report['cleaned_end_date']}",
        f"Processed rows: {report['processed_rows']:,}",
        f"Processed date range: {report['processed_start_date']} - {report['processed_end_date']}",
        f"Technical indicator rows dropped: {technical_rows_dropped:,}",
        f"Lookback periods: {config.lookback}",
        f"Forecast steps: {config.forecast_steps}",
        "",
        "Missing value treatment:",
        json.dumps(cleaning_report["fill_strategy"], indent=2),
        "",
        "Feature columns:",
        ", ".join(FEATURE_COLUMNS),
    ]
    (paths["reports"] / f"dataset_report_{config.name}.txt").write_text("\n".join(text), encoding="utf-8")

    missing_treatment = pd.DataFrame(
        {
            "column": sorted(set(raw_daily.columns) | set(cleaned_daily.columns) | set(featured.columns)),
        }
    )
    missing_treatment["raw_missing"] = missing_treatment["column"].map(
        raw_daily.isna().sum().astype(int).to_dict()
    ).fillna(0).astype(int)
    missing_treatment["cleaned_missing"] = missing_treatment["column"].map(
        cleaned_daily.isna().sum().astype(int).to_dict()
    ).fillna(0).astype(int)
    missing_treatment["processed_missing"] = missing_treatment["column"].map(
        featured.isna().sum().astype(int).to_dict()
    ).fillna(0).astype(int)
    missing_treatment.to_csv(paths["reports"] / f"missing_value_treatment_{config.name}.csv", index=False)


def make_dataset_info_plots(
    raw_daily: pd.DataFrame,
    cleaned_daily: pd.DataFrame,
    resampled: pd.DataFrame,
    featured: pd.DataFrame,
    config: HorizonConfig,
    paths: dict[str, Path],
) -> None:
    sns.set_theme(style="whitegrid")

    row_counts = pd.DataFrame(
        {
            "stage": ["Raw daily", "Cleaned daily", "Resampled", "Processed"],
            "rows": [len(raw_daily), len(cleaned_daily), len(resampled), len(featured)],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(row_counts["stage"], row_counts["rows"], color=["#264653", "#2a9d8f", "#e76f51"])
    ax.set_title(f"{SYMBOL} {config.label} Veri Isleme Satir Sayilari")
    ax.set_ylabel("Rows")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"dataset_row_counts_{config.name}.png", dpi=180)
    plt.close(fig)

    coverage = pd.DataFrame(
        {
            "stage": ["Raw daily", "Cleaned daily", "Resampled", "Processed"],
            "start": [
                raw_daily.index.min(),
                cleaned_daily.index.min(),
                resampled.index.min(),
                featured.index.min(),
            ],
            "end": [
                raw_daily.index.max(),
                cleaned_daily.index.max(),
                resampled.index.max(),
                featured.index.max(),
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    for index, row in coverage.iterrows():
        start_num = mdates.date2num(row["start"])
        end_num = mdates.date2num(row["end"])
        ax.barh(row["stage"], end_num - start_num, left=start_num, height=0.45, color="#457b9d")
        ax.text(end_num, index, str(row["end"].date()), va="center", ha="left", fontsize=8)
        ax.text(start_num, index, str(row["start"].date()), va="center", ha="right", fontsize=8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"{SYMBOL} {config.label} Veri Tarih Kapsami")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"dataset_date_coverage_{config.name}.png", dpi=180)
    plt.close(fig)

    missing = pd.DataFrame(
        {
            "column": list(raw_daily.columns) + list(cleaned_daily.columns) + list(featured.columns),
            "missing": (
                list(raw_daily.isna().sum().astype(int))
                + list(cleaned_daily.isna().sum().astype(int))
                + list(featured.isna().sum().astype(int))
            ),
            "stage": (
                ["Raw daily"] * len(raw_daily.columns)
                + ["Cleaned daily"] * len(cleaned_daily.columns)
                + ["Processed"] * len(featured.columns)
            ),
        }
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=missing,
        x="column",
        y="missing",
        hue="stage",
        ax=ax,
        palette=["#264653", "#2a9d8f", "#e76f51"],
    )
    ax.set_title(f"{SYMBOL} {config.label} Eksik Deger Sayilari")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing values")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"dataset_missing_values_{config.name}.png", dpi=180)
    plt.close(fig)

    summary_rows = [
        ["Symbol", SYMBOL],
        ["Source", "Yahoo Finance via yfinance"],
        ["Token required", "No"],
        ["Cache file", str(_cache_path(SYMBOL))],
        ["Raw rows", f"{len(raw_daily):,}"],
        ["Cleaned rows", f"{len(cleaned_daily):,}"],
        ["Processed rows", f"{len(featured):,}"],
        ["Raw date range", f"{raw_daily.index.min().date()} - {raw_daily.index.max().date()}"],
        ["Cleaned date range", f"{cleaned_daily.index.min().date()} - {cleaned_daily.index.max().date()}"],
        ["Processed date range", f"{featured.index.min().date()} - {featured.index.max().date()}"],
        ["Horizon", config.name],
        ["Resample rule", config.resample_rule],
        ["Lookback", str(config.lookback)],
        ["Features", str(len(FEATURE_COLUMNS))],
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    table = ax.table(
        cellText=summary_rows,
        colLabels=["Dataset property", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.28, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
    ax.set_title(f"{SYMBOL} {config.label} Veri Seti Ozeti", pad=18)
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"dataset_summary_table_{config.name}.png", dpi=180)
    plt.close(fig)

    zscored = featured[FEATURE_COLUMNS].copy()
    zscored = (zscored - zscored.mean()) / zscored.std(ddof=0).replace(0, np.nan)
    zscored = zscored.replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=zscored, ax=ax, color="#a8dadc", fliersize=1.8)
    ax.set_title(f"{SYMBOL} {config.label} Normalize Ozellik Dagilimlari")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Z-score")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"dataset_feature_distributions_{config.name}.png", dpi=180)
    plt.close(fig)


def resample_ohlcv(daily_data: pd.DataFrame, config: HorizonConfig) -> pd.DataFrame:
    base = daily_data[["open", "high", "low", "close", "volume"]].copy()
    if config.resample_rule == "1D":
        resampled = base
    else:
        resampled = base.resample(config.resample_rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    resampled = resampled.dropna()
    resampled = resampled[resampled["volume"] > 0]
    return resampled


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["return_1"] = df["close"].pct_change()
    df["log_return_1"] = np.log(df["close"]).diff()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = df["volume"].pct_change()
    df["ma_7"] = df["close"].rolling(7, min_periods=7).mean()
    df["ma_21"] = df["close"].rolling(21, min_periods=21).mean()
    df["volatility_21"] = df["log_return_1"].rolling(21, min_periods=21).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def prepare_market_frame(config: HorizonConfig, paths: dict[str, Path]) -> pd.DataFrame:
    raw_daily = download_daily_market_data()
    cleaned_daily, cleaning_report = clean_daily_ohlcv(raw_daily)
    cleaned_daily.to_csv(paths["data"] / f"{SYMBOL}_daily_cleaned.csv", index_label="date")
    resampled = resample_ohlcv(cleaned_daily, config)
    featured = add_technical_features(resampled)

    minimum_rows = config.lookback + 40
    if len(featured) < minimum_rows:
        raise RuntimeError(
            f"{config.name} icin veri yetersiz: {len(featured)} satir var, "
            f"en az {minimum_rows} satir gerekli."
        )

    output_path = paths["data"] / f"{SYMBOL}_{config.name}_processed.csv"
    featured.to_csv(output_path, index_label="date")
    write_dataset_report(raw_daily, cleaned_daily, resampled, featured, config, paths, cleaning_report)
    make_dataset_info_plots(raw_daily, cleaned_daily, resampled, featured, config, paths)
    return featured


def write_eda_report(data: pd.DataFrame, config: HorizonConfig, paths: dict[str, Path]) -> None:
    report = {
        "symbol": SYMBOL,
        "horizon": config.name,
        "rows": int(len(data)),
        "start_date": str(data.index.min().date()),
        "end_date": str(data.index.max().date()),
        "lookback_periods": config.lookback,
        "forecast_steps": config.forecast_steps,
        "missing_values": data.isna().sum().astype(int).to_dict(),
        "close_summary": data["close"].describe().to_dict(),
        "feature_columns": FEATURE_COLUMNS,
    }
    with (paths["reports"] / f"eda_{config.name}.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    data.describe().to_csv(paths["reports"] / f"descriptive_stats_{config.name}.csv")


def make_eda_plots(data: pd.DataFrame, config: HorizonConfig, paths: dict[str, Path]) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(data.index, data["close"], color="#1f6f8b", linewidth=1.6)
    axes[0].set_title(f"{SYMBOL} {config.label} Kapanis Fiyati")
    axes[0].set_ylabel("Close")
    axes[1].bar(data.index, data["volume"], color="#758e4f", width=1.0)
    axes[1].set_title("Hacim")
    axes[1].set_ylabel("Volume")
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"price_volume_{config.name}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data["log_return_1"], bins=80, kde=True, ax=ax, color="#8f2d56")
    ax.set_title(f"{config.label} Log Getiri Dagilimi")
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"returns_distribution_{config.name}.png", dpi=160)
    plt.close(fig)

    corr = data[FEATURE_COLUMNS].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.3, ax=ax)
    ax.set_title(f"{config.label} Ozellik Korelasyonu")
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"feature_correlation_{config.name}.png", dpi=160)
    plt.close(fig)
