# LSTM vs Large Time-Series Transformer Price Forecasting

Bu proje sabit veri varligi olarak `SPY` kullanir. Veri Yahoo Finance uzerinden indirilir, gunluk OHLCV verisinden haftalik, aylik ve 3 aylik seriler uretilir. Her periyot kendi cikti klasorune yazar.

Veri ayarlari `.env` dosyasindan okunur:

```text
LTSM_SYMBOL=SPY
LTSM_FORCE_DOWNLOAD=false
LTSM_SEEDS=42,1337,2026
```

Yahoo Finance/yfinance bu veri icin token gerektirmez. Ham veri `outputs/_cache` klasorune kaydedilir; cache dosyasi varsa tekrar indirilmez. Veriyi zorla yenilemek icin `.env` icinde `LTSM_FORCE_DOWNLOAD=true` yapin.

## Kurulum

```powershell
pip install -r requirements.txt
```

PyTorch GPU kullanimi icin CUDA destekli PyTorch kurulumu gerekiyorsa resmi PyTorch kurulum komutunu kullanin. Kod `torch.cuda.is_available()` true ise otomatik GPU kullanir, degilse CPU ile calisir.

## Calistirma

```powershell
python daily_forecast.py
python weekly_forecast.py
python monthly_forecast.py
python quarterly_forecast.py
```

Alternatif:

```powershell
python main.py daily
python main.py weekly
python main.py monthly
python main.py quarterly
```

## Ciktilar

Her periyot icin ayri klasor olusur:

- `outputs/<periyot>/data`: islenmis veri
- `outputs/<periyot>/models`: kaydedilen LSTM, Transformer ve scaler dosyalari
- `outputs/<periyot>/plots`: EDA, egitim, tahmin ve metrik grafikleri
- `outputs/<periyot>/reports`: metrikler, tahmin CSV'leri ve ozet JSON dosyalari
- `outputs/_cache`: tekrar indirmeyi engelleyen ham Yahoo Finance verisi

Modeller ayni train/validation/test ayrimi, ayni ozellik seti ve ayni test datasiyla karsilastirilir. Early stopping, gradient clipping ve validation loss tabanli en iyi model kaydi aktiftir.
Learning rate icin `ReduceLROnPlateau` callback'i kullanilir; `lr_patience`, `lr_factor` ve `min_learning_rate` ayarlari `ltsm_config.py` icindedir. Egitim grafiginde loss degerleriyle birlikte learning rate degisimi de kaydedilir.
Egitim ilerleme ciktilari `verbose` ile kontrol edilir: `0` sessiz, `1` ozet, `2` her epoch. Validation tahminleri ve metrikleri test metriklerinden ayri olarak kaydedilir.
Her model `LTSM_SEEDS` listesindeki seed'ler ile tekrar egitilir. Seed bazli metrikler ve mean/std aggregate metrikleri ayri kaydedilir. Metrikler R2, MAE, MSE, RMSE, MAPE ve SMAPE degerlerini icerir; egitim suresi ve model dosya boyutu da raporlanir.

Makale/raporlama icin ek model dokumantasyonu da uretilir:

- `model_summary_<model>_<periyot>.json`: mimari, egitim ayarlari, feature listesi ve katman ozeti
- `model_summary_<model>_<periyot>.txt`: okunabilir PyTorch mimari ozeti
- `layer_summary_<model>_<periyot>.csv`: katman bazli input/output shape ve parametre sayilari
- `model_comparison_parameters_<periyot>.csv`: iki modelin karsilastirmali parametre tablosu
- `architecture_<model>_<periyot>.png`: model mimari diyagrami
- `parameter_totals_<periyot>.png`: toplam parametre karsilastirma grafigi
- `layer_parameter_breakdown_<periyot>.png`: katman bazli parametre dagilimi

Validation/test raporlari:

- `validation_predictions_<periyot>.csv`: validation datasindaki tahminler
- `validation_predictions_aggregate_<periyot>.csv`: seed ortalamali validation tahminleri
- `validation_metrics_<periyot>.csv`: seed bazli validation R2, MAE, MSE, RMSE, MAPE, SMAPE metrikleri
- `test_predictions_<periyot>.csv`: test datasindaki tahminler
- `test_predictions_aggregate_<periyot>.csv`: seed ortalamali test tahminleri
- `metrics_<periyot>.csv`: seed bazli test R2, MAE, MSE, RMSE, MAPE, SMAPE metrikleri
- `all_metrics_<periyot>.csv`: validation ve test metrikleri tek tabloda
- `all_metrics_aggregate_<periyot>.csv`: seed tekrarlarinin mean/std ozet tablosu

Veri seti dokumantasyonu:

- `dataset_report_<periyot>.json`: kaynak, cache, tarih araligi, satir sayilari, kolonlar, missing value ve ozet istatistikler
- `dataset_report_<periyot>.txt`: makaleye hizli aktarim icin okunabilir veri seti ozeti
- `missing_value_treatment_<periyot>.csv`: ham, temizlenmis ve islenmis veri icin kolon bazli eksik deger tablosu
- `dataset_summary_table_<periyot>.png`: veri seti ozet tablosu
- `dataset_row_counts_<periyot>.png`: ham, temizlenmis, yeniden orneklenmis ve islenmis satir sayilari
- `dataset_date_coverage_<periyot>.png`: veri setinin tarih kapsami
- `dataset_missing_values_<periyot>.png`: ham, temizlenmis ve islenmis veri eksik deger grafigi
- `dataset_feature_distributions_<periyot>.png`: normalize edilmis feature dagilimlari

Eksik deger stratejisi:

- `close`: once forward fill, sonra backward fill
- `open`, `high`, `low`: once `close` ile doldurma, sonra forward/backward fill
- `volume`: eksik degerleri `0` ile doldurma
- Teknik indikatorlerin rolling window kaynakli ilk `NaN` satirlari veri sizintisi olmamasi icin doldurulmaz, dusurulur ve raporlanir
