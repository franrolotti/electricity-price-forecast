# Electricity-Price-Forecast ⚡️

Hourly electricity-price forecasting for the Iberian power market (OMIE) using a **Temporal Fusion Transformer (TFT)**, the multi-horizon time-series model introduced by Lim et al. (2021).  
The goal is to deliver an end-to-end, reproducible pipeline—from raw data download to a production-ready REST API.

---

## 🗺️ Roadmap

1. **Data ingestion**  
   - OMIE hourly prices (`data/raw/omie_prices.csv`)  
   - Optional exogenous factors: weather (AEMET), demand, holiday calendar  
2. **Pre-processing**  
   - Hourly resampling, missing-value imputation, robust scaling  
   - Date/time features (*hour*, *dayofweek*, *month*, *holiday*)  
3. **Model — Temporal Fusion Transformer**  
   - Implementation: [`pytorch_forecasting.TemporalFusionTransformer`](https://pytorch-forecasting.readthedocs.io)  
   - Forecast horizons: 24 h and 168 h  
   - Loss: Quantile Loss (p50, p90)  
4. **Evaluation & interpretability**  
   - Rolling 7-day back-testing  
   - Metrics: MAPE, RMSE, P50/P90 coverage  
   - Feature importance with `calculate_prediction_actual_by_variable()`  
5. **API & Deployment**  
   - FastAPI endpoint `/forecast?hours=24`  
   - Docker container deployable to Fly.io / Render / Heroku  

---

## ✨ Methodology (paper replicated)

> **Temporal Fusion Transformers for Interpretable Multi-Horizon Time-Series Forecasting**  
> *Bryan Lim, Sercan Ö. Arik, et al.* — *Nature Communications*, 2021

| Component                        | Purpose                                                  |
|----------------------------------|----------------------------------------------------------|
| **Self-attention**               | Captures long-range temporal dependencies                |
| **Gating & skip connections**    | Selectively routes relevant information                  |
| **Variable-selection networks**  | Provide interpretability for each input feature          |

*Simplifications in this repo*  
- Smaller network (`hidden_size = 32`, `attention_head_size = 4`) so training fits in < 30 min on free Colab/Kaggle GPUs.  
- Minimal categorical embeddings; weather features are optional.




## 🚀 Quickstart



git clone https://github.com/franrolotti/electricity-price-forecast.git
cd electricity-price-forecast

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/data/make_dataset.py --start 2022-01-01 --end 2024-12-31

python src/models/train_tft.py --config config/tft.yaml

python src/models/evaluate.py --model_path models/tft_latest.ckpt




## 📊 Expected results

| Horizon | MAPE  | RMSE | P90 coverage |
| ------- | ----- | ---- | ------------ |
| 24 h    | < 6 % | —    | ≥ 90 %       |
| 168 h   | < 9 % | —    | ≥ 87 %       |




## 🛠️ Requirements

* Python ≥ 3.10
* PyTorch ≥ 2.2
* PyTorch-Forecasting ≥ 1.0
* CUDA-enabled GPU (optional, speeds up training)


## ✍️ Contributing

1. Fork the repo
2. Create a branch `feature/<name>`
3. Run `pytest` and `pre-commit run --all-files`
4. Open a pull request

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

