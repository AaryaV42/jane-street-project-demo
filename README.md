# Jane Street LightGBM Model Deployment

This repository contains deployment code for the Jane Street Real-Time Market Data Forecasting LightGBM model.

##  Project Structure

```
.
├── app.py                  # Flask API (recommended)
├── streamlit_app.py        # Streamlit UI (alternative)
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Flask web interface
└── model/
    └── lgbm_model.pkl     # Trained model (generated)
```

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Save Model

```bash
python train_model.py
```

**Important:** Update `train_model.py` with your actual data loading code. The current version uses dummy data for demonstration.

### 3. Choose Deployment Method

#### Option A: Flask API (Recommended) 

**Why Flask?**
- Clean REST API endpoints
- Easy to integrate with other applications
- Better for production deployment
- Lower resource usage

**Run:**
```bash
python app.py
```

**Access:**
- Web Interface: http://localhost:5000
- API Endpoints: See below

#### Option B: Streamlit (Interactive UI)

**Why Streamlit?**
- Beautiful interactive interface
- Great for demos and testing
- No API knowledge needed

**Run:**
```bash
streamlit run streamlit_app.py
```

**Access:** http://localhost:8501

## 📡 Flask API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Model Information
```bash
curl http://localhost:5000/model_info
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_00": 0.5,
      "feature_01": -0.3,
      "feature_02": 1.2,
      ...
      "feature_78": 0.1
    }
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"feature_00": 0.5, "feature_01": -0.3, ...},
      {"feature_00": 0.2, "feature_01": 0.1, ...}
    ]
  }'
```

##  Python Usage Example

```python
import requests
import numpy as np

# Generate sample features
features = {f'feature_{i:02d}': np.random.randn() for i in range(79)}

# Make prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'features': features}
)

result = response.json()
print(f"Prediction: {result['prediction']}")
```

## Streamlit Features

1. **Single Prediction**
   - Manual input for all 79 features
   - Random sample generation
   - JSON input support

2. **Batch Prediction**
   - Generate multiple random samples
   - View statistics and distributions
   - Download results as CSV

3. **CSV Upload**
   - Upload CSV file with features
   - Batch process all rows
   - Download predictions

## 🔧 Customization

### Modify Model Parameters

Edit `train_model.py`:

```python
params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': -1,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### Update Data Loading

Replace the dummy data in `train_model.py` with your actual data:

```python
# Replace this section
train_ds = pl.concat([
    pl.scan_parquet(BASE_PATH / f'partition_id={i}' / 'part-0.parquet')
    for i in range(8, 9)
])
```

### Change Prediction Range

The model clips predictions to [-5, 5]. To modify:

In `app.py` and `streamlit_app.py`:
```python
predictions = np.clip(predictions, -5, 5)  # Change range here
```

##  Model Details

- **Type:** LightGBM Regressor
- **Features:** 79 (feature_00 to feature_78)
- **Output:** Single continuous value
- **Range:** Clipped to [-5, 5]
- **Metric:** R² Score with sample weights


##  Production Considerations

1. **Security:**
   - Add API key authentication
   - Use HTTPS
   - Validate input data thoroughly

2. **Scalability:**
   - Use Gunicorn/uWSGI for Flask
   - Implement caching for predictions
   - Add load balancer for multiple instances

3. **Monitoring:**
   - Log all predictions
   - Track API response times
   - Monitor model performance

##  Troubleshooting

### Model not loading
- Ensure `train_model.py` has been run
- Check that `model/lgbm_model.pkl` exists

### Missing features error
- Verify all 79 features are provided
- Feature names must match exactly: `feature_00` to `feature_78`

### Import errors
- Run: `pip install -r requirements.txt`
- Check Python version (3.8+ recommended)

##  Next Steps

1.  Train model with real data
2.  Test API endpoints
3.  Deploy to production server
4.  Set up monitoring
5.  Add authentication

##  License

This is based on the Jane Street Real-Time Market Data Forecasting competition.

##  Contributing

Feel free to open issues or submit pull requests for improvements!

---

