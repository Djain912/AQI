# 🌫️ Real-Time AQI Predictor for Indian Urban Zones

A comprehensive Python-based AI application that fetches live Air Quality Index (AQI) and pollutant data from the official Central Pollution Control Board (CPCB) API via data.gov.in. The project predicts and classifies AQI using three different Soft Computing and Probabilistic Artificial Intelligence models side-by-side.

### 🌟 Live Dashboard Highlights
- **Real-Time Data Pipeline:** Fetches and processes PM2.5, PM10, NO2, SO2, CO, OZONE, and NH3 levels directly from the Indian Government's live API.
- **Data Historian:** Automatically stores and handles historical station snapshots, building localized datasets.
- **Interactive UI:** Built with Streamlit, providing real-time maps, historical tracking charts, model probability side-by-sides, and interactive controls to trigger model re-training. 

---

## 🧠 The 3 AI Models

This project implements three distinct mathematical paradigms to solve the AQI classification problem:

### 1. Probabilistic Graphical Model (Bayesian Network)
- **Library:** `pgmpy`
- **How it works:** Encodes causal conditional relationships among features like time (traffic proxy), precise pollutant levels (PM2.5, NOx), and overall AQI. It inherently handles missing data via probability distributions and outputs the *confidence* of each category bucket (Good, Moderate, Severe, etc.).

### 2. Soft Computing: Fuzzy Logic
- **Library:** `scikit-fuzzy`
- **How it works:** Instead of hard boundaries (e.g., 50 is Good, 51 is Satisfactory), Fuzzy Logic maps pollutants into continuous overlapping membership sets. By feeding PM2.5, NOX, and SO2 through Mamdani inference rules, it outputs a customized and continuous "AQI Risk Risk Score". 

### 3. Soft Computing: Neural Network
- **Library:** `scikit-learn` / `tensorflow`
- **How it works:** A Deep Multi-Layer Perceptron (128 → 64 → 32 neurons) that classifies the 8-dimensional space (Pollutants + Time attributes). 
- **Intelligent Fallback:** Engineered to auto-switch to a lightweight Scikit-Learn Multilayer Perceptron if TensorFlow is absent, and utilizes a mathematical rule-based fallback if it detects insufficient training data (< 200 rows), preventing untrained model guessing.

---

## 📂 Project Structure

```text
AQI-Predictor/
├── app.py                      # Main Streamlit Dashboard UI
├── requirements.txt            # Python dependencies
├── .env.example                # Template for CPCB API Keys
├── data/                       
│   ├── fetcher.py              # Bulk LIVE API fetcher & AQI logic
│   ├── historical_store.py     # City-level historical time-series storage
│   ├── station_store.py        # Station-level feature storage for Deep Learning
├── modules/
│   ├── bayesian_network.py     # pgmpy Graphical Model
│   ├── fuzzy_logic.py          # skfuzzy Mamdani Engine
│   ├── neural_network.py       # scikit-learn / keras MLP Classifier
├── scripts/
│   ├── generate_history.py     # Simulator for backdated data (for instant NN training)
├── models/                     # Saved scaler weights and .pkl/.h5 trained network weights
└── outputs/                    # Output visual confusion matrices and metrics
```

---

## 🛠️ Quick Start Guide

### 1. Requirements & Setup
Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone https://github.com/Djain912/AQI.git
cd AQI

# Install core dependencies (TensorFlow is optional but recommended)
pip install -r requirements.txt
```

### 2. Add API Credentials
1. Register for an API key at [data.gov.in](https://data.gov.in/user/register).
2. Copy `.env.example` to `.env`.
3. Paste your key inside `.env` (`DATA_GOV_API_KEY=your_key_here`).

### 3. (Optional but Recommended) Pre-Train the Neural Network
The live API does not provide historical backlog downloads cleanly. To instantly train the Neural Network, use the included synthetic historical data generator:
```bash
python scripts/generate_history.py
```
*(This will generate ~12,000 realistic rows of Mumbai AQI data spanning exactly the parameters expected by the NN).*

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 🤝 Team / Credits
- **Developed By:** [Your Name / Team]
- **API Data:** Central Pollution Control Board (CPCB) of India via data.gov.in.
