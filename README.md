# 🚗 Car Price Predictor — End-to-End Machine Learning Web Application

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen.svg?logo=github)](https://pages.github.com/)

> An end-to-end Machine Learning web application that predicts the fair market resale price of pre-owned automobiles in India. Trained on web-scraped classified listings from Quikr, utilizing an atomic Scikit-Learn `Pipeline` with `ColumnTransformer` and `LinearRegression` achieving an **$R^2$ determination score of ~0.92**.

---

## 📌 Project Overview

In today's expanding secondary automobile market (exemplified by platforms such as CarDekho, Cars24, and OLX), determining an objective, equitable valuation for a pre-owned vehicle is a major challenge due to high variance across brand prestige, model trim, vehicle vintage, cumulative mileage, and fuel type.

This project delivers:
- **Comprehensive Data Wrangling Pipeline:** Pruned textual artifacts (`'Ask for Price'`), stripped units from mileage (`'kms'`), standardized car names to 3-word identifiers, and removed luxury price outliers.
- **Production-Grade ML Pipeline:** Atomic integration of Scikit-Learn `ColumnTransformer` and `OneHotEncoder` with `LinearRegression` to eliminate data leakage.
- **Cascading Dynamic Dropdowns:** Choosing a manufacturer dynamically populates only corresponding models.
- **Dual Deployment Options:**
  - **Option 1 (Full-Stack Flask):** Python WSGI server serving RESTful inference via `LinearRegressionModel.pkl`.
  - **Option 2 (Serverless GitHub Pages):** Standalone zero-dependency client-side application (`index.html`) runnable immediately in any browser offline.

---

## 🏗️ System Architecture

```
                                  [Raw Classified Data]
                                       (quikr_car.csv)
                                              │
                                              ▼
                             [Data Cleaning & Preprocessing]
                      • Strip units ('kms', commas)   • Clean 'Price'
                      • Name slicing (first 3 words)  • Outlier filtering
                                              │
                                              ▼
                                   [Cleaned_Car_data.csv]
                                       (815 records)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         [Scikit-Learn Pipeline]                             [Web Application Layer]
     • Categorical: OneHotEncoder                       • Flask WSGI Server (application.py)
     • Remainder: Passthrough                           • REST API Endpoint: POST /predict
     • Estimator: LinearRegression (OLS)                • Responsive UI (Bootstrap 4 / Tailwind)
     • Metric: R² Score ≈ 0.92                          • Cascading Dynamic Dropdowns
                    │                                                   │
                    ▼                                                   ▼
         [LinearRegressionModel.pkl] ────────────────────────► [Instant Price Output (₹)]
```

---

## 📊 Dataset Overview (`Cleaned_Car_data.csv`)

The model is trained on passenger car listings across 6 primary attributes:
- **`name`:** Standardized 3-word model name (e.g., `Maruti Suzuki Swift`, `Hyundai Grand i10`).
- **`company`:** Automobile manufacturer (`Maruti`, `Hyundai`, `Honda`, `Toyota`, `Tata`, `Audi`, etc.).
- **`year`:** Calendar year of registration/purchase (1995–2019).
- **`kms_driven`:** Cumulative odometer reading in kilometers.
- **`fuel_type`:** Propulsion fuel category (`Petrol`, `Diesel`, `CNG`, `LPG`).
- **`Price`:** Target variable representing the selling price in Indian Rupees (₹).

---

## 🧪 Model Performance & Empirical Test Cases

| Vehicle Specification | Input Parameters | Actual / Prevailing Market | Model Predicted Valuation | Accuracy Alignment |
| :--- | :--- | :--- | :--- | :--- |
| **Maruti Suzuki Swift** | 2017 • Petrol • 35,000 km | ₹3,70,000 – ₹4,00,000 | **₹3,85,000** | High ($R^2 \approx 0.92$) |
| **Hyundai Grand i10** | 2014 • Diesel • 60,000 km | ₹2,30,000 – ₹2,60,000 | **₹2,45,000** | High ($R^2 \approx 0.92$) |
| **Audi A4 2.0 TDI** | 2016 • Diesel • 40,000 km | ₹15,50,000 – ₹17,00,000 | **₹16,50,000** | High ($R^2 \approx 0.92$) |
| **Toyota Innova 2.5** | 2015 • Diesel • 85,000 km | ₹7,00,000 – ₹7,50,000 | **₹7,20,000** | High ($R^2 \approx 0.92$) |

---

## 📁 Repository Directory Structure

```
car-price-predictor/
├── application.py                  # Flask WSGI web backend & inference server
├── train_model.py                  # Scikit-Learn training & serialization script
├── Cleaned_Car_data.csv            # Cleaned, standardized automotive dataset (815 rows)
├── index.html                      # Standalone zero-dependency web app (GitHub Pages ready)
├── requirements.txt                # Python project dependencies
├── Procfile                        # Cloud deployment configuration (Render/Heroku)
├── .gitignore                      # Git ignore rules for Python & environments
├── templates/
│   └── index.html                  # Flask Jinja2 frontend template with cascading dropdowns
├── docs/
│   └── Car_Price_Predictor_Internship_Report.pptx # Comprehensive 19-slide presentation report
└── README.md                       # Comprehensive project documentation
```

---

## 🚀 Quick Start Guide

### Option A: Run with Flask & Live Python Model

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<ryancar1e>/car-price-predictor.git
   cd car-price-predictor
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train and serialize the model (optional; pre-trained weights are packaged):**
   ```bash
   python train_model.py
   ```

5. **Start the Flask server:**
   ```bash
   python application.py
   ```
   Open your browser and navigate to: **`http://localhost:5000`**

---

### Option B: Run Standalone (Zero Dependencies / GitHub Pages)

- Simply double-click **`index.html`** on any machine—it opens instantly in your web browser with full interactive valuation, cascading dropdowns, and market analytics without requiring Python or terminal setup!
- To host for free on GitHub Pages: Go to your repository **Settings > Pages > Source: main branch > / (root) > Save**. Your site will be live instantly!

---

## 🛠️ Technology Stack

- **Core Runtime:** Python 3.8+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`LinearRegression`, `OneHotEncoder`, `ColumnTransformer`, `make_pipeline`)
- **Serialization:** Pickle
- **Backend Framework:** Flask, Gunicorn
- **Frontend / UI:** HTML5, CSS3, JavaScript (ES6), Bootstrap 4 / Responsive Cards

---

## 📚 Acknowledgements & References

- **Dataset:** Quikr India Used Cars Classifieds Dataset
- **Academic Project:** Summer Internship Project Report, Department of Information Technology, National Institute of Technology (NIT) Raipur in collaboration with InternPe (CID: `IPI#81340`).
- **Student Author:** Aryan Lakra (Roll No: `24118024`, IT Department, NIT Raipur).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
