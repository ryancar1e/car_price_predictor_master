"""
Train Script for Car Price Predictor
Based on reference architecture: rajtilakls2510/car_price_predictor
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

def train_and_evaluate():
    print("Loading Cleaned_Car_data.csv...")
    car = pd.read_csv('Cleaned_Car_data.csv')
    print(f"Dataset Shape: {car.shape}")
    print(car.head())

    X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']

    # Preprocessing with ColumnTransformer and OneHotEncoder
    ohe = OneHotEncoder()
    ohe.fit(X[['name', 'company', 'fuel_type']])

    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
        remainder='passthrough'
    )

    # Estimator
    lr = LinearRegression()

    # Create Atomic Pipeline
    pipe = make_pipeline(column_trans, lr)

    # Train / Test Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=433)

    # Fit Pipeline
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"R² Determination Score on Test Split: {score:.4f}")

    # Serialize Model
    pickle_path = 'LinearRegressionModel.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"Serialized trained pipeline successfully saved to: {pickle_path}")

    # Sample Verification Prediction
    sample = pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2017, 35000, 'Petrol']],
                          columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    pred_val = pipe.predict(sample)[0]
    print(f"Verification Test (Maruti Swift 2017): ₹{np.round(pred_val, 2):,}")

if __name__ == '__main__':
    train_and_evaluate()
