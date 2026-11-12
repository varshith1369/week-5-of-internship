"""
Diamond Price Prediction - Machine Learning Pipeline
Week 5 Task: Implementing Machine Learning Algorithms and Model Evaluation
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("Loading diamonds dataset...")
    df = sns.load_dataset('diamonds')

    # Data Cleaning: Remove rows where physical dimensions are 0
    df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]

    X = df.drop(columns=['price'])
    y = df['price']

    cat_cols = ['cut', 'color', 'clarity']
    num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(), cat_cols)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning using RandomizedSearchCV
    param_distributions = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=5,
        cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest model with randomized hyperparameter tuning...")
    random_search.fit(X_train, y_train)

    print("Best Parameters Found:", random_search.best_params_)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

    # Generate Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Diamond Price ($)')
    plt.ylabel('Predicted Diamond Price ($)')
    plt.title('Actual vs Predicted Diamond Prices (Random Forest)')
    plt.savefig('actual_vs_predicted.png', dpi=300)
    print("Saved evaluation plot as 'actual_vs_predicted.png'.")

if __name__ == '__main__':
    main()
