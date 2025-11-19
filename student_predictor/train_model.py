# train_model.py
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'predictor', 'model_bundle.joblib')

# CSV columns you have
FEATURE_COLUMNS = [
    'student_name',
    'category',
    'class',
    'gender',
    'age',
    'study_hours',
    'attendance',
    'internal_marks',
    'assignments_completed',
]

TARGET_COLUMN = 'final_output'

def train_from_csv(csv_file_path, model_path=MODEL_PATH):

    df = pd.read_csv(csv_file_path)

    # Convert school/college output
    df['final_output'] = df.apply(
        lambda row: row['percentage'] if row['category'] == 'school' else row['cgpa'],
        axis=1
    )

    # Features
    X = df[FEATURE_COLUMNS]
    y = df['final_output']

    numeric_cols = ['age', 'study_hours', 'attendance', 'internal_marks', 'assignments_completed']
    categorical_cols = ['student_name', 'category', 'class', 'gender']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # compute MSE then RMSE (works for all sklearn versions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    r2 = metrics.r2_score(y_test, predictions)
    print(f"RMSE: {rmse:.3f}, R2: {r2:.3f}")

    joblib.dump({
        'pipeline': pipeline,
        'features': FEATURE_COLUMNS
    }, model_path)

    print("Model saved at:", model_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_model.py path/to/csv")
        sys.exit(1)

    train_from_csv(sys.argv[1])
