# train_model.py
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
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'predictor', 'model_bundle.joblib')

# FIXED schema we will use in the app:
FEATURE_COLUMNS = [
    'gender',               # categorical: 'M','F','O'
    'age',                  # numeric
    'study_hours',          # numeric (hours/week)
    'attendance',           # numeric (percentage 0-100)
    'previous_grade',       # numeric (0-100)
    'assignments_completed' # numeric (count)
]
TARGET_COLUMN = 'final_grade'  # numeric 0-100

def train_from_csv(csv_file_path, model_path=MODEL_PATH):
    print("Loading:", csv_file_path)
    df = pd.read_csv(csv_file_path)

    # Validate columns
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {missing_columns}")

    x= df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(float)

    numeric_features = ['age','study_hours','attendance','previous_grade','assignments_completed']
    categorical_features = ['gender']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', regressor)])

    x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    r2 = metrics.r2_score(y_test, predictions)
    print(f"RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # Save pipeline + metadata for prediction code
    model_bundle = {
        'pipeline': pipeline,
        'features': FEATURE_COLUMNS
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)
    print("Saved model to:", model_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_model.py path/to/student_data.csv")
        sys.exit(1)
    csv_input_path = sys.argv[1]
    train_from_csv(csv_input_path)
