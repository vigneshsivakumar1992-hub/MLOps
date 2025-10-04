# ----------------------------
# Training Script for Production
# ----------------------------

# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for hugging face hub
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
# for experiment tracking
import mlflow
import os

# ----------------------------
# MLflow Setup
# ----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

api = HfApi()

# ----------------------------
# Load Pre-split Data from Hugging Face Dataset Repo
# ----------------------------
Xtrain_path = "hf://datasets/Vignesh-vigu/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path  = "hf://datasets/Vignesh-vigu/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/Vignesh-vigu/Tourism-Package-Prediction/ytrain.csv"
ytest_path  = "hf://datasets/Vignesh-vigu/Tourism-Package-Prediction/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test  = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path).squeeze()  # squeeze converts DataFrame → Series
y_test  = pd.read_csv(ytest_path).squeeze()

print("✅ Data loaded successfully from Hugging Face repo")

# ----------------------------
# Feature Engineering
# ----------------------------
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Handle imbalance
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# ----------------------------
# Model Definition
# ----------------------------
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, device="cuda")

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ----------------------------
# Training & Logging
# ----------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log best params
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluation
    classification_threshold = 0.45
    y_pred_train = (best_model.predict_proba(X_train)[:, 1] >= classification_threshold).astype(int)
    y_pred_test  = (best_model.predict_proba(X_test)[:, 1] >= classification_threshold).astype(int)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report  = classification_report(y_test, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save model
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Model saved at {model_path}")

    # ----------------------------
    # Upload to Hugging Face Model Hub
    # ----------------------------
    repo_id = "Vignesh-vigu/Tourism-Package-Prediction"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

print(" Training complete. Model uploaded to Hugging Face.")
