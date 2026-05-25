from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def build_logistic_regression_pipeline(preprocessor):
  

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def build_random_forest_pipeline(preprocessor):


    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def train_models(preprocessor, X_train, y_train):


    models = {
        "Logistic Regression": build_logistic_regression_pipeline(preprocessor),
        "Random Forest": build_random_forest_pipeline(preprocessor)
    }

    trained_models = {}

    for model_name, model_pipeline in models.items():
        print(f"Training model: {model_name}")
        model_pipeline.fit(X_train, y_train)
        trained_models[model_name] = model_pipeline

    return trained_models