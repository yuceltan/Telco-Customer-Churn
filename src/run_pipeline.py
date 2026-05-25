import sys
import joblib

from config import (
    DATA_PATH,
    MODELS_DIR,
    create_output_directories
)

from data_preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    build_preprocessor,
    create_train_test_split
)

from train_model import train_models
from evaluate_model import evaluate_models

from explain_xai import (
    save_permutation_importance,
    save_shap_global_explanations,
    save_shap_local_explanation,
    save_counterfactual_analysis
)


def main():
    print("=" * 80)
    print("TELCO CUSTOMER CHURN XAI PROJECT")
    print("=" * 80)

    create_output_directories()

    if not DATA_PATH.exists():
        print("ERROR: Dataset file not found.")
        print(f"Expected file path: {DATA_PATH}")
        print("Please download the Kaggle Telco Customer Churn CSV and place it there.")
        sys.exit(1)

    print("\nLoading dataset...")
    df = load_data(DATA_PATH)
    print(f"Original dataset shape: {df.shape}")

    print("\nCleaning dataset...")
    df_clean = clean_data(df)
    print(f"Cleaned dataset shape: {df_clean.shape}")

    print("\nSplitting features and target...")
    X, y = split_features_target(df_clean)

    print("\nCreating train-test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    print("\nBuilding preprocessor...")
    preprocessor, numerical_features, categorical_features = build_preprocessor(X_train)

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    print("\nTraining models...")
    trained_models = train_models(preprocessor, X_train, y_train)

    print("\nEvaluating models...")
    metrics_df = evaluate_models(trained_models, X_test, y_test)

    print("\nModel comparison:")
    print(metrics_df)

    best_model_name = metrics_df.sort_values(by="ROC AUC", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model selected by ROC-AUC: {best_model_name}")

    model_path = MODELS_DIR / "best_churn_model.joblib"
    joblib.dump(best_model, model_path)

    print(f"Best model saved to: {model_path}")

    print("\nRunning XAI explanations using Random Forest model...")
    xai_model = trained_models["Random Forest"]

    save_permutation_importance(xai_model, X_test, y_test)
    save_shap_global_explanations(xai_model, X_test)
    save_shap_local_explanation(xai_model, X_test)
    save_counterfactual_analysis(xai_model, X_test)

    print("\nProject completed successfully.")
    print("Check the outputs folder for figures, reports, and saved model.")
    print("=" * 80)


if __name__ == "__main__":
    main()