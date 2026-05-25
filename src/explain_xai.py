import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.inspection import permutation_importance

from config import FIGURES_DIR, REPORTS_DIR, RANDOM_STATE


def get_feature_names_from_pipeline(model_pipeline):


    preprocessor = model_pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(preprocessor.transformers_.shape[0])]

    return feature_names


def get_preprocessed_data(model_pipeline, X):


    preprocessor = model_pipeline.named_steps["preprocessor"]
    X_processed = preprocessor.transform(X)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    feature_names = get_feature_names_from_pipeline(model_pipeline)

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=X.index
    )

    return X_processed_df


def save_permutation_importance(model_pipeline, X_test, y_test):


    print("Calculating permutation importance...")

    result = permutation_importance(
        model_pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }
    ).sort_values(by="importance_mean", ascending=False)

    importance_df.to_csv(
        REPORTS_DIR / "permutation_feature_importance.csv",
        index=False
    )

    top_features = importance_df.head(15).sort_values(
        by="importance_mean",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance_mean"])
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean decrease in ROC-AUC after shuffling")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "permutation_feature_importance.png", dpi=300)
    plt.close()

    return importance_df


def select_shap_values_for_churn_class(shap_values):


    if isinstance(shap_values, list):
        return shap_values[1]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[2] == 2:
            return shap_values[:, :, 1]
        if shap_values.shape[0] == 2:
            return shap_values[1]

    return shap_values


def select_expected_value_for_churn_class(expected_value):


    if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
        if len(expected_value) > 1:
            return expected_value[1]

    return expected_value


def save_shap_global_explanations(model_pipeline, X_test, sample_size=500):


    print("Creating SHAP global explanations...")

    X_sample = X_test.sample(
        n=min(sample_size, len(X_test)),
        random_state=RANDOM_STATE
    )

    X_processed_df = get_preprocessed_data(model_pipeline, X_sample)

    rf_model = model_pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_processed_df)

    shap_values_churn = select_shap_values_for_churn_class(shap_values)

    shap.summary_plot(
        shap_values_churn,
        X_processed_df,
        show=False,
        max_display=20
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values_churn,
        X_processed_df,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap_importance_df = pd.DataFrame(
        {
            "feature": X_processed_df.columns,
            "mean_absolute_shap": np.abs(shap_values_churn).mean(axis=0)
        }
    ).sort_values(by="mean_absolute_shap", ascending=False)

    shap_importance_df.to_csv(
        REPORTS_DIR / "shap_global_feature_importance.csv",
        index=False
    )

    return shap_importance_df, explainer, X_processed_df, shap_values_churn


def save_shap_local_explanation(model_pipeline, X_test):


    print("Creating SHAP local explanation...")

    churn_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
    highest_risk_position = np.argmax(churn_probabilities)

    selected_customer = X_test.iloc[[highest_risk_position]]
    selected_probability = churn_probabilities[highest_risk_position]

    X_processed_df = get_preprocessed_data(model_pipeline, selected_customer)

    rf_model = model_pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_processed_df)
    shap_values_churn = select_shap_values_for_churn_class(shap_values)
    expected_value_churn = select_expected_value_for_churn_class(explainer.expected_value)

    explanation = shap.Explanation(
        values=shap_values_churn[0],
        base_values=expected_value_churn,
        data=X_processed_df.iloc[0],
        feature_names=X_processed_df.columns.tolist()
    )

    try:
        shap.plots.waterfall(
            explanation,
            max_display=15,
            show=False
        )

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_local_waterfall_high_risk_customer.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception:
        local_df = pd.DataFrame(
            {
                "feature": X_processed_df.columns,
                "shap_value": shap_values_churn[0]
            }
        )

        local_df["absolute_value"] = local_df["shap_value"].abs()
        local_df = local_df.sort_values(by="absolute_value", ascending=False).head(15)
        local_df = local_df.sort_values(by="shap_value")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(local_df["feature"], local_df["shap_value"])
        ax.set_title("Local SHAP Explanation - Highest Risk Customer")
        ax.set_xlabel("SHAP value")
        ax.set_ylabel("Feature")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_local_bar_high_risk_customer.png", dpi=300)
        plt.close()

    customer_original_data = selected_customer.copy()
    customer_original_data["predicted_churn_probability"] = selected_probability

    customer_original_data.to_csv(
        REPORTS_DIR / "highest_risk_customer_explanation.csv",
        index=False
    )

    return customer_original_data


def apply_what_if_change(customer, column_name, new_value):


    changed_customer = customer.copy()

    if column_name in changed_customer.columns:
        changed_customer[column_name] = new_value

    return changed_customer


def save_counterfactual_analysis(model_pipeline, X_test):
   

    print("Creating counterfactual what-if analysis...")

    churn_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
    highest_risk_position = np.argmax(churn_probabilities)

    original_customer = X_test.iloc[[highest_risk_position]].copy()
    original_probability = model_pipeline.predict_proba(original_customer)[:, 1][0]

    scenarios = []

    scenarios.append(
        {
            "scenario": "Original customer",
            "predicted_churn_probability": original_probability
        }
    )

    possible_changes = [
        ("Contract", "One year"),
        ("Contract", "Two year"),
        ("PaymentMethod", "Credit card (automatic)"),
        ("PaymentMethod", "Bank transfer (automatic)"),
        ("OnlineSecurity", "Yes"),
        ("TechSupport", "Yes"),
        ("PaperlessBilling", "No")
    ]

    for column_name, new_value in possible_changes:
        if column_name in original_customer.columns:
            changed_customer = apply_what_if_change(
                original_customer,
                column_name,
                new_value
            )

            changed_probability = model_pipeline.predict_proba(changed_customer)[:, 1][0]

            scenarios.append(
                {
                    "scenario": f"Change {column_name} to {new_value}",
                    "predicted_churn_probability": changed_probability
                }
            )

    counterfactual_df = pd.DataFrame(scenarios)
    counterfactual_df["change_vs_original"] = (
        counterfactual_df["predicted_churn_probability"] - original_probability
    )

    counterfactual_df.to_csv(
        REPORTS_DIR / "counterfactual_what_if_analysis.csv",
        index=False
    )

    plot_df = counterfactual_df.sort_values(
        by="predicted_churn_probability",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["scenario"], plot_df["predicted_churn_probability"])
    ax.set_title("What-if Analysis for Highest-Risk Customer")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Scenario")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "counterfactual_what_if_analysis.png", dpi=300)
    plt.close()

    return counterfactual_df