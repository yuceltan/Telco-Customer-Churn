import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from config import FIGURES_DIR, REPORTS_DIR


def evaluate_single_model(model_name, model, X_test, y_test):


    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

    report = classification_report(y_test, y_pred)

    report_path = REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Classification Report: {model_name}\n")
        file.write("=" * 60)
        file.write("\n\n")
        file.write(report)

    return metrics


def save_confusion_matrix(model_name, model, X_test, y_test):


    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")

    file_name = model_name.lower().replace(" ", "_")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{file_name}_confusion_matrix.png", dpi=300)
    plt.close()


def save_roc_curve(model_name, model, X_test, y_test):


    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve - {model_name}")

    file_name = model_name.lower().replace(" ", "_")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{file_name}_roc_curve.png", dpi=300)
    plt.close()


def save_precision_recall_curve(model_name, model, X_test, y_test):


    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"Precision-Recall Curve - {model_name}")

    file_name = model_name.lower().replace(" ", "_")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{file_name}_precision_recall_curve.png", dpi=300)
    plt.close()


def evaluate_models(trained_models, X_test, y_test):


    all_metrics = []

    for model_name, model in trained_models.items():
        print(f"Evaluating model: {model_name}")

        metrics = evaluate_single_model(model_name, model, X_test, y_test)
        all_metrics.append(metrics)

        save_confusion_matrix(model_name, model, X_test, y_test)
        save_roc_curve(model_name, model, X_test, y_test)
        save_precision_recall_curve(model_name, model, X_test, y_test)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = REPORTS_DIR / "model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_df