"""
Task 4: Evaluate Model
-----------------------
โหลดโมเดลที่เทรนแล้ว มาประเมินผลกับ test set และบันทึกผลลัพธ์เป็น metrics.json
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TEST_DATA_PATH, MODEL_PATH, METRICS_PATH, TARGET_COL, FEATURE_COLS


def main():
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    feature_importance = dict(
        sorted(
            zip(FEATURE_COLS, model.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True,
        )
    )
    metrics["feature_importance"] = feature_importance

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[evaluate] ผลการประเมินโมเดลบน test set:")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"  {k}: {metrics[k]:.4f}")
    print(f"[evaluate] confusion matrix: {metrics['confusion_matrix']}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    print(f"[evaluate] บันทึกผลไปที่ {METRICS_PATH}")


if __name__ == "__main__":
    main()
