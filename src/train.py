"""
Task 3: Train Model
-----------------------
โมเดลที่เลือก: Random Forest Classifier

เหตุผลที่เลือก:
- ข้อมูลเป็น tabular ขนาดเล็ก-กลาง มี feature ผสมทั้งตัวเลขต่อเนื่องและนับจำนวน
  RandomForest จับ non-linear relationship และ interaction ระหว่าง feature ได้ดี
  โดยไม่ต้อง scale ข้อมูลก่อน
- ทนต่อ outlier/noise ได้ดีกว่า Logistic Regression หรือ Decision Tree เดี่ยว ๆ
- ให้ feature importance ทำให้ตีความได้ว่าปัจจัยไหน (เช่น Glucose, BMI)
  มีผลต่อการทำนายเบาหวานมากที่สุด ซึ่งมีประโยชน์ในบริบททางการแพทย์
- เทรนเร็ว ไม่ต้อง tune มากก็ได้ผลลัพธ์ baseline ที่ดี เหมาะกับ pipeline อัตโนมัติ
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TRAIN_DATA_PATH, MODEL_PATH, MODEL_DIR, TARGET_COL, FEATURE_COLS, RANDOM_STATE,
)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    base_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 10, None],
        "min_samples_leaf": [1, 2, 4],
    }

    print("[train] กำลังหา hyperparameter ที่ดีที่สุดด้วย GridSearchCV (cv=5, scoring=f1)...")
    grid = GridSearchCV(
        base_model, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"[train] best params: {grid.best_params_}")
    print(f"[train] best CV f1-score: {grid.best_score_:.4f}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"[train] บันทึกโมเดลไปที่ {MODEL_PATH}")

    with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
        json.dump({"best_params": grid.best_params_, "cv_f1_score": grid.best_score_}, f, indent=2)


if __name__ == "__main__":
    main()
