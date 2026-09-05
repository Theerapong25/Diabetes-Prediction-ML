"""
Task 2: Preprocessing
-----------------------
- แทนค่า 0 ที่ผิดปกติ (Glucose, BloodPressure, SkinThickness, Insulin, BMI
  เป็น 0 ไม่ได้จริงในทางการแพทย์ -> ถือเป็น missing) ด้วยค่า median
- Train/test split แบบ stratify ตาม Outcome
- บันทึกไฟล์ train.csv / test.csv สำหรับ task ถัดไป
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
    TARGET_COL, TEST_SIZE, RANDOM_STATE,
)

# คอลัมน์ที่ค่า 0 หมายถึงข้อมูลขาดหาย ไม่ใช่ค่าจริง
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    for col in ZERO_AS_MISSING_COLS:
        if col in df.columns:
            median_val = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median_val)

    df = df.dropna()
    return df


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[preprocessing] โหลดข้อมูลดิบ {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

    df_clean = clean_data(df)
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"[preprocessing] ทำความสะอาดเสร็จ เหลือ {df_clean.shape[0]} แถว -> {CLEAN_DATA_PATH}")

    train_df, test_df = train_test_split(
        df_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_clean[TARGET_COL],
    )
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"[preprocessing] แบ่งข้อมูล train={len(train_df)} แถว, test={len(test_df)} แถว")


if __name__ == "__main__":
    main()
