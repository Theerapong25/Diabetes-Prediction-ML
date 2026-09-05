"""
Task 1: Data Ingestion
-----------------------
ดึง Diabetes Dataset (Pima Indians Diabetes Database) เข้ามาใน data/diabetes.csv

วิธีใช้กับข้อมูลจริงจาก Kaggle:
  1. ไปที่ https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
  2. Download แล้ววาง diabetes.csv ไว้ที่ data/diabetes.csv
     (คอลัมน์ต้องมี: Pregnancies, Glucose, BloodPressure, SkinThickness,
      Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)
  3. หรือถ้าตั้งค่า Kaggle API (kaggle.json) ไว้แล้ว สคริปต์นี้จะ
     ดาวน์โหลดให้อัตโนมัติผ่าน kaggle CLI

ถ้าไม่มีทั้งสองอย่าง สคริปต์จะ generate synthetic dataset ที่มี
โครงสร้าง/สถิติใกล้เคียงกับ dataset จริง เพื่อให้ pipeline รันได้ end-to-end
(ควรแทนที่ด้วยข้อมูลจริงก่อนใช้งานจริง)
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_PATH, DATA_DIR


def try_download_from_kaggle() -> bool:
    """พยายามดึงข้อมูลจริงผ่าน Kaggle API ถ้ามีการตั้งค่า credentials ไว้"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "uciml/pima-indians-diabetes-database",
            "-p", DATA_DIR, "--unzip",
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        downloaded = os.path.join(DATA_DIR, "diabetes.csv")
        if os.path.exists(downloaded):
            print("[data_ingestion] ดาวน์โหลดจาก Kaggle สำเร็จ")
            return True
        return False
    except Exception as e:
        print(f"[data_ingestion] ดึงจาก Kaggle ไม่สำเร็จ ({e}) -> ใช้ fallback")
        return False


def generate_synthetic_fallback(n: int = 768, seed: int = 42) -> pd.DataFrame:
    """
    สร้างข้อมูลตัวอย่างที่มีคอลัมน์/ช่วงค่าเลียนแบบ Pima Indians Diabetes Dataset
    ใช้เฉพาะตอนไม่มีไฟล์จริงหรือดึงจาก Kaggle ไม่ได้ (เช่น ไม่มีอินเทอร์เน็ต)
    """
    rng = np.random.default_rng(seed)

    outcome = rng.binomial(1, 0.35, size=n)

    def rnorm(mean0, mean1, std, low, high):
        vals = np.where(
            outcome == 1,
            rng.normal(mean1, std, n),
            rng.normal(mean0, std, n),
        )
        return np.clip(vals, low, high)

    df = pd.DataFrame({
        "Pregnancies": np.clip(rng.poisson(3, n) + outcome * rng.poisson(1, n), 0, 17),
        "Glucose": rnorm(110, 145, 25, 44, 199).round(0),
        "BloodPressure": rnorm(68, 74, 12, 24, 122).round(0),
        "SkinThickness": rnorm(20, 27, 10, 0, 99).round(0),
        "Insulin": rnorm(70, 130, 90, 0, 846).round(0),
        "BMI": rnorm(30, 35, 6, 18, 67).round(1),
        "DiabetesPedigreeFunction": np.clip(rnorm(0.35, 0.55, 0.25, 0.08, 2.42), 0.08, 2.42).round(3),
        "Age": np.clip(rnorm(29, 37, 11, 21, 81), 21, 81).round(0).astype(int),
        "Outcome": outcome,
    })
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_DATA_PATH):
        print(f"[data_ingestion] พบไฟล์ข้อมูลอยู่แล้วที่ {RAW_DATA_PATH} -> ใช้ไฟล์นี้")
        return RAW_DATA_PATH

    if try_download_from_kaggle():
        return RAW_DATA_PATH

    print("[data_ingestion] ไม่พบไฟล์จริงและดึงจาก Kaggle ไม่ได้ -> สร้าง synthetic dataset แทน")
    df = generate_synthetic_fallback()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[data_ingestion] บันทึกข้อมูล {len(df)} แถวไปที่ {RAW_DATA_PATH}")
    return RAW_DATA_PATH


if __name__ == "__main__":
    main()
