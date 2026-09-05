"""
Config กลางของโปรเจกต์ - เก็บ path และค่าคงที่ที่ใช้ร่วมกันทุก task
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "diabetes.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "diabetes_clean.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET_COL = "Outcome"

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
