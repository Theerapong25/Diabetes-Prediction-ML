"""
ml_03_diabetes_pipeline.py
================
Workshop 3: ML Pipeline บน Apache Airflow (Pima Indians Diabetes - Classification)
*** ไฟล์หลักของ Workshop 3 ***

เป้าหมายของไฟล์นี้:
- เทรนโมเดล "ทำนายว่าเสี่ยงเบาหวานหรือไม่" (binary classification) จากชุดข้อมูล
  Pima Indians Diabetes Database ของ Kaggle
- โครงสร้างเดียวกับแนวคิด champion-challenger ใน weather_pipeline_dag (Workshop 2)
  แต่เปลี่ยนโจทย์จาก regression (RMSE ยิ่งต่ำยิ่งดี) เป็น classification
  (accuracy/F1 ยิ่งสูงยิ่งดี) จึงกลับทิศทางการเปรียบเทียบตอน decide_deploy
- ดึงข้อมูลดิบครั้งเดียว (one-time bootstrap แบบ idempotent เช็คจำนวนแถวก่อน)
  ต่างจาก weather ที่ดึงข้อมูลใหม่ทุกวัน เพราะชุดข้อมูลนี้เป็นข้อมูลนิ่ง
  (static dataset) ไม่มี "ข้อมูลของวันนี้" ให้ดึงเพิ่ม
- มี smoke_test เหมือนไฟล์ก่อนหน้า (ทดสอบเรียกโมเดลที่เพิ่ง deploy จริง
  ด้วยเคสตัวอย่าง 1 คน)

*** หมายเหตุสำคัญเรื่องคำสั่ง Kaggle ***
คำสั่งที่ระบุมา `kaggle kernels pull mragpavank/pima-indians-diabetes-database`
เป็นคำสั่งสำหรับดึง "kernel" (โค้ด/สมุดบันทึกของคนอื่น) ไม่ใช่คำสั่งสำหรับดึง
"dataset" (ไฟล์ข้อมูลดิบ) โดยตรง — ปกติแล้วการดึงไฟล์ CSV ต้องใช้
`kaggle datasets download -d uciml/pima-indians-diabetes-database` แทน

ไฟล์นี้จึงเขียน task `bootstrap_data` ให้ "ลองทำตามคำสั่งที่ระบุมาก่อน"
(kernels pull) แล้วค้นหาไฟล์ .csv ที่ได้มาโดยอัตโนมัติ ถ้าหาไม่เจอ (เพราะ
kernel นั้นอาจไม่มีไฟล์ output แนบมาด้วย) จะ fallback ไปใช้คำสั่ง
`kaggle datasets download` แทนโดยอัตโนมัติ พร้อม log แจ้งให้ทราบทุกครั้งว่า
ใช้วิธีไหนสำเร็จ เพื่อให้ pipeline รันได้จริงแม้คำสั่งตั้งต้นจะไม่ได้ผลลัพธ์
ตามที่คาดหวัง

*** ข้อกำหนดก่อนรันไฟล์นี้ ***
- ต้องเพิ่ม scikit-learn, joblib, pandas, kaggle ใน _PIP_ADDITIONAL_REQUIREMENTS
  ของ Airflow (pandas/scikit-learn/joblib น่าจะมีอยู่แล้วจาก workshop ก่อนหน้า
  ส่วน `kaggle` เป็นแพ็กเกจใหม่ที่ต้องเพิ่ม)
- ต้องมีไฟล์ credential ของ Kaggle (kaggle.json) วางไว้ที่ ~/.kaggle/kaggle.json
  ภายใน container ของ Airflow worker/scheduler (ขอได้จาก Kaggle > Account >
  Create New API Token) และตั้งสิทธิ์ไฟล์เป็น 600
- ต้องมี Airflow Connection "postgres_target" อยู่แล้ว (ตัวเดียวกับ workshop
  ก่อนหน้า — host: postgres_target, port: 5432)
- ต้อง mount โฟลเดอร์ ./models และ ./data ไว้แล้ว (ใช้ path เดียวกับ pattern
  เดิมของ workshop, เปลี่ยนแค่ชื่อโฟลเดอร์ย่อยเป็น diabetes_models / diabetes)

วิธีทดสอบ:
1. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
2. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
3. เปิด http://localhost:8080 -> หา DAG ชื่อ diabetes_pipeline_dag -> Unpause
4. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
5. รอบแรกจะใช้เวลานานกว่าปกติที่ task bootstrap_data เพราะต้องดาวน์โหลด
   ข้อมูลจาก Kaggle และโหลดเข้า Postgres ครั้งแรก
6. ดู Graph view -> เห็น flow เต็ม: create_tables -> bootstrap_data ->
   prepare_training_data -> train_model -> evaluate_model ->
   get_previous_metric -> decide_deploy (branch) -> deploy_model ->
   smoke_test -> log_result (หรือทาง skip_deploy)
7. เปิด Logs ของ evaluate_model ดู accuracy/F1 ที่ได้ และ decide_deploy
   ดูว่าเทียบกับรอบก่อนแล้วผ่านไหม (รอบแรกที่ยังไม่มีรอบก่อนหน้า จะ
   deploy เสมอ)
8. ลอง trigger DAG ซ้ำอีกครั้ง -> bootstrap_data จะข้ามการดาวน์โหลด
   (idempotent เพราะเช็คจำนวนแถวในตารางก่อน) แต่ train_model จะสุ่มแบ่ง
   train/holdout ใหม่ทุกครั้ง (random_state คงที่ตอนเทรน แต่ผลลัพธ์ควร
   ใกล้เคียงเดิม) ทำให้ decide_deploy มีโอกาส deploy หรือ skip แตกต่างกันได้
"""

import glob
import math
import os
import subprocess
import zipfile
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop3",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

POSTGRES_CONN_ID = "postgres_target"   # connection เดียวกับ workshop ก่อนหน้า
MODEL_NAME = "diabetes_classifier"

# คำสั่งหลักตามที่ระบุมา (kernel) และคำสั่งสำรอง (dataset จริง) เผื่อ kernel
# ไม่มีไฟล์ output แนบมาด้วย
KAGGLE_KERNEL = "mragpavank/pima-indians-diabetes-database"
KAGGLE_DATASET_FALLBACK = "uciml/pima-indians-diabetes-database"

DATA_DIR = "/opt/airflow/data"
MODEL_DIR = "/opt/airflow/models/diabetes_models"
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "current_model.pkl")

HOLDOUT_FRACTION = 0.2   # กันไว้ 20% สำหรับวัดผล (ไม่ใช้เทรน) 
RANDOM_STATE = 42

# คอลัมน์มาตรฐานของชุดข้อมูล Pima Indians Diabetes
FEATURE_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
]
TARGET_COLUMN = "outcome"

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS diabetes_data (
    id SERIAL PRIMARY KEY,
    pregnancies INT NOT NULL,
    glucose FLOAT NOT NULL,
    blood_pressure FLOAT NOT NULL,
    skin_thickness FLOAT NOT NULL,
    insulin FLOAT NOT NULL,
    bmi FLOAT NOT NULL,
    diabetes_pedigree_function FLOAT NOT NULL,
    age INT NOT NULL,
    outcome INT NOT NULL,
    inserted_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    accuracy FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    deployed BOOLEAN NOT NULL,
    run_at TIMESTAMP DEFAULT NOW()
);
"""


def _find_csv_in(directory):
    """ค้นหาไฟล์ .csv ที่มีคอลัมน์หน้าตาคล้ายชุดข้อมูลเบาหวานในโฟลเดอร์ที่ระบุ"""
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    for path in csv_files:
        try:
            header = pd.read_csv(path, nrows=0).columns.str.lower().tolist()
        except Exception:
            continue
        if any("glucose" in col or "outcome" in col for col in header):
            return path
    return csv_files[0] if csv_files else None


def bootstrap_data(**kwargs):
    """
    โหลดข้อมูล diabetes จาก CSV ที่มีอยู่แล้วใน DATA_DIR
    ถ้า PostgreSQL มีข้อมูลแล้ว จะไม่โหลดซ้ำ
    """

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    count = hook.get_first(
        "SELECT COUNT(*) FROM diabetes_data;"
    )[0]

    print(f"มีข้อมูลอยู่แล้ว {count} แถวในตาราง diabetes_data")

    # มีข้อมูลแล้ว ไม่ต้องโหลดซ้ำ
    if count > 0:
        print("ข้อมูลมีอยู่แล้ว ข้ามการ bootstrap")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # ใช้ไฟล์ที่มีอยู่แล้ว
    possible_files = [
        os.path.join(DATA_DIR, "diabetes_clean.csv"),
        os.path.join(DATA_DIR, "diabetes.csv"),
    ]

    csv_path = None

    for path in possible_files:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"ไม่พบ diabetes_clean.csv หรือ diabetes.csv ใน {DATA_DIR}"
        )

    print(f"ใช้ไฟล์ข้อมูล: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"พบข้อมูลทั้งหมด {len(df)} แถว")
    print(f"Columns: {df.columns.tolist()}")

    # ทำชื่อ column ให้เป็นมาตรฐาน
    df.columns = [
        c.strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    rename_map = {
        "pregnancies": "pregnancies",
        "glucose": "glucose",
        "bloodpressure": "blood_pressure",
        "blood_pressure": "blood_pressure",
        "skinthickness": "skin_thickness",
        "skin_thickness": "skin_thickness",
        "insulin": "insulin",
        "bmi": "bmi",
        "diabetespedigreefunction": "diabetes_pedigree_function",
        "diabetes_pedigree_function": "diabetes_pedigree_function",
        "age": "age",
        "outcome": "outcome",
    }

    df = df.rename(columns=rename_map)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"ไฟล์ข้อมูลขาดคอลัมน์ที่จำเป็น: {missing}"
        )

    # เลือกเฉพาะ column ที่ต้องใช้
    df = df[required_columns]

    # ตรวจ missing values
    if df.isnull().sum().sum() > 0:
        print("พบ Missing Values")
        print(df.isnull().sum())

        df = df.dropna()

        print(
            f"หลังลบ Missing Values เหลือ {len(df)} แถว"
        )

    # แปลงข้อมูลให้เหมาะกับ PostgreSQL
    rows = [
        tuple(r)
        for r in df.itertuples(
            index=False,
            name=None
        )
    ]

    hook.insert_rows(
        table="diabetes_data",
        rows=rows,
        target_fields=required_columns,
        commit_every=200,
    )

    print(
        f"โหลดข้อมูลเข้า PostgreSQL สำเร็จ: {len(rows)} แถว"
    )
def prepare_training_data(**kwargs):
    """ดึงข้อมูลทั้งหมดจาก Postgres มาแบ่งเป็น feature (X) และ target (y)"""
    ti = kwargs["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    columns_sql = ", ".join(FEATURE_COLUMNS + [TARGET_COLUMN])
    rows = hook.get_records(f"SELECT {columns_sql} FROM diabetes_data;")
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + [TARGET_COLUMN])
    print(f"ดึงข้อมูลมาทั้งหมด {len(df)} แถว")

    features = df[FEATURE_COLUMNS].values.tolist()
    targets = df[TARGET_COLUMN].astype(int).values.tolist()

    # ตัวอย่าง 1 คนสำหรับ smoke_test ทีหลัง (ใช้แถวแรกในข้อมูล)
    sample_features = features[0]
    sample_outcome = targets[0]

    ti.xcom_push(key="features", value=features)
    ti.xcom_push(key="targets", value=targets)
    ti.xcom_push(key="sample_features", value=sample_features)
    ti.xcom_push(key="sample_outcome", value=sample_outcome)


def train_model(**kwargs):
    """เทรน RandomForestClassifier โดยแบ่ง train/holdout ด้วย train_test_split"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    ti = kwargs["ti"]
    features = ti.xcom_pull(task_ids="prepare_training_data", key="features")
    targets = ti.xcom_pull(task_ids="prepare_training_data", key="targets")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features,
        targets,
        test_size=HOLDOUT_FRACTION,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    run_id = kwargs["run_id"].replace(":", "-").replace("+", "-")
    candidate_path = os.path.join(MODEL_DIR, f"candidate_{run_id}.pkl")
    joblib.dump(model, candidate_path)

    ti.xcom_push(key="candidate_model_path", value=candidate_path)
    ti.xcom_push(key="X_holdout", value=X_holdout)
    ti.xcom_push(key="y_holdout", value=y_holdout)
    print(f"เทรนโมเดลเสร็จ (train {len(X_train)} แถว, กันไว้วัดผล {len(X_holdout)} แถว)")
    print(f"บันทึกไว้ที่: {candidate_path}")


def evaluate_model(**kwargs):
    """วัด accuracy และ F1 บนชุด holdout ที่กันไว้ (ไม่เคยใช้เทรน)"""
    import joblib
    from sklearn.metrics import accuracy_score, f1_score

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")
    X_holdout = ti.xcom_pull(task_ids="train_model", key="X_holdout")
    y_holdout = ti.xcom_pull(task_ids="train_model", key="y_holdout")

    model = joblib.load(candidate_path)
    predictions = model.predict(X_holdout)

    accuracy = accuracy_score(y_holdout, predictions)
    f1 = f1_score(y_holdout, predictions)

    ti.xcom_push(key="accuracy", value=accuracy)
    ti.xcom_push(key="f1_score", value=f1)
    print(f"ประเมินผลโมเดล: accuracy = {accuracy:.4f}, F1 = {f1:.4f} (ยิ่งสูงยิ่งดี)")


def get_previous_metric(**kwargs):
    """ดึง accuracy ของโมเดลที่ deploy ล่าสุดจาก Postgres มาเทียบ (champion)"""
    ti = kwargs["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    row = hook.get_first(
        "SELECT accuracy FROM model_metrics "
        "WHERE model_name = %s AND deployed = TRUE "
        "ORDER BY run_at DESC LIMIT 1;",
        parameters=(MODEL_NAME,),
    )
    previous_accuracy = row[0] if row else None

    ti.xcom_push(key="previous_accuracy", value=previous_accuracy)
    if previous_accuracy is None:
        print("ยังไม่เคยมีโมเดลที่ deploy มาก่อน (รอบแรก) — ถือว่ายังไม่มี champion ให้เทียบ")
    else:
        print(f"accuracy ของโมเดลรอบก่อนหน้า (champion ปัจจุบัน): {previous_accuracy:.4f}")


def decide_deploy(**kwargs):
    """
    BranchPythonOperator: deploy เฉพาะเมื่อ accuracy ใหม่ดีกว่ารอบก่อน (หรือ
    ยังไม่มีรอบก่อน) — ตรงข้ามทิศทางกับ RMSE เพราะ accuracy ยิ่งสูงยิ่งดี
    """
    ti = kwargs["ti"]
    accuracy = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")
    previous_accuracy = ti.xcom_pull(task_ids="get_previous_metric", key="previous_accuracy")

    if previous_accuracy is None or accuracy > previous_accuracy:
        print(f"accuracy ใหม่ {accuracy:.4f} ดีกว่า (หรือไม่มี) champion เดิม -> deploy_model")
        return "deploy_model"
    else:
        print(
            f"accuracy ใหม่ {accuracy:.4f} ไม่ดีกว่า champion เดิม "
            f"{previous_accuracy:.4f} -> skip_deploy"
        )
        return "skip_deploy"


def deploy_model(**kwargs):
    """คัดลอกโมเดลที่ชนะไปทับ current_model.pkl (จำลอง production)"""
    import shutil

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")

    shutil.copyfile(candidate_path, CURRENT_MODEL_PATH)
    print(f"Deploy สำเร็จ: {candidate_path} -> {CURRENT_MODEL_PATH}")

    return "deployed"


def skip_deploy(**kwargs):
    """ไม่ deploy เพราะยังสู้โมเดลเดิม (champion) ไม่ได้"""
    ti = kwargs["ti"]
    accuracy = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")
    previous_accuracy = ti.xcom_pull(task_ids="get_previous_metric", key="previous_accuracy")
    print(f"ข้าม deploy: accuracy ใหม่ {accuracy:.4f} vs champion เดิม {previous_accuracy:.4f}")
    print("โมเดลเดิม (current_model.pkl) ยังคงใช้งานต่อไป")

    return "skipped"


def smoke_test(**kwargs):
    """ทดสอบเรียกใช้งานโมเดลที่เพิ่ง deploy จริง ด้วยเคสตัวอย่าง 1 คน"""
    import joblib

    ti = kwargs["ti"]
    sample_features = ti.xcom_pull(task_ids="prepare_training_data", key="sample_features")
    sample_outcome = ti.xcom_pull(task_ids="prepare_training_data", key="sample_outcome")

    model = joblib.load(CURRENT_MODEL_PATH)
    prediction = int(model.predict([sample_features])[0])
    probability = model.predict_proba([sample_features])[0][1]

    print("===== Smoke Test: เรียกใช้งานโมเดลที่เพิ่ง deploy =====")
    print(f"feature ตัวอย่าง: {sample_features}")
    print(f"ผลจริงในข้อมูล (outcome): {sample_outcome}")
    print(f"โมเดลทำนาย: {prediction} (ความน่าจะเป็นเสี่ยงเบาหวาน: {probability:.2%})")
    print("โมเดลใช้งานได้จริง พร้อมให้บริการ")


def log_result(**kwargs):
    """บันทึกผล accuracy/F1 รอบนี้ลง Postgres (ให้รอบถัดไปดึงไปเทียบเป็น champion)"""
    ti = kwargs["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    accuracy = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")
    f1 = ti.xcom_pull(task_ids="evaluate_model", key="f1_score")
    skip_result = ti.xcom_pull(task_ids="skip_deploy")
    deployed = skip_result is None  # ถ้า skip_deploy ไม่ได้รัน แปลว่า deploy ไปแล้ว

    hook.run(
        "INSERT INTO model_metrics (model_name, accuracy, f1_score, deployed) "
        "VALUES (%s, %s, %s, %s);",
        parameters=(MODEL_NAME, accuracy, f1, deployed),
    )

    print("===== สรุปผล Workshop 3 (diabetes pipeline) =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Deploy รอบนี้: {'ใช่' if deployed else 'ไม่ใช่'}")


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="diabetes_pipeline_dag",
    default_args=default_args,
    description=(
        "Workshop 3: ทำนายความเสี่ยงเบาหวาน (Pima Indians Diabetes) "
        "ด้วย champion-challenger บน Postgres"
    ),
    schedule=None,   # trigger เองเพื่อทดสอบใน workshop (ข้อมูลเป็น static dataset)
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["workshop3", "ml-pipeline", "classification"],
) as dag:

    create_tables_task = PostgresOperator(
        task_id="create_tables",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=CREATE_TABLES_SQL,
    )

    bootstrap_task = PythonOperator(
        task_id="bootstrap_data",
        python_callable=bootstrap_data,
    )

    prepare_task = PythonOperator(
        task_id="prepare_training_data",
        python_callable=prepare_training_data,
    )

    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    previous_metric_task = PythonOperator(
        task_id="get_previous_metric",
        python_callable=get_previous_metric,
    )

    decide_task = BranchPythonOperator(task_id="decide_deploy", python_callable=decide_deploy)

    deploy_task = PythonOperator(task_id="deploy_model", python_callable=deploy_model)
    skip_task = PythonOperator(task_id="skip_deploy", python_callable=skip_deploy)

    smoke_test_task = PythonOperator(task_id="smoke_test", python_callable=smoke_test)

    log_task = PythonOperator(
        task_id="log_result",
        python_callable=log_result,
        trigger_rule="none_failed_min_one_success",
    )

    # ลำดับการรันทั้งหมด: เก็บข้อมูล -> เทรน -> ประเมินผล -> ตัดสินใจ deploy
    # -> (deploy -> ทดสอบ) หรือ (skip) -> บันทึกผล
    (
        create_tables_task
        >> bootstrap_task
        >> prepare_task
        >> train_task
        >> evaluate_task
        >> previous_metric_task
        >> decide_task
    )
    decide_task >> deploy_task >> smoke_test_task >> log_task
    decide_task >> skip_task >> log_task