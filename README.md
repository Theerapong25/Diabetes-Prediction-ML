# Diabetes-Prediction-ML
งานในวิชา Big DATA ของอาจารย์สุริยะ
# Diabetes ML Pipeline (Airflow)

Pipeline สำหรับเทรนโมเดลทำนายโรคเบาหวานจาก **Pima Indians Diabetes Dataset**
(dataset ที่นิยมที่สุดบน Kaggle ในหมวด "Diabetes Dataset")
orchestrate ด้วย **Apache Airflow**

## โครงสร้างโปรเจกต์
```
diabetes_ml_pipeline/
├── data/                       # ข้อมูลดิบ / clean / train / test
├── models/                     # โมเดลที่เทรนแล้ว + metrics
├── src/
│   ├── config.py                # path และค่าคงที่ทั้งหมด
│   ├── data_ingestion.py        # Task 1: ดึงข้อมูล
│   ├── preprocessing.py         # Task 2: clean + train/test split
│   ├── train.py                 # Task 3: เทรนโมเดล (Random Forest + GridSearch)
│   └── evaluate.py              # Task 4: ประเมินผลบน test set
├── dags/
│   └── diabetes_pipeline_dag.py # Airflow DAG เชื่อม 4 task ข้างบน
├── requirements.txt
└── README.md
```

## ⚠️ เรื่องข้อมูล (สำคัญ)
โค้ดนี้รันในสภาพแวดล้อมที่ไม่มีอินเทอร์เน็ต จึง**ดึงข้อมูลจริงจาก Kaggle ให้ไม่ได้**
`data_ingestion.py` จึงมี fallback สร้าง **synthetic dataset** ที่มีคอลัมน์และ
การกระจายของค่าใกล้เคียงกับ dataset จริง เพื่อให้ pipeline รันได้ครบ end-to-end
ให้ดูก่อนใช้งานจริง

ก่อนใช้งานจริง ให้ทำอย่างใดอย่างหนึ่ง:
1. **ดาวน์โหลดเอง**: ไปที่ https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
   แล้ววาง `diabetes.csv` ไว้ที่ `data/diabetes.csv` (ต้องมีคอลัมน์ตรงกับใน `src/config.py`)
2. **ใช้ Kaggle API อัตโนมัติ**: วาง `kaggle.json` (API token) ไว้ที่ `~/.kaggle/kaggle.json`
   แล้ว `data_ingestion.py` จะดาวน์โหลดให้เองผ่านคำสั่ง `kaggle datasets download`

โครงสร้างคอลัมน์ที่ pipeline ต้องการ:
`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

## โมเดลที่เลือก: Random Forest Classifier
เลือกให้เพราะเหมาะกับข้อมูล tabular ขนาดกลางแบบนี้ที่สุด: จับความสัมพันธ์แบบไม่เชิงเส้น
และ interaction ระหว่าง feature ได้ดี ไม่ต้อง scale ข้อมูล ทนต่อ outlier
และให้ feature importance ที่ตีความได้ในบริบทการแพทย์ (ดูเหตุผลเต็มใน `src/train.py`)
มีการทำ `GridSearchCV` (cv=5, scoring=f1) หา hyperparameter ที่ดีที่สุดให้อัตโนมัติ

## วิธีรันแบบ manual (ไม่ผ่าน Airflow) — ทดสอบ pipeline
```bash
pip install -r requirements.txt --break-system-packages
cd diabetes_ml_pipeline
python3 -m src.data_ingestion
python3 -m src.preprocessing
python3 -m src.train
python3 -m src.evaluate
```
ผลลัพธ์จากรอบทดสอบด้วย synthetic data: accuracy ≈ 0.87, f1-score ≈ 0.82, ROC-AUC ≈ 0.95
(ตัวเลขจริงจะเปลี่ยนไปตามข้อมูลจริงที่ใช้)

## วิธีรันผ่าน Airflow
```bash
# 1. ติดตั้ง Airflow (ถ้ายังไม่มี)
pip install apache-airflow --break-system-packages

# 2. ตั้งค่า AIRFLOW_HOME (เช่น export AIRFLOW_HOME=~/airflow) แล้ว init db
airflow db migrate

# 3. คัดลอกทั้งโฟลเดอร์ diabetes_ml_pipeline ไปไว้ในตำแหน่งที่ Airflow เข้าถึงได้
#    เช่น /opt/airflow/diabetes_ml_pipeline
export DIABETES_PIPELINE_DIR=/opt/airflow/diabetes_ml_pipeline

# 4. คัดลอกไฟล์ DAG เข้า Airflow DAGs folder
cp dags/diabetes_pipeline_dag.py $AIRFLOW_HOME/dags/

# 5. สร้าง user แล้วเปิด webserver + scheduler
airflow users create --username admin --password admin \
  --firstname a --lastname b --role Admin --email a@b.com
airflow webserver --port 8080 &
airflow scheduler &

# 6. เปิด http://localhost:8080 -> เปิด DAG "diabetes_ml_pipeline" -> Trigger
```

DAG จะรันตามลำดับ: `data_ingestion` → `preprocessing` → `train_model` → `evaluate_model`
ถ้า task ไหน fail Airflow จะ retry อัตโนมัติ 2 ครั้ง (ตั้งไว้ใน `default_args`)
สามารถปรับ schedule ได้ที่พารามิเตอร์ `schedule` ใน `dags/diabetes_pipeline_dag.py`
(ปัจจุบันตั้งเป็น `@weekly`)

## ผลลัพธ์ที่ได้หลังรัน
- `models/diabetes_rf_model.pkl` — โมเดลที่เทรนเสร็จ
- `models/best_params.json` — hyperparameter ที่ดีที่สุดจาก GridSearch
- `models/metrics.json` — accuracy, precision, recall, f1, ROC-AUC, confusion matrix,
  feature importance
