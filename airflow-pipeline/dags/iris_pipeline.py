from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd 
default_args={
    'start_date': datetime(2025,5,5)
}

def load_data():
    df=pd.read_csv('/airflow-pipeline/data/iris.csv')
    df.to_csv('/airflow-pipeline/data/iris_loaded.csv')


def process_data():
    df=pd.read_csv('/airflow-pipeline/data/iris.csv')
    df['sepal_area']=df['sepal length (cm)']* df['sepal width (cm)']
    df.to_csv('/airflow-pipeline/data/iris_processed.csv')

with DAG(
    dag_id='iris_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)as dag:

    t1=PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    t2=PythonOperator(
        task_id='process_data',
        python_callable=process_data
    )

    t1 >> t2
