

from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.models import Variable

import pandas as pd
import json
import csv


default_args = {'owner':'ishan'}


@dag(dag_id='data_transformation_storage_pipeline',
     description='Data transformation and storage pipeline.',
     default_args=default_args,
     start_date=days_ago(1),
     schedule_interval='@once',
     tags=['sqlite', 'python', 'xcoms', 'taskflow_api'])
def data_transformation_storage_pipeline():

    @task
    def read_dataset():
        df = pd.read_csv('/root/airflow/datasets/car_data.csv')
        return df.to_json()

    @task(task_id='create_table')
    def create_table():
        sqlite_operator = SqliteOperator(
            sql_conn_id = 'my_sqlite_conn',
            sql = """CREATE TABLE IF NOT EXISTS car_data (
            id INTEGER PRIMARY KEY,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            body_style TEXT NOT NULL,
            seat INTEGER NOT NULL,
            price INTEGER NOT NULL);""")

        sqlite_operator.execute(context=None)


