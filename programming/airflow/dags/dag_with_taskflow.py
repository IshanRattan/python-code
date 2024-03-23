
import time
from datetime import datetime, timedelta

from airflow.utils.dates import days_ago
from airflow.decorators import dag, task

default_args = {'owner':'ishan'}

@dag(dag_id='dag_with_taskflow',
     description='DAG using Taskflow API',
     default_args=default_args,
     start_date=days_ago(1),
     schedule_interval='@once',
     tags=['dependencies', 'python', 'taskflow_api'])
def dag_with_taskflow_api():

    @task
    def task_a():
        print("TASK A executed!")

    @task
    def task_b():
        time.sleep(2)
        print("TASK B executed!")

    @task
    def task_c():
        time.sleep(2)
        print("TASK C executed!")

    @task
    def task_d():
        time.sleep(2)
        print("TASK D executed!")

    @task
    def task_e():
        print("TASK E executed!")

    task_a() >> [task_b(), task_c(), task_d()] >> task_e()


dag_with_taskflow_api()