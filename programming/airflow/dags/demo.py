

from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta
from airflow import DAG


savepath_txt = 'log.txt'

bash_cp_cmd = 'cp -r /airflow/*.py /airflow/test/'

default_args = {'owner': 'ishan',
                'start_date': datetime(2024, 3, 9)}


def save_time_stamp():
    with open(savepath_txt, 'a') as outfile:
        outfile.write(str(datetime.now()) + '\n')

dag = DAG('cp_pyfiles_save_ts',
          default_args=default_args,
          description="copy files from source to dest",
          schedule_interval=timedelta(minutes=2),
          catchup=False)

copy_files_task = BashOperator(task_id='copy_files_task',
                               bash_command=bash_cp_cmd,
                               dag=dag)


write_log = PythonOperator(task_id='save_ts',
                           python_callable=save_time_stamp,
                           dag=dag)

