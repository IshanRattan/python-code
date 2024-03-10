
# from airflow.providers.slack.notifications.slack import send_slack_notification
from airflow.models.connection import Connection
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.slack.operators.slack import SlackAPIOperator, SlackAPIPostOperator


from datetime import datetime, timedelta
from airflow import DAG


savepath_txt = '/airflow/test/log.txt'
bash_cp_cmd = 'cp -r /airflow/*.py /airflow/test/'

default_args = {'owner': 'ishan',
                'start_date': datetime(2024, 3, 9)}

def save_time_stamp():
    with open(savepath_txt, 'a') as outfile:
        outfile.write(str(datetime.now()) + '\n')

dag = DAG('cp_pyfiles_save_ts',
          default_args=default_args,
          description="copy files from source to dest",
          schedule_interval=timedelta(minutes=1),
          catchup=False,
    )

copy_files_task = BashOperator(task_id='copy_files_task',
                               bash_command=bash_cp_cmd,
                               dag=dag)

write_log = PythonOperator(task_id='save_ts',
                           python_callable=save_time_stamp,
                           dag=dag)

slack_conn = SlackAPIOperator(slack_conn_id='slack_conn',
                              task_id='conn')


copy_files_task >> write_log
