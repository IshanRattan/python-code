

from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from airflow.models import Variable

import pandas as pd
import json
import csv


default_args = {'owner':'ishan'}



