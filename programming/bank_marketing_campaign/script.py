
import pandas as pd
import numpy as np

infile = pd.read_csv("bank_marketing.csv")
infile.head()


client = infile[['client_id', 'age', 'job', 'marital', 'education', 'credit_default', 'mortgage']]
client['job'] = client['job'].str.replace('.', '_')
client['education'] = client['education'].str.replace('.', '_').replace("unknown", np.NaN)
client['mortgage'] = client['mortgage'].replace('no', 'false').replace('yes', 'true').replace('unknown', 'false')
client['mortgage'] = client['mortgage'].map({'false': False, 'true': True})
client['credit_default'] = client['credit_default'].replace('no', 'false').replace('yes', 'true').replace('unknown', 'false')
client['credit_default'] = client['credit_default'].map({'false': False, 'true': True})
client.to_csv('client.csv')


