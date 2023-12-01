
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


campaign = infile[['client_id', 'number_contacts', 'contact_duration', 'previous_campaign_contacts', 'previous_outcome', 'campaign_outcome', 'month', 'day']]
campaign['previous_outcome'] = campaign['previous_outcome'].str.replace('nonexistent', 'failure')
campaign['previous_outcome'] = campaign['previous_outcome'].map({'failure':False, 'success':True})
campaign['campaign_outcome'] = campaign['campaign_outcome'].map({'no':False, 'yes':True})
campaign['last_contact_date'] = '2022-' + campaign['month'].astype('str') + '-' + campaign['day'].astype('str')
campaign['last_contact_date'] = pd.to_datetime(campaign['last_contact_date'], format="%Y-%b-%d")
campaign.drop(['month', 'day'], axis=1, inplace=True)
campaign.to_csv('campaign.csv')


