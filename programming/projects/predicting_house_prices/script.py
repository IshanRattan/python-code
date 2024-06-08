
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns

data = pd.read_csv(
    '/Users/ishanrattan/Desktop/Study/github/python-code/programming/predicting_house_prices/data/taiwan_house_prices.csv')

data_to_pred = pd.DataFrame({'n_convenience' : np.arange(-5, -2)})

model = ols('price_twd_msq ~ n_convenience', data=data).fit()
new_data = data_to_pred.assign(price_twd_msq = model.predict(data_to_pred))

sns.regplot(x = 'n_convenience', y = 'price_twd_msq', data=data)
sns.scatterplot(x = 'n_convenience', y = 'price_twd_msq', data=new_data, color='red')
plt.show()
