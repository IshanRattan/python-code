#!/usr/bin/env python
# coding: utf-8

# # Sowing Success: How Machine Learning Helps Farmers Select the Best Crops
# 
# ![Farmer in a field](farmer_in_a_field.jpg)
# 
# Measuring essential soil metrics such as nitrogen, phosphorous, potassium levels, and pH value is an important aspect of assessing soil condition. However, it can be an expensive and time-consuming process, which can cause farmers to prioritize which metrics to measure based on their budget constraints.
# 
# Farmers have various options when it comes to deciding which crop to plant each season. Their primary objective is to maximize the yield of their crops, taking into account different factors. One crucial factor that affects crop growth is the condition of the soil in the field, which can be assessed by measuring basic elements such as nitrogen and potassium levels. Each crop has an ideal soil condition that ensures optimal growth and maximum yield.
# 
# A farmer reached out to you as a machine learning expert for assistance in selecting the best crop for his field. They've provided you with a dataset called `soil_measures.csv`, which contains:
# 
# - `"N"`: Nitrogen content ratio in the soil
# - `"P"`: Phosphorous content ratio in the soil
# - `"K"`: Potassium content ratio in the soil
# - `"pH"` value of the soil
# - `"crop"`: categorical values that contain various crops (target variable).
# 
# Each row in this dataset represents various measures of the soil in a particular field. Based on these measurements, the crop specified in the `"crop"` column is the optimal choice for that field.  
# 
# In this project, you will apply machine learning to build a multi-class classification model to predict the type of `"crop"`, while using techniques to avoid multicollinearity, which is a concept where two or more features are highly correlated.



# All required libraries are imported here for you.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
crops = pd.read_csv("soil_measures.csv")
crops['crop'] = crops['crop'].astype('category')
crops['crop_code'] = crops['crop'].cat.codes

# Check NA vals
print(crops.isna().sum())

X_train, X_test, y_train, y_test = train_test_split(crops.drop(['crop', 'crop_code'], axis=1),
                                                    crops['crop_code'],
                                                    test_size=.2,
                                                    random_state=42)

# Iterate over each feature to build a model
for feature in ['N', 'P', 'K', 'ph']:
    log_reg = LogisticRegression(max_iter=2000, multi_class='multinomial')
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])  # Fixed the error here
    feature_performance = f1_score(y_pred, y_test, average='weighted')
    print(f"F1-score for {feature}: {feature_performance}")

# Corr heatmap
sns.heatmap(crops.corr(), annot=True)

# Final_features = ["N", "K", "ph"] based on heatmap
# Split data into train, test batches
X_train, X_test, y_train, y_test = train_test_split(crops.drop(['crop', 'P', 'crop_code'], axis=1), crops['crop_code'], test_size=.2, random_state=42)

# Initiate logistic regression as log_reg
log_reg = LogisticRegression(max_iter=2000, multi_class='multinomial')

# Train the model using features as X_train and target as y_train
log_reg.fit(X_train, y_train)

# Predictions using test data
y_pred = log_reg.predict(X_test)
model_performance = f1_score(y_pred, y_test, average='weighted')
print(f"F1-score : {model_performance}")


