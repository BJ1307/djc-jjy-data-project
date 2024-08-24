"""
This is a Random Forest Model trained by the steam reviews 
on https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_business.json
This project is helped by learning from Rob Mulla 
from https://www.youtube.com/watch?v=QpzMWQvxXWk
and from https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial

And inspired by Nicholas Renotte https://www.youtube.com/watch?v=szczpgOEdXs
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Cleaning the data
# This part of code is adapted from (helped by) https://www.kaggle.com/code/dansbecker/random-forests
data = pd.read_csv("/Users/ericdai/Documents/GitHub/djc-jjy-data-project/archive/all_reviews/all_reviews.csv", nrows=100)
y = data.copy().voted_up
cols_used = []
X = data.copy()[cols_used]
# 
# pass

# # Building the Model
RFRegressor = RandomForestRegressor(n_estimators=100,random_state=1)



