# Carnival Wars Dataset
###### HackerEarth Machine Learning Hackathon

# Contents
```
├── Carnival_Wars_Solved.ipynb
├── Dataset
│   └── sample_submission.csv
│   └── test.csv
│   └── train.csv
└── README.md
```
# Installation
- Python 3
- Jupyter Notebok
Make sure the following Libraries are Installed
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable)

# Data Selection
This Data was provided by [HackerEarth](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-selling-price/), which was a part of this Hackathon.

# Preprocessing data:
Data Various Data Preprocessing Techniques are used to Clean the Data.
- Removal of NAs.
- Handling the Categorical data with Dummy Varibles and Count Encoding.
- Extracting informative data from instock_date column like month, dayofweek, weekofyear using pandas.
# Visualization
- For this part, I used Seaborn and Matplotlib which are most popular visualisation libraries available.
- To detect outliers and get better insites from the Data.

# Modeling 
- I used different machine learning regressors like Linear Regression, Lasso Regression, Tree-based regressors, CatBoost Regressor and LightGBM Regressor.
- Among that CatBoost performed the Best.
- After that I took Voting of 2 CatBoost, 1 LightGBM, 1 Decision Tree, which gave even less overall MSE (Mean Square Error).

# Ranking 
![alt text](https://github.com/omkarbhad/Carnival-Wars-Hackathon/blob/main/hackathon_result.jpg)


