# Risk Modelling In SAS

This repository contains all the files of the Risk Modelling Analytics project in Purdue


# Table of Contents
- [Overview](#overview)
- [Data Dictionary](#data-dictionary)
- [Variable Descriptions](#variable-descriptions)
- [Methodology](#methodology)
  - [1. Descriptive Analysis](#1-descriptive-analysis)
  - [2. Data Pre Processing](#2-data-pre-processing)
    - [Target Class Imbalance](#target-class-imbalance)
    - [Missing Value Imputations](#missing-value-imputations-(mvi))
  - [3. Modeling](#3-modeling)
    - [Train & Validation Split](#train-and-validation-split)
    - [Train Multiple Models](#train-multiple-models)
    - [Model Selection](#model-selection)
  - [4. Scoring](#4scoring)
  
## Overview
Credit Default forecast has been a subject of interest for every credit card company, and the accurate prediction on default can balance the lenders' risk and return. Based on users' historical and demographical information, lenders can modify their policy to the borrowers such as charging a higher rate for higher risks. Overall speaking, the aim of predicting credit default is to develop a predictive model that utilizes historical payment status and certain demographical information to evaluate the likeliness of credit default.

## Data Dictionary
We have been provided with a ```Train (10000 x 24)```  and ```Test (5000 x 23)``` datasets. The dependant variable is `````"Default"`````
![Data Dictionary](Images/Data_Dictionary.PNG)
  

## Methodology
Now that we have understood what the problem statement is, let us follow a methodology to solve this. 

### 1. Descriptive Analysis
Let us first look at the class distribution for the number of job seekers in our train data set as shown below-
![Missing](Images/Class_Distribution.PNG)

Furthermore lets looks at the bivariate analysis of the other varaibles with the target variable
![Missing](Images/BiVariate.PNG)


Some inferences that we can take from this graph,

-We note that most job-seekers are Male. This is not all that surprising as in this dataset Males make up the majority of the sample population.

-What is more interesting though is the City Development Index (CDI) chart. There we see that there are two peaks for job-seekers. The peaks are at high and low CDI scores.

-We can ponder why this might be. In high CDI areas perhaps there are a lot of opportunities and therefore people feel encouraged to seek better roles.

-Perhaps in lower CDI areas candidates want to improve their circumstances by searching for new jobs, maybe in new areas.

-This is all conjecture, but interesting nonetheless.

-It is also interesting to see that job-seekers have changed job more often that non-job seekers within that past 1 year, and also those that have never looked for a job also seem to be ready for a new challenge.

-However it is only the graduate level people who have more job seekers when compared to other education levels. Some are even seeking a job in their primary school! (Start networking BAIMers)

### 2. Data Pre Processing

#### Target Class Imbalance
We seem from the above EDA that there is a class imbalance and to solve this we used SMOTE as shown below-
![Missing](Images/SMOTE.PNG)


#### Missing Value Imputations (MVI)
For treating the missing values we have used K-Nearest Neighbours imputation with a K=6 to impute the categorical and continuous variables.This method seemed more appropriate as compared to mean imputation(for continuous variables) or mode imputation.
![Age](Images/KNN.PNG)



### 3. Modeling
We have `cleaned` the data and `derived` some variables so that we can make better predictions. So let us `predict` now. But we need to follow some steps to make a robust model and `avoid over-fitting` the data.

#### Train and Validation Split
The training data will be `randomly` split into `75:25` ratio into `training` and `validation` datasets. We now use the first one to train our model, and the validation data to validate our model's accuracy.
#### Train Multiple Models
I have explored `three` different techniques to train the model. Click on the links for literature review.
- [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/03/logistic-regression/)
- [Random Forest](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-random-forest-and-its-hyper-parameters/)
- [Extreme Gradient Boosting](https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/)
#### Model Selection
The performance of the above models can be judged based on the validation dataset. The results are below, so my best model is Random Forest.
```python
{
"""
Logit model validation Accuracy: 70.00%
RF model validation Accuracy: 88.1%
XGB model validation Accuracy: 83.8%  
""" 
}
```

![Age](Images/Results.PNG)
### 4.Scoring
We now have a model, trained and validated. Recollect that we have been provided a `test` dataset to make predictions for the `future`. So we perform the same `data-preprocessing` steps on this as well and predict the `Survived` column. But, for this we can `train` our model on the `whole training` dataset and again and use that model so that we have more data to train our model.

We now `submit` the predictions and the `leaderboard score` tells the accuracy we have obtained on the test data. This whole modeling process is an `iterative` one because a `huge number parameters` are involved in the whole lifecycle.

This project has been a great starting point for me. Hopefully it is the same for the readers as well. Thanks!




