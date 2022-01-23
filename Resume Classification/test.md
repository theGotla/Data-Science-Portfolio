# HR Analytics 

This repository contains all the files of the HR Analytics project in Purdue


# Table of Contents
- [Overview](#overview)
- [Data Dictionary](#data-dictionary)
- [Variable Descriptions](#variable-descriptions)
- [Methodology](#methodology)
  - [1. Data Processing](#1-data-pre-processing)
    - [Feature Elimination](#feature-elimination)
    - [Missing Value Imputation](#missing-value-imputations-mvi)
  - [2. Feature Engineering](#2-feature-engineering)
    - [Name Analysis](#name-analysis)
    - [Group Variables](#group-categories)
    - [Weight of Evidence](#weight-of-evidence)
  - [3. Modeling](#3-modeling)
    - [Train & Validation Split](#train-and-validation-split)
    - [Train Multiple Models](#train-multiple-models)
    - [Model Selection](#model-selection)
  - [4. Scoring](#4scoring)
  
## Overview
Our client is craigslist, we are interested in improving ‘Resumes’ sub-section of the website.The reason we chose this is because, we believe that enhancing and solving some of the problems associated with this section could give competitive advantage to our client.As for the problem statement, we saw that the interface where the resumes are present is highly unstructured. When one opens the section, it is obvious that there is a lot of scope for improvement, as it is way behind craigslist competitors. As seen in the below picture, it’s very hard to understand which resume belongs to what industry.


![Data Dictionary](Images/Unstructured_resume.PNG)

It is harder for recruiters to find what they are looking for and we know that the recruiters don’t prefer to spend much time on any website, this will make the engagement loss on both ends, from recruiters’ side and candidate’s side. There is a high probability that the churn rate can be increased because of these factors.
When one looks at Indeed, LinkedIn and Facebook marketplace, to survive in the market, it’s important to have a competitive edge.



## Methodology
Now that we have understood what the problem statement is, let us follow a methodology to solve this. We will follow the below steps
![Data Dictionary](Images/Methodology.PNG)


### 0. Data Collection
The objective is to model the natural language and tag the corresponding resumes into different categories
• To categorize the resumes, the data required for training the model is collected from two different data sources in Kaggle
• The Data has 2317 labelled resumes and a total of 37 classes in the target variable, which was scraped using a tool called ‘ParseHub’
• Those 37 variables include most from the white-collar jobs than the blue-collar jobs
• The data required for testing the model is scrapped from Craigslist

### 1. Data Pre Processing
The Data analysis is the most important step for the natural language processing models. The following steps are carried out to convert the text into numerical representation/ vector representation
#### Target Class Imbalance
We seem from the above EDA that there is a class imbalance and to solve this we used SMOTE as shown below-
![Missing](Images/SMOTE.PNG)


#### Missing Value Imputations (MVI)
For treating the missing values we have used K-Nearest Neighbours imputation with a K=6 to impute the categorical and continuous variables.This method seemed more appropriate as compared to mean imputation(for continuous variables) or mode imputation.
![Age](Images/KNN.PNG)



### 2. Modeling
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
### 3.Scoring
We now have a model, trained and validated. Recollect that we have been provided a `test` dataset to make predictions for the `future`. So we perform the same `data-preprocessing` steps on this as well and predict the `Survived` column. But, for this we can `train` our model on the `whole training` dataset and again and use that model so that we have more data to train our model.

We now `submit` the predictions and the `leaderboard score` tells the accuracy we have obtained on the test data. This whole modeling process is an `iterative` one because a `huge number parameters` are involved in the whole lifecycle.

This project has been a great starting point for me. Hopefully it is the same for the readers as well. Thanks!





