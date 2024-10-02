# **Credit Card Fraud Detection: A Machine Learning Approach to Anomaly Detection and Class Imbalance Handling**

## **Overview**

This project aims to detect fraudulent credit card transactions using machine learning anomaly detection models. The primary challenge is the highly imbalanced nature of the dataset, where fraudulent transactions constitute a very small portion of the total transactions. To address this, various techniques, including feature scaling, class imbalance handling, and different model selection, are employed to improve detection performance.

## **Table of Contents**

1. [Introduction](#introduction)
2. [Libraries and Dependencies](#libraries-and-dependencies)
3. [Data Overview & Preprocessing](#data-overview--preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Scaling and Train-Test Split](#feature-scaling-and-train-test-split)
6. [Model Building](#model-building)
7. [Model Evaluation](#model-evaluation)
   - [7.1 Isolation Forest](#71-isolation-forest)
   - [7.2 Local Outlier Factor (LOF)](#72-local-outlier-factor-lof)
   - [7.3 Support Vector Machine (SVM)](#73-support-vector-machine-svm)
8. [Conclusion and Recommendations](#conclusion-and-recommendations)
   - [8.1 Conclusions](#81-conclusions)
   - [8.2 Recommendations](#82-recommendations)

## **Introduction**

Fraudulent credit card transactions pose a serious risk to businesses and consumers. This project leverages machine learning anomaly detection models, including Isolation Forest, Local Outlier Factor (LOF), and Support Vector Machine (SVM), to detect fraudulent transactions in a highly imbalanced dataset. The dataset comes from the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## **Libraries and Dependencies**

To run this project, you need to install and import the following libraries:

```python
# Libraries for data manipulation
import pandas as pd
import numpy as np

# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Libraries for evaluation
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Libraries for handling class imbalance
from imblearn.over_sampling import SMOTENC
