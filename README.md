# **Credit Card Fraud Detection: A Machine Learning Approach to Anomaly Detection and Class Imbalance Handling**
![](https://th.bing.com/th/id/R.2df7a2d83918b6371b29a9108d2167db?rik=fCuwBC5jHr2aww&riu=http%3a%2f%2fuponarriving.com%2fwp-content%2fuploads%2f2016%2f10%2fALL-CARDS.jpg&ehk=zqO%2bQduRuTivTE4PY0dTPnjAu8TY6eJjmyGJPJade9M%3d&risl=&pid=ImgRaw&r=0)

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
```

Ensure you have installed the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## **Data Overview & Preprocessing**

The dataset contains anonymized credit card transactions labeled as fraudulent or valid. After loading the data, the following steps are performed:

- Handling duplicates
- Checking for missing values
- Feature scaling
- Train-test split

To load the dataset, download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in your working directory.

```python
# Load dataset
data = pd.read_csv('creditcard.csv')

# Basic dataset information
data.info()

# Remove duplicates
data.drop_duplicates(inplace=True)
```

## **Exploratory Data Analysis**

Exploratory Data Analysis (EDA) is conducted to explore the relationships between features and the target variable, such as identifying the distribution of fraudulent and valid transactions and transaction amounts.

```python
# Example: Distribution of transaction amounts by class
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Transaction Amount by Class')

ax1.hist(Fraud['Amount'], bins=50)
ax1.set_title('Fraud')

ax2.hist(Valid['Amount'], bins=50)
ax2.set_title('Valid')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()
```

## **Feature Scaling and Train-Test Split**

```python
# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
```

## **Model Building**

We will build three anomaly detection models:

1. **Isolation Forest**
2. **Local Outlier Factor (LOF)**
3. **Support Vector Machine (SVM)**

### **6.1 Isolation Forest**

```python
# Build Isolation Forest model
isolation_forest = IsolationForest(n_estimators=100, contamination=0.00167, random_state=42)
isolation_forest.fit(X_train)

# Predict on test set
y_pred_if = isolation_forest.predict(X_test)
y_pred_if = np.where(y_pred_if == -1, 1, 0)  # Convert to 1 for fraud, 0 for valid
```

### **6.2 Local Outlier Factor (LOF)**

```python
# Build Local Outlier Factor (LOF) model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.00167)
y_pred_lof = lof.fit_predict(X_test)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)  # Convert to 1 for fraud, 0 for valid
```

### **6.3 Support Vector Machine (SVM)**

```python
# Build One-Class SVM model
svm_model = OneClassSVM(nu=0.00167, kernel='rbf', gamma='auto')
svm_model.fit(X_train)

# Predict on test set
y_pred_svm = svm_model.predict(X_test)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)  # Convert to 1 for fraud, 0 for valid
```

## **Model Evaluation**

The models are evaluated using accuracy, precision, recall, and F1 score.

```python
# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))

# Example: Evaluate Isolation Forest
evaluate_model(y_test, y_pred_if, "Isolation Forest")
```

### **7.1 Isolation Forest**

- **Accuracy**: 99.72%
- **Precision**: 18.37%
- **Recall**: 18.95%
- **F1 Score**: 18.65%

### **7.2 Local Outlier Factor (LOF)**

- **Accuracy**: 99.67%
- **Precision**: 0.00%
- **Recall**: 0.00%
- **F1 Score**: 0.00%

### **7.3 Support Vector Machine (SVM)**

- **Accuracy**: 98.84%
- **Precision**: 7.00%
- **Recall**: 48.42%
- **F1 Score**: 12.23%

## **Conclusion and Recommendations**

### **8.1 Conclusions**

1. **Isolation Forest** and **LOF** models suffer from class imbalance and fail to accurately detect fraud.
2. **SVM** performs better but still generates too many false positives.

### **8.2 Recommendations**

1. **Hyperparameter tuning** for the models to improve performance.
2. **Class imbalance handling** with techniques like SMOTE.
3. **Ensemble learning** to combine models.
4. **Feature engineering** with additional contextual data.

---

## **Dataset**

The dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Once downloaded, upload the dataset to your working environment to run the code.
Source: Available on Kaggle, this dataset contains credit card transactions made by European cardholders in September 2013. It has 284,807 transactions with only 492 fraud cases, making it ideal for anomaly detection since fraud is rare.
Features:
Time: Seconds elapsed between this transaction and the first transaction in the dataset.
V1 to V28: The result of a PCA transformation (Principal Component Analysis) applied to protect the identities of the users.
Amount: The transaction amount.
Class: The label for fraud detection. 1 represents a fraud, and 0 represents a normal transaction.
