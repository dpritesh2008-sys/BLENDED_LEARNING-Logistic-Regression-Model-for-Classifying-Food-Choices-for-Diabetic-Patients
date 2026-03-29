# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Logistic Regression is used to classify data into different categories based on input features.

2.Label Encoding converts categorical target values into numerical form for model training.

3.Min-Max Scaling normalizes feature values to improve model performance.

4.Train-Test Split divides the dataset into training and testing sets for evaluation.

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Ritesh DP
RegisterNumber: 212225040339
*/

EXP-6

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("food_items.csv")

# Inspect the dataset
print("Name: Balasurya S")
print("Reg. No: 212225100003")
print("Dataset Overview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()

# Scaling the raw input features
X = scaler.fit_transform(X_raw)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Encode the target variable
y = label_encoder.fit_transform(y_raw.values.ravel())
# Note that ravel() function flattens the vector.

# First, let's split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 123)

# L2 penalty to shrink coefficients without removing any features from the model
penalty= 'l2'

# Our classification problem is multinomial
multi_class = 'multinomial'

# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'

# Max iteration = 1000
max_iter = 1000

# Define a logistic regression model with above arguments
l2_model = LogisticRegression(random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)

# Evaluate the model
print("Name:Ritesh DP")
print("Reg. No:212225040339")
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("Name: Ritesh DP")
print("Reg. No: 212225040339")
```

## Output:
<img width="653" height="487" alt="image" src="https://github.com/user-attachments/assets/8ebc62cd-494b-463b-b887-ffc9c6caea06" />
<img width="450" height="242" alt="image" src="https://github.com/user-attachments/assets/7cde4263-6ad2-4922-850f-8bfc66893eeb" />
<img width="165" height="53" alt="image" src="https://github.com/user-attachments/assets/ec29000c-22eb-4522-8f3c-2f32987c02dc" />
<img width="161" height="37" alt="image" src="https://github.com/user-attachments/assets/81b5e1b0-357a-41f3-a79b-5f39c0d8c8b4" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
