<img width="450" height="100" alt="Screenshot 2026-08-02 122800" src="https://github.com/user-attachments/assets/93719759-37ce-49b9-9390-48390b97cac6" />

#### _Capstone Project 2021 for Specialist Diploma in Applied Artificial Intelligence_

# Streamlit-Based Diabetes Risk Classification Using Machine Learning

<img width="1200" height="675" alt="superfoods-for-diabetes" src="https://github.com/user-attachments/assets/5d1bee3c-dc88-4804-8081-f62a6ed61db1" />



## 1. Problem Statement
During the COVID-19 period, disruptions to healthcare services and reduced access to routine medical screening created challenges in identifying individuals at higher risk of diabetes. A machine-learning-based classification system can help assess diabetes risk using available demographic and health-related factors, providing a quick and accessible way to support early risk identification.

## 2. Project Objectives
1. Develop a diabetes classification model to predict whether an individual is likely to have diabetes based on relevant health and demographic features.
2. Preprocess and balance the dataset by cleaning the data, encoding categorical variables, standardizing numerical features, and applying SMOTEENN to address class imbalance.
3. Compare multiple ML classifiers, including Random Forest, Logistic Regression, and Decision Tree, using cross-validation and hyperparameter tuning.
4. Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
5. Deploy the selected model through Streamlit to provide an interactive interface for users to enter patient information and obtain a diabetes-risk classification.

## 3. Project Overview
This project develops a machine learning-based diabetes risk classification system using patient-related demographic and health information. The workflow includes data preprocessing, exploratory data analysis, class imbalance handling, feature standardization, model training and hyperparameter tuning, model evaluation, feature importance analysis, and model deployment preparation. 

Three machine learning classification algorithms were evaluated: 
- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier

The final workflow prepares the best-performing model for integration with a Streamlit application for diabetes risk prediction. 


## 4. Imported Libaries
The required Python libraries were imported to support the complete machine learning workflow. These libraries were used for:

- Data manipulation and processing
- Data visualization
- Categorical data encoding
- Feature standardization
- Handling class imbalance
- Machine learning model training
- Hyperparameter tuning
- Model evaluation
- Saving trained models for deployment

**Code Snippet:**
``` python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib 
```


## 5. Load Data
The diabetes dataset was loaded from data.csv. The initial records were displayed to understand the structure of the dataset, while the dataset information was inspected to identify the available features and their data types.

**Code Snippet:**
```python
# load the data file 
diabetes_df = pd.read_csv("data.csv")
# display rows 
print(diabetes_df.head())

# display data info 
print(diabetes_df.info())
```

## 6. Data Inspection and Cleaning
The dataset was inspected for duplicate and missing values before model development. Duplicate records were removed to reduce repeated observations, and the dataset was checked again to confirm that duplicates had been successfully removed. Missing values were also examined before proceeding with modelling.

**Code Snippet:**
```python
# check the number of duplicated rows 
print(diabetes_df.duplicated().sum())

# remove duplicateed rows 
diabetes_df.drop_duplicates(inplace = True)

# re-check the number of duplicated rows
print(diabetes_df.duplicated().sum())

# check the number of missing data in each col 
print(diabetes_df.isnull().sum())

```


## 7. Categorical Encoding
Machine learning algorithms require numerical input; therefore, categorical variables were converted into numerical representations.

The following encoding was applied:
- **gender** was converted into binary numerical values (0 and 1).
- **smoking_history** was mapped into numerical categories ranging from 0 to 5.

This transformation allowed the categorical variables to be used as input features for the machine learning models.

**Code Snippet:**
```python
# define custom function for the gender conversion 
def gender_to_numeric(gender):
    if gender == "Female":
        return 0 
    else:
        return 1 

# use apply function to transform the categorical column into a numeric column 
diabetes_df['gender'] = diabetes_df['gender'].apply(gender_to_numeric)

# check the data
print(diabetes_df.head())


# define custom function for the smoking history conversion 
def smoking_history_to_numeric(smoking):
    if smoking == "current":
        return 0
    elif smoking == "ever":
        return 1 
    elif smoking == "former":
        return 2
    elif smoking == "never":
        return 3
    elif smoking == "not current":
        return 4
    elif smoking == "No Info":
        return 5

# use apply function to transform the categorical column into a numeric column 
diabetes_df['smoking_history'] = diabetes_df['smoking_history'].apply(smoking_history_to_numeric)

# check the data
print(diabetes_df.head())

```


## 8. Exploratory Data Analysis (EDA)
Exploratory Data Analysis was conducted to examine the distributions of important features according to diabetes status. The analysis included variables such as:

- Gender
- Smoking history
- Heart disease
- Hypertension
- BMI
- Age
- HbA1c level

Visualisations were used to compare the distributions of these features between patients with and without diabetes.

**Code Snippet:**
```python
# create a contigency table for gender and diabetes
crosstab_gender_diab = pd.crosstab(diabetes_df['gender'], diabetes_df['diabetes'])
# plot and save the chart for gender
crosstab_gender_diab.plot(kind = "bar", stacked = False, figsize = (10, 5))
plt.title("Gender Distribution by Diabetes Status")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(ticks = [0, 1], labels = ['Female', 'Male'])
plt.savefig("gender_diabetes.png")
plt.close()

# plot and save the chart for age 
diabetes_df.groupby("diabetes")["age"].plot(kind = "hist", legend = True, alpha = 0.6, bins = 20)
plt.title("Age Distribution by Diabetes Status")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_diabetes.png")
plt.close()

# plot and save the chart for hypertension 
crosstab_hypertension = pd.crosstab(diabetes_df['hypertension'], diabetes_df['diabetes'])
crosstab_hypertension.plot(kind = "bar", stacked = False, figsize = (10, 5))
plt.title("Hypertension Distribution by Diabetes Status")
plt.xlabel("Hypertension")
plt.ylabel("Count")
plt.xticks(ticks = [0, 1], labels = ['No', 'Yes'])
plt.savefig("hypertension_diabetes.png")
plt.close()

# plot and save the chart for heart disease  
crosstab_heartDisease = pd.crosstab(diabetes_df['heart_disease'], diabetes_df['diabetes'])
crosstab_heartDisease.plot(kind = "bar", stacked = False, figsize = (10, 5))
plt.title("Heart Disease Distribution by Diabetes Status")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.xticks(ticks = [0, 1], labels = ['No', 'Yes'])
plt.savefig("heartDisease_diabetes.png")
plt.close()

# plot and save the chart for smoke history  
crosstab_smokeHistory = pd.crosstab(diabetes_df['smoking_history'], diabetes_df['diabetes'])
crosstab_smokeHistory.plot(kind = "bar", stacked = False, figsize = (10, 5))
plt.title("Smoking History Distribution by Diabetes Status")
plt.xlabel("Smoking History")
plt.ylabel("Count")
plt.xticks(ticks = [0, 1, 2, 3, 4, 5], labels = ['current', 'ever', 'former', 'never', 'not current', 'no info'])
plt.savefig("smokeHistory_diabetes.png")
plt.close()

# plot and save the chart for bmi
diabetes_df.groupby("diabetes")["bmi"].plot(kind = "hist", legend = True, alpha = 0.6, bins = 20)
plt.title("BMI Distribution by Diabetes Status")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.savefig("bmi_diabetes.png")
plt.close()

# plot and save the chart for hbac1 level
diabetes_df.groupby("diabetes")["HbA1c_level"].plot(kind = "hist", legend = True, alpha = 0.6, bins = 20)
plt.title("HbA1c Level Distribution by Diabetes Status")
plt.xlabel("HbA1c")
plt.ylabel("Frequency")
plt.savefig("hba1c_diabetes.png")
plt.close()

# plot and save the chart for blood glucose level
diabetes_df.groupby("diabetes")["blood_glucose_level"].plot(kind = "hist", legend = True, alpha = 0.6, bins = 20)
plt.title("Blood Glucose Level Distribution by Diabetes Status")
plt.xlabel("Blood Glucose Level")
plt.ylabel("Frequency")
plt.savefig("blood_glucose_level_diabetes.png")
plt.close()
```


## 9. Train-Test Split
The dataset was divided into:

- 80% training data
- 20% testing data

**Stratified sampling** was applied during the split to maintain a similar distribution of diabetes classes in both the training and testing datasets.

**Code Snippet:**
```python
# split the data into 80% training and 20% testing 
X_train, X_test, y_train, y_test = train_test_split(diabetes_df.drop(columns = ['diabetes'], axis = 1), 
                                                    diabetes_df['diabetes'], 
                                                    test_size = 0.20, 
                                                    shuffle = True, 
                                                    stratify = diabetes_df['diabetes'], 
                                                    random_state = 42)

```

## 10. Class Distribution Analysis
The distribution of diabetes classes in the training dataset was examined before applying any resampling technique. A bar chart was generated to compare the number of observations belonging to the non-diabetes and diabetes classes.

**Code Snippet:**
```python
# before applying resampling 
y_train.value_counts().plot(kind = "bar", color = ['crimson', 'navy'])
plt.title("Distribution of Diabetes (Before Resampling)")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.xticks(ticks = [0, 1], labels = ["No", "Yes"])
plt.savefig("before_resampling_diabetes.png")
plt.close()
```

## 11. Class Imbalance Handling
**SMOTEENN** was applied to the training dataset to address the class imbalance.

SMOTEENN combines:

- SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic observations for the minority class.
- ENN (Edited Nearest Neighbours) to remove potentially noisy or ambiguous observations.

The technique was applied only to the training data to create a more balanced dataset for model training.

**Code Snippet:**
```python 
# apply SMOTE-Tomek 
smt = SMOTEENN(random_state = 42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

# after applying resampling 
y_train_res.value_counts().plot(kind = "bar", color = ['crimson', 'navy'])
plt.title("Distribution of Diabetes (After Resampling)")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.xticks(ticks = [0, 1], labels = ["No", "Yes"])
plt.savefig("after_resampling_diabetes.png")
plt.close()
```


## 12. Feature Standardization
**StandardScaler** was applied to the following numerical features:

- Age
- BMI
- HbA1c level
- Blood glucose level

The scaler was fitted using only the resampled training data. The same fitted scaler was then used to transform the test data.

**Code Snippet:**
```python
scaler = StandardScaler()
X_train_res[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.fit_transform(X_train_res[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
X_test[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(X_test[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])

```





References:

[1] Khan, N. S., Muaz, M. H., Kabir, A., & Islam, M. N. (2017). Diabetes Predicting mHealth Application Using Machine Learning. In IEEE Xplore (pp. 237–240). https://doi.org/10.1109/WIECON-ECE.2017.8468885
