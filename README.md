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
- gender was converted into binary numerical values (0 and 1).
- smoking_history was mapped into numerical categories ranging from 0 to 5.

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



References:

[1] Khan, N. S., Muaz, M. H., Kabir, A., & Islam, M. N. (2017). Diabetes Predicting mHealth Application Using Machine Learning. In IEEE Xplore (pp. 237–240). https://doi.org/10.1109/WIECON-ECE.2017.8468885
