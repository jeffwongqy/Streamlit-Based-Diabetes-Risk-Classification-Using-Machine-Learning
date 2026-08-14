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

# load the data file 
diabetes_df = pd.read_csv("data.csv")
# display rows 
print(diabetes_df.head())

# display data info 
print(diabetes_df.info())

# check the number of duplicated rows 
print(diabetes_df.duplicated().sum())

# remove duplicateed rows 
diabetes_df.drop_duplicates(inplace = True)

# re-check the number of duplicated rows
print(diabetes_df.duplicated().sum())

# check the number of missing data in each col 
print(diabetes_df.isnull().sum())

######################## convert categorical variable to a numeric one ########################
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

######################## investigate association with the target variable ########################
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

######################## split dataset ########################
# split the data into 80% training and 20% testing 
X_train, X_test, y_train, y_test = train_test_split(diabetes_df.drop(columns = ['diabetes'], axis = 1), 
                                                    diabetes_df['diabetes'], 
                                                    test_size = 0.20, 
                                                    shuffle = True, 
                                                    stratify = diabetes_df['diabetes'], 
                                                    random_state = 42)


######################## investigate distribution of target variables ########################

# before applying resampling 
y_train.value_counts().plot(kind = "bar", color = ['crimson', 'navy'])
plt.title("Distribution of Diabetes (Before Resampling)")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.xticks(ticks = [0, 1], labels = ["No", "Yes"])
plt.savefig("before_resampling_diabetes.png")
plt.close()

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

######################## standardization of variables ########################
scaler = StandardScaler()
X_train_res[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.fit_transform(X_train_res[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
X_test[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(X_test[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])

######################## model building via grid-search CV ########################
# initialize stratified k-fold
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

# random forest classifier
param_grid_rfc = {'n_estimators': [10, 20, 30, 40, 50], 
                  'max_depth': [3, 5, 7]}
gs_rfc = GridSearchCV(estimator = RandomForestClassifier(random_state = 42), 
                        param_grid = param_grid_rfc, 
                        cv = cv, 
                        scoring = "accuracy", 
                        n_jobs = 1, 
                        return_train_score = True, 
                        verbose = 3)
gs_rfc.fit(X_train_res, y_train_res) 


# logistic regression 
param_grid_lg = {'C': [0.1, 0.01, 1]}
gs_lg = GridSearchCV(estimator = LogisticRegression(random_state = 42), 
                      param_grid = param_grid_lg, 
                      cv = cv, 
                      scoring = "accuracy", 
                      n_jobs = 1, 
                      return_train_score = True, 
                      verbose = 3)
gs_lg.fit(X_train_res, y_train_res)

# decision trees classifier
param_grid_dtc = { 'max_depth': [3, 5, 7]}
gs_dtc = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 42), 
                      param_grid = param_grid_dtc, 
                      cv = cv, 
                      scoring = "accuracy", 
                      n_jobs = 1, 
                      return_train_score = True, 
                      verbose = 3)
gs_dtc.fit(X_train_res, y_train_res)


######################## model evaluation ########################

# perform prediction on train set and test set
y_pred_rfc_train = gs_rfc.predict(X_train_res)
y_pred_rfc_test = gs_rfc.predict(X_test)

y_pred_lg_train = gs_lg.predict(X_train_res)
y_pred_lg_test = gs_lg.predict(X_test)

y_pred_dtc_train = gs_dtc.predict(X_train_res)
y_pred_dtc_test = gs_dtc.predict(X_test)


# classification report for random forest classifier
print("Classification Report for Random Forest Classifier")
print("==================================================================")
print("TRAIN: ")
print(classification_report(y_train_res, y_pred_rfc_train))
print("TEST: ")
print(classification_report(y_test, y_pred_rfc_test))

# classification report for logistic regression
print("Classification Report for Logistic Regression")
print("==================================================================")
print("TRAIN: ")
print(classification_report(y_train_res, y_pred_lg_train))
print("TEST: ")
print(classification_report(y_test, y_pred_lg_test))

# classification report for decision tree classifier
print("Classification Report for Random Forest Classifier")
print("==================================================================")
print("TRAIN: ")
print(classification_report(y_train_res, y_pred_dtc_train))
print("TEST: ")
print(classification_report(y_test, y_pred_dtc_test))

# confusion matrix for random forest classifier
disp_rfc_train = ConfusionMatrixDisplay.from_predictions(y_train_res, y_pred_rfc_train)
plt.title("Confusion Matrix For Random Forest Classifier (TRAIN)")
plt.tight_layout()
plt.savefig("confusion_matrix_rfc_train.png")
plt.close()

disp_rfc_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rfc_test)
plt.title("Confusion Matrix For Random Forest Classifier (TEST)")
plt.tight_layout()
plt.savefig("confusion_matrix_rfc_test.png")
plt.close()



# confusion matrix for logistic regression
disp_lg_train = ConfusionMatrixDisplay.from_predictions(y_train_res, y_pred_lg_train)
plt.title("Confusion Matrix For Logistic Regression (TRAIN)")
plt.tight_layout()
plt.savefig("confusion_matrix_lg_train.png")
plt.close()

disp_lg_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lg_test)
plt.title("Confusion Matrix For Logistic Regression (TEST)")
plt.tight_layout()
plt.savefig("confusion_matrix_lg_test.png")
plt.close()


# confusion matrix for Decision Trees 
disp_dtc_train = ConfusionMatrixDisplay.from_predictions(y_train_res, y_pred_dtc_train)
plt.title("Confusion Matrix For Decision Tree Classifier (TRAIN)")
plt.tight_layout()
plt.savefig("confusion_matrix_dtc_train.png")
plt.close()

disp_dtc_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dtc_test)
plt.title("Confusion Matrix For Decision Tree Classifier (TEST)")
plt.tight_layout()
plt.savefig("confusion_matrix_dtc_test.png")
plt.close()

# feature importances
best_rfc = gs_rfc.best_estimator_

feature_importances = pd.DataFrame({"Feature": diabetes_df.drop(columns = ["diabetes"], axis = 1).columns, 
                                    "Importances": best_rfc.feature_importances_})

feature_importances = feature_importances.sort_values(by = "Importances", ascending = False)

feature_importances.plot(kind = "bar")
plt.title("Random Forest Importance Features")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png")
plt.close()

# save the models and scalers
joblib.dump(gs_rfc.best_estimator_, "random_forest_model.joblib")
joblib.dump(scaler, "scaler.joblib")


