import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, precision_recall_curve
)
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

# Load data
credit_risk = pd.read_csv("UCI_credit_card.csv")
df = credit_risk.copy()

# Drop irrelevant column
df.drop(["ID"], axis=1, inplace=True)

# Independent and dependent features
X = df.drop(['default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Handle class imbalance with SMOTE
print("Before oversampling:", Counter(y_train))
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After oversampling:", Counter(y_train))

# Logistic Regression
logit = LogisticRegression()
logit.fit(X_train, y_train)
pred_logit = logit.predict(X_test)

# Logistic Regression evaluation
print("The accuracy of logit model is:", accuracy_score(y_test, pred_logit))
print(classification_report(y_test, pred_logit))

# Plot confusion matrix for Logistic Regression
ConfusionMatrixDisplay.from_estimator(logit, X_test, y_test, cmap="Blues")
plt.show()

# Plot precision-recall curve for Logistic Regression
precision, recall, _ = precision_recall_curve(y_test, logit.predict_proba(X_test)[:, 1])
plt.plot(recall, precision, label="Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# Random Forest evaluation
print("The accuracy of Random Forest Classifier is:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)
xgb_predict = xgb_clf.predict(X_test)

# XGBoost evaluation
print("The accuracy of XGB Classifier model is:", accuracy_score(y_test, xgb_predict))
print(classification_report(y_test, xgb_predict))

# Plot confusion matrix for XGBoost
ConfusionMatrixDisplay.from_estimator(xgb_clf, X_test, y_test, cmap="Blues")
plt.show()

# Plot precision-recall curve for XGBoost
precision, recall, _ = precision_recall_curve(y_test, xgb_clf.predict_proba(X_test)[:, 1])
plt.plot(recall, precision, label="XGBoost")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Hyperparameter optimization for XGBoost using RandomizedSearchCV
params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "reg_lambda": [3, 4, 5, 6, 8, 10, 12, 15],
    "subsample": [0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

random_search = RandomizedSearchCV(
    xgb_clf, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3
)
print("Fitting the RandomizedSearchCV")
random_search.fit(X_train, y_train)

# Best parameters and estimator from RandomizedSearchCV
print("Best estimator:", random_search.best_estimator_)
print("Best parameters:", random_search.best_params_)

# XGBoost Classifier with best parameters
classifierXGB = xgb.XGBClassifier(
    objective='binary:logistic',
    gamma=random_search.best_params_['gamma'],
    learning_rate=random_search.best_params_['learning_rate'],
    max_depth=random_search.best_params_['max_depth'],
    reg_lambda=random_search.best_params_['reg_lambda'],
    min_child_weight=random_search.best_params_['min_child_weight'],
    subsample=random_search.best_params_['subsample'],
    colsample_bytree=random_search.best_params_['colsample_bytree'],
    use_label_encoder=False
)

classifierXGB.fit(X_train, y_train)
y_pred = classifierXGB.predict(X_test)

# Cross-validation score
score = cross_val_score(classifierXGB, X, y, cv=10)
print(f"\n\nCross-Validation Scores: {score}")
print(f"Mean of the scores: {score.mean()}")
