import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
from scipy.stats import randint, uniform

# Load dataset
df = pd.read_csv("kubernetes_performance_metrics_dataset.csv")

# Convert timestamp column to numeric
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').astype('int64') // 10**9  # Convert to Unix time

# Drop unnecessary columns
drop_columns = ['pod_name'] if 'pod_name' in df.columns else []
df.drop(columns=drop_columns, errors='ignore', inplace=True)

# Identify numerical and categorical columns
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Handle missing values
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

# Encode categorical variables
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature].astype(str))
    label_encoders[feature] = le

# Feature scaling
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define target variable (binary classification)
latency_threshold = np.percentile(df['network_latency'], 75)  # 75th percentile
df['network_issue'] = (df['network_latency'] > latency_threshold).astype(int)
X = df.drop(columns=['network_latency', 'network_issue'])
y = df['network_issue']

# Handle class imbalance using SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(sampling_strategy=0.7, random_state=42)  # Balance the minority class to 70% of the majority class
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:\n", pd.Series(y_train).value_counts())

# RandomizedSearchCV for hyperparameter tuning
param_dist = {
    'n_estimators': randint(100, 500),  # Randomly sample between 100 and 500
    'max_depth': randint(3, 10),        # Randomly sample between 3 and 10
    'learning_rate': uniform(0.01, 0.2),  # Randomly sample between 0.01 and 0.2
    'subsample': uniform(0.8, 0.2),     # Randomly sample between 0.8 and 1.0
    'colsample_bytree': uniform(0.8, 0.2),  # Randomly sample between 0.8 and 1.0
    'gamma': uniform(0, 0.2),           # Randomly sample between 0 and 0.2
    'reg_lambda': uniform(1, 2),        # Randomly sample between 1 and 3
    'scale_pos_weight': [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]  # Handle class imbalance
}

xgb_model = XGBClassifier(random_state=42, n_jobs=-1)  # Use all CPU cores
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=100,  # Increase the number of iterations
    scoring='roc_auc',  # Use ROC-AUC for imbalanced datasets
    cv=5,       # Increase the number of folds
    random_state=42,
    n_jobs=-1   # Use all CPU cores
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best model from random search
best_xgb_model = random_search.best_estimator_

# Evaluate model
cv_scores = cross_val_score(best_xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
print("Cross-validation ROC-AUC:", np.mean(cv_scores))

y_pred = best_xgb_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
with open('best_xgb_model.pkl', 'wb') as f:
    pickle.dump(best_xgb_model, f)