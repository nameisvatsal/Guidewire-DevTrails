# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("kubernetes_performance_dataset.csv")

# Convert Timestamp to datetime and set as index
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)

# Define features to scale
features_to_scale = ['CPU_Usage(%)', 'Memory_Usage(MB)', 'Disk_IO_Latency(ms)',
                     'Pods_Running', 'Request_Limit_Ratio', 'Network_Throughput(Mbps)',
                     'Latency(ms)', 'Packet_Loss(%)', 'Request_Response_Time(ms)']

# Scale the data using StandardScaler
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[features_to_scale] = scaler.fit_transform(data_scaled[features_to_scale])

# Define thresholds for failure classification
data_scaled['Issue'] = ((data['CPU_Usage(%)'] > 90) |
                        (data['Memory_Usage(MB)'] > 7000) |
                        (data['Pod_Restarts'] > 0) |
                        (data['Connection_Errors'] > 0) |
                        (data['Packet_Loss(%)'] > 0.1) |
                        (data['Latency(ms)'] > 80)).astype(int)

# Define features and target
X = data_scaled.drop(columns=['Issue'])
y = data_scaled['Issue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ MODEL 1: XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print(f"🎯 XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")

# ✅ MODEL 2: LightGBM Classifier
lgbm_model = LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_preds)
print(f"⚡️ LightGBM Accuracy: {lgbm_accuracy * 100:.2f}%")

# ✅ MODEL 3: CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=200, learning_rate=0.1, random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)
catboost_preds = catboost_model.predict(X_test)
catboost_accuracy = accuracy_score(y_test, catboost_preds)
print(f"🐱 CatBoost Accuracy: {catboost_accuracy * 100:.2f}%")

# ✅ MODEL 4: Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"🕹️ SVM Accuracy: {svm_accuracy * 100:.2f}%")

# ✅ MODEL 5: Isolation Forest (for anomaly detection)
iso_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_model.fit(X_train)
iso_preds = iso_model.predict(X_test)
# Map anomaly labels to binary (1: anomaly, 0: normal)
iso_preds = np.where(iso_preds == -1, 1, 0)
iso_accuracy = accuracy_score(y_test, iso_preds)
print(f"🔎 Isolation Forest (Anomaly Detection) Accuracy: {iso_accuracy * 100:.2f}%")

# 🎉 Final Model Evaluation
print("\n📚 XGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))

print("\n⚡️ LightGBM Classification Report:")
print(classification_report(y_test, lgbm_preds))

print("\n🐱 CatBoost Classification Report:")
print(classification_report(y_test, catboost_preds))

print("\n🕹️ SVM Classification Report:")
print(classification_report(y_test, svm_preds))

# 🎯 Feature importance from XGBoost
feature_importance = xgb_model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance from XGBoost')
plt.show()

# ✅ Save the Best Model (Choose based on highest accuracy)
best_model = None
best_accuracy = max(xgb_accuracy, lgbm_accuracy, catboost_accuracy, svm_accuracy)

if best_accuracy == xgb_accuracy:
    best_model = xgb_model
    print("🎯 XGBoost Selected as Best Model!")
elif best_accuracy == lgbm_accuracy:
    best_model = lgbm_model
    print("⚡️ LightGBM Selected as Best Model!")
elif best_accuracy == catboost_accuracy:
    best_model = catboost_model
    print("🐱 CatBoost Selected as Best Model!")
else:
    best_model = svm_model
    print("🕹️ SVM Selected as Best Model!")

# Save the selected best model
import joblib
joblib.dump(best_model, 'kubernetes_failure_model.pkl')
print("🚀 Best Model saved as 'kubernetes_failure_model.pkl'")