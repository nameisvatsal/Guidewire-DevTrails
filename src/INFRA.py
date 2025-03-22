# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Upload the dataset manually from local system
uploaded = files.upload()

# Load the uploaded dataset
filename = list(uploaded.keys())[0]
data = pd.read_csv(filename)

print("✅ Dataset loaded successfully!")
print(f"Available columns: {data.columns.tolist()}")

# Strip whitespaces from column names
data.columns = data.columns.str.strip()

# Check for the correct target column
if 'Failure_Class' not in data.columns:
    raise ValueError("❌ Column 'Failure_Class' not found. Check the dataset for correct column names!")

# Handle missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)  # Fill categorical with mode
    else:
        data[col].fillna(data[col].median(), inplace=True)  # Fill numeric with median

print("✅ Missing values handled correctly!")

# Encode categorical variables using LabelEncoder
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print(f"✅ Categorical columns encoded: {categorical_cols.tolist()}")

# Define features and target
X = data.drop(['Failure_Class'], axis=1)
y = data['Failure_Class']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Dataset split successfully!")

# Build a Random Forest model to predict failure and identify risky uptimes
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Model predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📚 Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Plot
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('🔎 Feature Importance in Predicting Failures')
plt.show()

# Identify cases causing frequent failures
failure_cases = data[data['Failure_Class'] == 1]['Event_Type'].value_counts().head(5)
print("🚨 Top 5 Causes of Frequent Failures:\n", failure_cases)

# Identify risky uptimes likely to cause failures
uptime_risk = data.groupby('Uptime (mins)')['Failure_Class'].mean().sort_values(ascending=False).head(5)
print("⏰ Uptime Intervals with Highest Failure Risk:\n", uptime_risk)

# Save the model to a file for future use
import pickle
with open('k8s_failure_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved successfully as 'k8s_failure_model.pkl'")