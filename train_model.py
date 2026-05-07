import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load REAL cleaned data (67.87% legit, 32.13% fraud per README)
df = pd.read_csv('cleaned_fraud_data.csv')

print("Dataset shape:", df.shape)
print("Fraud %:", df['Fraud_Label'].mean()*100, "%")
print("Features:", df.columns.tolist())

# EXACT preprocessing from README
# Drop irrelevant (ID, Timestamp per analysis)
drop_cols = ['Transaction_ID', 'User_ID', 'Timestamp']  # Adjust if different names
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Encode categoricals (your maps)
le = LabelEncoder()
cat_cols = ['Transaction_Type', 'Device_Type', 'Merchant_Category', 'Authentication_Method', 'Card_Type']
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Features + target (15 features)
features = ['Transaction_Amount', 'Account_Balance', 'IP_Address_Flag', 'Previous_Fraudulent_Activity',
            'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d',
            'Card_Age', 'Transaction_Distance', 'Risk_Score', 'Is_Weekend',
            'Transaction_Type', 'Device_Type', 'Merchant_Category', 'Authentication_Method']
X = df[features].fillna(0)
y = df['Fraud_Label']

print("Final X shape:", X.shape)

# Train (XGBoost-like but RandomForest for simplicity)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f" Accuracy: {acc:.1%} | Fraud weight optimized")

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("model.pkl saved - matches app.py!")