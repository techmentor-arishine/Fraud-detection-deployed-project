import pickle
import numpy as np
import streamlit as st

def encode_inputs(Transaction_Amount, Account_Balance, IP_Address_Flag, Previous_Fraudulent_Activity,
                  Daily_Transaction_Count, Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d,
                  Card_Age, Transaction_Distance, Risk_Score, Is_Weekend,
                  Transaction_Type, Device_Type, Merchant_Category, Authentication_Method):
    # EXACT 15 FEATURES (removed Card_Type to match model)
    maps = {
        'Transaction_Type': {"POS": 0, "Online": 1, "ATM Withdrawal": 2, "Bank Transfer": 3},
        'Device_Type': {"Mobile": 0, "Laptop": 1, "Tablet": 2},
        'Merchant_Category': {"Clothing": 0, "Electronics": 1, "Travel": 2, "Restaurants": 3},
        'Authentication_Method': {"OTP": 0, "Biometric": 1, "Password": 2}
    }
    
    return np.array([
        float(Transaction_Amount), float(Account_Balance), float(IP_Address_Flag), float(Previous_Fraudulent_Activity),
        float(Daily_Transaction_Count), float(Avg_Transaction_Amount_7d), float(Failed_Transaction_Count_7d),
        float(Card_Age), float(Transaction_Distance), float(Risk_Score), float(Is_Weekend),
        maps['Transaction_Type'][Transaction_Type],
        maps['Device_Type'][Device_Type],
        maps['Merchant_Category'][Merchant_Category],
        maps['Authentication_Method'][Authentication_Method]
    ]).reshape(1, -1)

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        st.success(f"Model OK: expects {model.n_features_in_} features")
        return model
    except:
        st.error("model.pkl missing")
        return None

st.set_page_config(layout="wide")
st.title("Fraud Detection")

model = load_model()
if model is None:
    st.stop()

# Sidebar (15 features only)
st.sidebar.header("Transaction")
Transaction_Amount = st.sidebar.number_input("Amount", value=100.0)
Account_Balance = st.sidebar.number_input("Balance", value=5000.0)
IP_Address_Flag = st.sidebar.selectbox("IP Flag", [0, 1])
Previous_Fraudulent_Activity = st.sidebar.selectbox("Past Fraud", [0, 1])
Daily_Transaction_Count = st.sidebar.number_input("Daily Count", value=5)
Avg_Transaction_Amount_7d = st.sidebar.number_input("Avg 7d", value=75.0)
Failed_Transaction_Count_7d = st.sidebar.number_input("Failed 7d", value=0)
Card_Age = st.sidebar.number_input("Card Age", value=365)
Transaction_Distance = st.sidebar.number_input("Distance km", value=15.0)
Risk_Score = st.sidebar.slider("Risk Score", 0.0, 1.0, 0.5)
Is_Weekend = st.sidebar.selectbox("Weekend", [0, 1])

st.sidebar.header("Categorical")
Transaction_Type = st.sidebar.selectbox("Type", ["POS", "Online", "ATM Withdrawal", "Bank Transfer"])
Device_Type = st.sidebar.selectbox("Device", ["Mobile", "Laptop", "Tablet"])
Merchant_Category = st.sidebar.selectbox("Merchant", ["Clothing", "Electronics", "Travel", "Restaurants"])
Authentication_Method = st.sidebar.selectbox("Auth", ["OTP", "Biometric", "Password"])

# Layout
col1, col2 = st.columns(2)
with col1:
    st.metric("Amount", f"${Transaction_Amount:,.0f}")
    st.metric("Balance", f"${Account_Balance:,.0f}")
with col2:
    st.metric("Risk Score", f"{Risk_Score:.3f}")
    st.metric("Daily Count", Daily_Transaction_Count)

if st.button("PREDICT", type="primary"):
    input_data = encode_inputs(
        Transaction_Amount, Account_Balance, IP_Address_Flag, Previous_Fraudulent_Activity,
        Daily_Transaction_Count, Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d,
        Card_Age, Transaction_Distance, Risk_Score, Is_Weekend,
        Transaction_Type, Device_Type, Merchant_Category, Authentication_Method
    )
    
    st.info(f"Input shape: {input_data.shape}")
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    c1, c2 = st.columns(2)
    c1.metric("Fraud Prob", f"{prob:.1%}")
    if prediction:
        c2.error("🚨 FRAUD")
    else:
        c2.success("✅ LEGIT")