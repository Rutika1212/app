import streamlit as st
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

# Snowflake connection parameters
conn_params = {
    "account": st.secrets["snowflake"]["BS88857"],
    "user": st.secrets["snowflake"]["Rutika"],
    "password": st.secrets["snowflake"]["Rutika@12"],
    "role": st.secrets["snowflake"]["ACCOUNTADMIN"],
    "warehouse": st.secrets["snowflake"]["COMPUTE_WH"],
    "database": "INTELLIGENT_CLIENT_MANAGEMENT",
    "schema": "DATA"
}

session = Session.builder.configs(conn_params).create()

# Function to read data from Snowflake
def read_snowflake_data(table_name):
    df = session.table(table_name).to_pandas()
    return df

# Main application function
def app():
    st.title('Intelligent Client Management System')

    # User input
    customer_code = st.text_input("Enter Customer Code:")
    if customer_code:
        st.write(f"Hello {customer_code}")
        # Load and prepare the data
        if st.button("Load Data and Train Model"):
            train_data = read_snowflake_data("TRAIN_DATA")
            test_data = read_snowflake_data("TEST_DATA")
            recommended_products = model(train_data, test_data, customer_code)
            st.write("Recommended Services:")
            st.write(recommended_products)

def model(train_data, test_data, customer_code):
    # Target columns as per your specification
    target_cols = ['SAVINGACCOUNT', 'CURRENTACCOUNTS', 'JUNIORACCOUNT', 'SHORTTERMOEPOSITS',
                   'MEDIUMTERMDEPOSITS', 'LONGTERMDEPOSITS', 'FUNDS', 'LOANS', 
                   'CREDITCARD', 'PENSIONS2']
    
    # Assuming train_data and test_data already include necessary preprocessing and feature selection
    y = train_data['target']
    train_data = train_data.drop(columns=['target'])

    # XGB Model Setup
    xgb_params = {
        'booster': 'gbtree',
        'max_depth': 6,
        'nthread': 4,
        'num_class': len(target_cols),
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    # Prepare data for XGB
    dtrain = xgb.DMatrix(train_data, label=y)
    dtest = xgb.DMatrix(test_data)

    # Training the model
    bst = xgb.train(xgb_params, dtrain, num_boost_round=100)

    # Making predictions
    predictions = bst.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in predictions])

    # For this example, let's assume the test data has the customer code and we just filter to get recommendations
    # This part needs to match your actual data structure
    customer_data = test_data[test_data['CustomerCode'] == customer_code]
    if not customer_data.empty:
        customer_preds = predictions[customer_data.index]
        top_services = np.argsort(-customer_preds)[:7]  # Top 7 services
        return [target_cols[i] for i in top_services]
    else:
        return ["No data available for this customer"]

streamlit run st.py
