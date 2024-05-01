import streamlit as st
import pandas as pd
import snowflake.connector
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix

# Snowflake Connection Details
SNOWFLAKE_ACCOUNT = 'wotfamo-bs88857'
SNOWFLAKE_USER = 'Rutika'
SNOWFLAKE_PASSWORD = 'Rutika@12'
SNOWFLAKE_WAREHOUSE = 'COMPUTE_WH'
SNOWFLAKE_DATABASE = 'INTELLIGENT_CLIENT_MANAGEMENT'
SNOWFLAKE_SCHEMA = 'DATA'
snowflake_role = 'ACCOUNTADMIN'



# Connect to Snowflake
@st.cache(hash_funcs={snowflake.connector.connection.SnowflakeConnection: lambda _: None})
def get_snowflake_connection():
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        role = snowflake_role,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        client_session_keep_alive = True,
         proxy_host='proxy_host',
        proxy_port='proxy_port'
    )
    return conn

# Fetch data from Snowflake
def fetch_data(query):
    with get_snowflake_connection() as conn:
        return pd.read_sql(query, conn)

# Function to preprocess data
def preprocess_data(df):
    # Dummy preprocessing: fill NA and encode categorical features
    df.fillna(-999, inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load or define your model here
    # Placeholder for xgboost model loading
    model = xgb.Booster()
    model.load_model("path_to_your_model_file.model")  # Update with your model path
    return model

# Main app
def main():
    st.title("Customer Service Recommendation System")
    
    customer_code = st.text_input("Enter Customer Code:", "")
    if st.button("Recommend Services"):
        if customer_code:
            data_query = f"""
            SELECT * FROM TRAIN_DATA WHERE CustomerCode = '{customer_code}'
            """
            customer_data = fetch_data(data_query)

            if not customer_data.empty:
                processed_data = preprocess_data(customer_data)
                model = load_model()
                dmatrix = DMatrix(processed_data)
                predictions = model.predict(dmatrix)
                
                # Process the predictions to readable format (e.g., extracting top predictions)
                top_services = predictions.argsort()[-5:][::-1]  # Example: Get top 5 predicted indices
                st.write(f"Top Recommended Services for Customer {customer_code}:")
                for i in top_services:
                    st.write(f"- {customer_data.columns[i]}")
            else:
                st.error("No data found for this customer.")
        else:
            st.error("Please enter a valid Customer Code.")

if __name__ == "__main__":
    main()
