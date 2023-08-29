#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 2: Define the Time Series Conversion Function
def time_series_conversion(df, t=5):
    data = df.copy()

    for i in range(1, t+1):
        temp = df['strain(micro)'].shift(i)  # 'strain(micro)' column represents the strain
        data[f'Strain_Lag_{i}'] = temp

    return data.dropna()

# Load the XGBoost model
model = xgb.XGBRegressor()

def main():
    st.title("Force Recovery Model")

    # Upload Train and Test Data CSV files
    st.sidebar.header("Upload Data")
    train_file = st.sidebar.file_uploader("Upload Train Data", type=["csv"])
    test_file = st.sidebar.file_uploader("Upload Test Data", type=["csv"])

    if train_file is not None and test_file is not None:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Rest of the preprocessing steps...
        scaler = MinMaxScaler()

        # Scale the features in train_data and test_data
        train_data_scaled = scaler.fit_transform(train_data.drop(columns=['F']))
        test_data_scaled = scaler.transform(test_data.drop(columns=['F']))

        # Step 2: Apply Time Series Conversion
        t = 5  # Set the time lag
        train_data_converted = time_series_conversion(train_data, t=t)
        test_data_converted = time_series_conversion(test_data, t=t)

        # Step 3: Data Splitting
        X_train = train_data_converted.drop(columns=['F'])  # Features without the 'F' column
        y_train = train_data_converted['F']  # Target is the 'F' column

        X_test = test_data_converted.drop(columns=['F'])
        y_test = test_data_converted['F']

        # Fit and transform the scaler on the training data features
        X_train = scaler.fit_transform(X_train)

        # Transform the test data features using the same scaler
        X_test = scaler.transform(X_test)

        # Initialize another MinMaxScaler for the target variable
        target_scaler = MinMaxScaler()

        # Fit and transform the scaler on the training data target
        y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1))

        # Transform the test data target using the same scaler
        y_test = target_scaler.transform(y_test.values.reshape(-1, 1))

        # Fit the XGBoost model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Display RMSE and Graph
        st.header("Model Evaluation")
        st.write(f"Root Mean Squared Error: {rmse}")

        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data_converted['time '], y_test, label='Original Force')
        ax.plot(test_data_converted['time '], y_pred, label='Predicted Force', linestyle='dashed')
        ax.set_xlabel('Time')
        ax.set_ylabel('Force')
        ax.set_title('Original vs Predicted Force over Time')
        ax.legend()
        st.pyplot(fig)

        # Download CSV
        with st.expander("Download Recovered Force Data"):
            download_button = st.button("Download CSV")
            if download_button:
                # Save the recovered force as a CSV
                recovered_force_df = pd.DataFrame({
                    'time': test_data_converted['time '],
                    'Recovered_Force': y_pred
                })
                csv_filename = "recovered_force.csv"
                csv = recovered_force_df.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv, file_name=csv_filename)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




