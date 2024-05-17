import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Database connection function
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
    return conn

# Function to load data from the database
def load_data_from_db(conn):
    query = "SELECT * FROM your_table_name"  # Replace 'your_table_name' with the actual table name
    try:
        data = pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()
    return data

def main():
    st.title("Behavioral Use Of Tobacco")
    db_file = st.text_input('Enter database path:', '/Users/adityasrivatsav/Documents/EAS503/Group Project/tobacco_use_data.db')
    conn = create_connection(db_file)
    if conn is not None:
        data = load_data_from_db(conn)
        if not data.empty:
            st.write(data.head())
            if st.button('Load Data and Visualize'):
                st.write("Data Loaded Successfully")
                # Display visualizations
                sns.pairplot(data.select_dtypes(include=['float64', 'int']))
                plt.show()
                st.pyplot()
            if st.button('Run Model'):
                # Example: Simple model prediction
                X_train, X_test, y_train, y_test = train_test_split(data[['year']], data['data_value'], test_size=0.2)
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions)}")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
        else:
            st.error("No data available or failed to load data.")
    else:
        st.error("Failed to connect to the database.")

if __name__ == "__main__":
    main()
