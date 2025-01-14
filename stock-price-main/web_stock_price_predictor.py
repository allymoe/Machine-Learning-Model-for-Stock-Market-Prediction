import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import hashlib
import sqlite3

# Function to connect to the SQLite database
def connect_db():
    conn = sqlite3.connect("users.db")  # Creates the database file if it doesn't exist
    return conn

# Function to create the users table if it doesn't exist
def create_user_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Utility function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to register a new user in the database
def register_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    try:
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

# Function to authenticate a user
def authenticate_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Initialize the database and create the user table
create_user_table()

# Login/Registration
def login_page():
    st.title("Welcome to the Stock Prediction App")
    st.subheader("Login to Your Account")

    if "register_mode" not in st.session_state:
        st.session_state["register_mode"] = False

    if not st.session_state["register_mode"]:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if authenticate_user(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success(f"Welcome back, {username}!")
                else:
                    st.error("Invalid username or password")

        st.markdown("Don't have an account?")
        if st.button("Register Here"):
            st.session_state["register_mode"] = True  # Switch to registration mode
    else:
        st.subheader("Register for an Account")
        with st.form("register_form"):
            reg_username = st.text_input("Choose a Username", key="register_username")
            reg_password = st.text_input("Choose a Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
            submitted = st.form_submit_button("Register")
            if submitted:
                if reg_password != confirm_password:
                    st.error("Passwords do not match")
                elif register_user(reg_username, reg_password):
                    st.success("Registration successful! You can now log in.")
                    st.session_state["register_mode"] = False  # Switch back to login mode
                else:
                    st.error("Username already exists")

        if st.button("Back to Login"):
            st.session_state["register_mode"] = False  # Switch back to login mode

# Main App Logic
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    # Add custom CSS to position the logout button
    st.markdown("""
        <style>
            .logout-button {
                position: absolute;
                top: 20px;
                left: 20px;
                z-index: 10;
            }
        </style>
    """, unsafe_allow_html=True)

    # Place the logout button at the top-left corner
    if st.button("Logout", key="logout", help="Click to log out", on_click=lambda: st.session_state.clear()):
        st.session_state.clear()
        st.success("You have been logged out.")

    st.title(f"Welcome, {st.session_state['username']}!")

    stock = st.text_input("Enter the Stock ID", "GOOG")
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)

    # Fetch stock data
    google_data = yf.download(stock, start, end)
    model = load_model("Latest_stock_price_model.keras")

    st.subheader("Stock Data")
    st.write(google_data)

    splitting_len = int(len(google_data) * 0.7)
    x_test = pd.DataFrame(google_data.Close[splitting_len:])

    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
        fig = plt.figure(figsize=figsize)
        plt.plot(values, 'Orange')
        plt.plot(full_data.Close, 'b')
        if extra_data:
            plt.plot(extra_dataset)
        return fig

    st.subheader('Original Close Price and MA for 250 days')
    google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 200 days')
    google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 100 days')
    google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

    # Preprocessing for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Predictions
    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Plotting predictions
    ploting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index=google_data.index[splitting_len + 100:]
    )

    st.subheader("Original values vs Predicted values")
    st.write(ploting_data)

    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
    plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)
