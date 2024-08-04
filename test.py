import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
credit_card_df = pd.read_csv(r"C:\Users\Akshay\Downloads\archive - 2024-08-01T171620.540\creditcard.csv")

# Separate the classes
legit = credit_card_df[credit_card_df["Class"] == 0]
fraud = credit_card_df[credit_card_df["Class"] == 1]


# Balance the dataset by sampling
legit_sample = legit.sample(n=492)
credit_card_df_balanced = pd.concat([legit_sample, fraud], axis=0)

# Separate features and target
X = credit_card_df_balanced.drop("Class", axis=1)
Y = credit_card_df_balanced["Class"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model performance
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

# Print accuracy scores for verification
print(f"Training Accuracy: {train_acc}")
print(f"Testing Accuracy: {test_acc}")

# Web app with Streamlit
st.title("Credit Card Fraud Detection Model")

# User input for prediction
input_df = st.text_input("Enter All Required Features Value (comma-separated)")
input_df_splited = input_df.split(",")

# Button for submission
submit = st.button("Submit")

# Predicting and displaying the result
if submit:
    try:
        features = np.asarray(input_df_splited, dtype=np.float64)
        prediction = model.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.write("Legitimate Transaction")
        else:
            st.write("Fraudulent Transaction")
    except ValueError as e:
        st.write("Please enter valid feature values, separated by commas.")
