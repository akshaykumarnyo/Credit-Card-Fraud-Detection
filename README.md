**Credit Card Fraud Detection Model**
This project implements a credit card fraud detection model using Logistic Regression, and provides a user-friendly web interface built with Streamlit to predict whether a transaction is legitimate or fraudulent. The model is trained and evaluated on a balanced dataset to ensure robust performance.

Features
Dataset: The dataset contains credit card transactions, including both legitimate and fraudulent transactions.
Model: Uses Logistic Regression for classification.
Balancing: The dataset is balanced by sampling to ensure fair training of the model.
Web Interface: Built with Streamlit, allowing users to input feature values and get real-time predictions.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install the required libraries:

bash
Copy code
pip install pandas scikit-learn streamlit
Download the dataset:

Ensure you have the creditcard.csv file in the appropriate directory (C:\Users\Akshay\Downloads\archive - 2024-08-01T171620.540\).
Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Interact with the Web App:

Enter the feature values of a credit card transaction (comma-separated) into the input field.
Click "Submit" to get the prediction.
The app will display whether the transaction is "Legitimate" or "Fraudulent."
Model Performance
Training Accuracy: The accuracy score of the model on the training dataset.
Testing Accuracy: The accuracy score of the model on the testing dataset.
Data Handling
Balancing the Dataset: The dataset is balanced by sampling to address class imbalance issues.
Feature Input: Ensure that feature values are entered in the correct format and order.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
Dataset: Credit card transactions dataset used in this project.
