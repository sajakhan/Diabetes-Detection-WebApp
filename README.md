# Diabetes-Detection-WebApp
This program applies basic machine learning concepts on kaggle  Dataset to predict if a person is diabetic or not.
----------------------------
Software and Libraries

Python 3.6.0
Sublime text
scikit-learn 0.18.1
stream lit 0.2.1

download streamlit using !pip install streamlit

Introduction


The program takes data from the training data set.
The program then divides the dataset into training and testing samples in 25:75 ratio randomly using train_test_learn() function available in sklearn module.
The program contains a get_user_data function storing input from user into dictionary.
The program then creates a Random forest classifier.
Accuracy score is then calculated by comparing with the correct results of the training dataset.
The training sample space is used to train the program and predictions are made on input provided by user

# HOW TO RUN
save the file in .py format
in your terminal type
# streamlit run (your file path)\filename.py



