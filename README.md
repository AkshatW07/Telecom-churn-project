# Project Overview

Customer churn is a major challenge in the telecom industry. This project focuses on analyzing customer behavior,identifying key churn drivers, and building a machine learning model to predict whether a customer is likely to churn. The project also includes an interactive Tableau dashboard for business insights and a deployed Streamlit web application for real-time predictions.

# Objectives
Understand factors influencing customer churn

Perform exploratory data analysis (EDA)

Build a churn prediction model

Visualize insights using Tableau

Deploy a web app for churn prediction

# Structure

Telecom_Churn_Project DA
│
├── app/
│   ├── app.py
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── data/
│   ├── churn_cleaned.csv
│   └── churn_features.csv
│
├── notebooks/
│   ├── 01_business_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_churn_modeling.ipynb
│
├── dashboard/
│   └── telecom_churn_dashboard.twb
│
├── requirements.txt
└── README.md

# Dataset
Telecom customer dataset

Contains demographic information, service usage, contract details, and churn label

# Tools & Technologies
Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Tableau

FASTAPI

GitHub


# Methodology

Business & Data Understanding

Data Cleaning

Exploratory Data Analysis

Feature Engineering

Model Building (Logistic Regression)

Dashboard Creation

Web App Deployment


# Key Insights
Month-to-month contracts have highest churn

Customers with short tenure churn more

Higher monthly charges increase churn risk

Fiber optic users show higher churn


# ML
Model: Logistic Regression

Task: Binary Classification (Churn / No Churn)

Evaluation using accuracy and classification report


# Dashboard
Interactive Tableau dashboard showing:

KPIs (Total Customers, Churn Rate, Churned Customers)

Churn by contract type, tenure group, payment method, and internet service

# Web Application
A Streamlit web app where users can enter customer details and receive churn risk predictions.
Live App Link:
(https://telecom-churn-project-cq2k8pduagoyov5uqyffht.streamlit.app/)

# Results
The project delivers a complete end-to-end churn analytics and prediction system with business insights, predictive modeling, visualization, and deployment.

# Author
Akshat Waghchoure



