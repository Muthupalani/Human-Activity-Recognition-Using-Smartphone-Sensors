# **Human Activity Recognition Using Smartphone Sensors**

This project leverages smartphone sensor data to classify human activities using machine learning techniques. It utilizes the **UCI HAR Dataset**, which contains accelerometer and gyroscope data collected from smartphone sensors during various human activities such as walking, standing, and sitting.

The primary goal is to preprocess the data, perform dimensionality reduction, train a machine learning model (Gaussian Naive Bayes), and evaluate its performance.

---

## **Table of Contents**
- [Project Features](#project-features)
- [Technologies and Libraries Used](#technologies-and-libraries-used)
- [Usage](#Usage)
-  [Dataset](#Dataset)
---

## **Project Features**

1. **Data Loading**:
   - Automatically downloads the UCI HAR Dataset.
   - Reads the training dataset (`X_train.txt` and `y_train.txt`).

2. **Exploratory Data Analysis (EDA)**:
   - Identifies missing values in the dataset.
   - Highlights feature redundancy and determines the necessity for dimensionality reduction.

3. **Data Preprocessing**:
   - Label encoding for categorical target labels.
   - Scaling numeric features using Min-Max Scaling.
   - Splits the dataset into training and test sets.

4. **Dimensionality Reduction**:
   - Implements K-Means clustering to reduce the feature set.
   - Selects representative features from clusters for improved performance.

5. **Model Training and Evaluation**:
   - Uses Gaussian Naive Bayes for activity classification.
   - Evaluates the model using accuracy score and confusion matrix.

6. **Interactive Gradio Interface**:
   - Provides a web-based application for users to experiment with dimensionality reduction (via K-Means) and view model evaluation metrics.

---

## **Technologies and Libraries Used**
- **Python**
- **Machine Learning**: scikit-learn
- **Data Handling**: pandas, numpy
- **Web Scraping**: requests, BeautifulSoup
- **Interactive UI**: Gradio
- **Visualization and Debugging**: Confusion matrix and accuracy metrics

---

## **Usage**
Clone the Repository
Install Dependencies
Run the Gradio App
Access the App

## **Dataset**
Source: UCI Machine Learning Repository
Contains accelerometer and gyroscope data from smartphones.
Six activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Lying Down.

