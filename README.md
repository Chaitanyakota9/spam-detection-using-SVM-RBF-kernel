Spam Email Classification Using SVM kernals

Overview

This project demonstrates the classification of emails into spam or ham (non-spam) using various Support Vector Machine (SVM) kernels. The dataset used for this task consists of labeled email messages, with "spam" indicating unwanted messages and "ham" representing legitimate ones. The primary goal is to evaluate the performance of different SVM kernels (linear, polynomial, RBF, and sigmoid) on text classification.

Features

Preprocessing of text data using TF-IDF vectorization.
Implementation of Support Vector Machine (SVM) models with different kernel functions:
Linear
Polynomial
Radial Basis Function (RBF)
Sigmoid
Evaluation metrics include:
Accuracy
Precision, Recall, and F1-Score
Classification Report for both classes (spam and ham).

Dataset

The dataset consists of two columns:

text: The email content.
label: The corresponding label (spam or ham).
Dataset Source:
Kaggle Spam Email Dataset (https://www.kaggle.com/code/prashant808/email-spam-detection-using-svm/input?select=spam.csv).
Dependencies

This project requires the following Python libraries:

pandas
numpy
scikit-learn

Install them using:

pip install pandas numpy scikit-learn

How It Works

Data Preprocessing:
The text data is converted into numerical format using TF-IDF vectorization.
Labels are mapped to binary values (spam = 1, ham = 0).

Model Training and Evaluation:
The dataset is split into training (80%) and testing (20%) sets.
SVM models with different kernels are trained on the training set and evaluated on the test set.

Performance Metrics:
The performance of each kernel is evaluated using accuracy and detailed metrics (precision, recall, F1-score).

Results

The project compares the performance of SVM models with different kernels:
Linear Kernel: Likely to perform best for text classification.
RBF Kernel: Flexible and provides good performance.
Polynomial Kernel: May overfit with high degrees.
Sigmoid Kernel: Typically less stable.

Potential Improvements

Implement hyperparameter tuning (e.g., GridSearchCV).
Explore alternative feature extraction methods (e.g., CountVectorizer).
Address class imbalance using techniques like SMOTE or class weighting.

Author

Chaitanya Kota
Master's Student in Artificial Intelligence.




Feel free to reach out for questions or feedback.

