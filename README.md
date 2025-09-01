# Parkinson-Disease-Prediction-using-Machine-Learning-Python
📖 Overview
The Parkinson’s Disease Prediction project applies machine learning models to predict whether a person has Parkinson’s disease using biomedical voice and signal features.
The workflow includes data cleaning, feature selection, handling class imbalance, and model training to deliver reliable predictions.

# 🚀 Features
*📊 Dataset Preprocessing – Cleaning, aggregation, and normalization of patient data.

*🔎 Feature Selection – Chi-square test to select top biomarkers.

*⚖️ Class Imbalance Handling – Oversampling minority cases with SMOTE/RandomOverSampler.

*🤖 Machine Learning Models – Logistic Regression, Support Vector Classifier, XGBoost.

*📈 Evaluation Metrics – ROC AUC, confusion matrix, classification report.

# 🛠️ Tech Stack
Language: Python (Google Colab / Jupyter)
Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn
Models Used: Logistic Regression, Support Vector Classifier, XGBoost
📂 Project Structure
project/ │── Parkinson_Disease_Prediction.ipynb # Colab notebook │── parkinson_disease.csv # Dataset │── outputs/ # Plots, metrics, confusion matrix │── README.md

# ▶️ Workflow

1.Data Loading & Exploration – Inspect dataset structure, missing values, and distributions.

2.Data Cleaning & Wrangling – Aggregated patient records, removed highly correlated features.

3.Feature Selection – Selected top 30 most relevant biomarkers using Chi-square test.

4.Data Splitting – Train-test split (80-20).

5.Class Balancing – Oversampled minority class.

6.Model Training & Evaluation – Trained Logistic Regression, SVC, and XGBoost.

7.Performance Analysis – Logistic Regression achieved best validation accuracy (~86%).

# 📊 Results

Logistic Regression – Accuracy: 86%
XGBoost – Validation Accuracy: ~65% (overfit)
SVC – Validation Accuracy: ~64%
Confusion Matrix showed good detection of Parkinson’s patients with minor false negatives.
📸 Visuals
ROC AUC curves
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
🔮 Future Enhancements
🧠 Apply deep learning models (ANN, CNN, RNN).
🌍 Use larger datasets for improved generalization.
📱 Deploy as a Flask/Streamlit web app for doctors & researchers.
🔊 Explore voice-based real-time prediction systems.
📜 License
This project is licensed under the MIT License – free to use and mod

