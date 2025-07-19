# Heart Disease Prediction

This project predicts the presence of heart disease using classical machine learning models and a neural network. The goal is to analyze patient health data and identify whether an individual is likely to have heart disease based on 13 medical attributes.

---

## Dataset

The dataset includes 303 patient records with 14 attributes:

- **Features**: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia
- **Target**: `0` (no disease) or `1` (disease)

Source: kaggle

---

## Models Used

### Classical Machine Learning Models
- Logistic Regression
- Naive Bayes
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine (SVM)

### Hyperparameter Tuning

Used GridSearchCV for:
Logistic Regression: Tuning regularization parameter C
K-Nearest Neighbors: Tuning n_neighbors and weights

### Deep Learning
- Feedforward Neural Network using TensorFlow/Keras
  - 2 hidden layers (16 + 8 neurons), dropout regularization
  - Trained with RMSprop and Binary Crossentropy

---

## Model Performance

| Model                  | Accuracy  | Precision   | Recall   | F1-Score |
|------------------------|-----------|-------------|----------|----------|
| Logistic Regression    | 82.44     | 83.24       | 82.44    | 82.32    |
| Random Forest          | 99.02     | 98.09       | 100      | 99.04    |
| K-Nearest Neighbors    | 100       | 100         | 100      | 100      |
| Naive Bayes            | 81.46     | 78.76       | 86.40    | 82.40    |
| Extreme Gradient Boost | 94.63     | 95.10       | 94.17    | 94.63    |
| Decision Tree          | 98.05     | 96.26       | 100      | 98.10    |
| Support Vector Machine | 86.34     | 82.60       | 92.23    | 87.15    |
| Neural Network         | 90.73     | 87.50       | 95.15    | 91.10    |

---

## Project Structure

├── Iteration_1.ipynb # Full notebook with EDA, ML models, tuning, and DL
├── heart.csv # Dataset
└── README.md # Project documentation

## Results Summary

- Best performing classical model: **Random Forest** with 99.02% accuracy
- Best overall score: **K-Nearest Neighbors** (100% on test set)
- Neural Network also performed well with **91.10% F1 Score**

## Notes

- Feature scaling was applied to improve performance.
- Categorical features were encoded using one-hot encoding.
- Train/Val/Test split: 60/20/20