# 🏦 Loan Approval Prediction Using Machine Learning

Predicting loan approval status based on customer demographic and financial attributes using machine learning models like Random Forest and Gaussian Naive Bayes.

## 📌 Project Overview

This project builds a binary classification model to predict whether a loan will be approved (`Loan_Status: Y/N`) based on applicant details such as income, education, credit history, loan amount, etc. The purpose is to assist financial institutions in automating and optimizing their loan approval process.

---

## 📂 Dataset

- **Data**: (Loan_Dataset)`loan.csv`
- **Features**: 13 (both categorical and numerical)
- **Target Variable**: `Loan_Status`

### Key Features:
- Gender
- Married
- Education
- Self_Employed
- ApplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area

---

## 🔧 Data Preprocessing

- **Missing Value Handling**:
  - Categorical: Filled with mode.
  - Numerical: Filled with mean.
- **Feature Engineering**:
  - Added `LoanAmount_log` using `np.log1p()` for normalization.
- **Encoding**:
  - Label encoding for categorical variables.
- **Feature Selection**:
  - Selected relevant columns: `[1:5, 9:11, 13:15]` → total 8 key features.
- **Scaling**:
  - StandardScaler used to normalize the data.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (optional for visualizations)

---

## Conclusion

- Random Forest performed better in accuracy.
- Robust data preprocessing ensured model consistency.
- The project demonstrates a full ML pipeline for binary classification using financial data.

---

## 🧠 Model Training

### 1. **Random Forest Classifier**
 ```python 
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test) 
```

### 2. **Gaussian Naive Bayes**
```python
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
```

### 3. **Evaluation Metrics**
```python
from sklearn import metrics
print("Accuracy of Random Forest:", metrics.accuracy_score(y_test, y_pred))
print("Accuracy of GaussianNB:", metrics.accuracy_score(y_test, y_pred_nb))
```

- Random Forest Accuracy: Higher performance
- GaussianNB Accuracy: Lower but faster

---
## 📬 Contact

**Akash Kumar Rajak**  
📧 Email: [akashkumarrajak200@gmail.com](mailto:akashkumarrajak200@gmail.com)  
💼 GitHub: [AkashKumarRajak](https://github.com/AkashKumarRajak)



