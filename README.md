# AutoJudge – Programming Problem Difficulty Predictor

**Author:** Aditi Joshi (24115012)

---

## 📌 Project Overview

Online competitive programming platforms such as Codeforces, CodeChef, and Kattis categorize problems into **Easy**, **Medium**, and **Hard**, and also assign a **numerical difficulty score**.  
These labels are usually assigned manually.

This project presents **AutoJudge**, an end-to-end **machine learning system** that predicts:

- **Problem Class** → Easy / Medium / Hard *(classification task)*
- **Problem Score** 

Predictions are made **using only the textual description of a programming problem**, without relying on user submissions or metadata.

A **Streamlit-based web interface** is also provided for interactive predictions.

---

## 📊 Dataset

- **Dataset Source:**  
  https://github.com/AREEG94FAHAD/TaskComplexityEval-24

- **Dataset Size:** ~ 4k+ programming problems

### Each data sample contains:
1. `title`  
2. `description`  
3. `input_description`  
4. `output_description`  
5. `sample_io`  
6. `problem_class` (Easy / Medium / Hard)  
7. `problem_score` 

---

## 🧹 Data Preprocessing

### Text Processing
- Converted `sample_io` (list of dictionaries) into a single string field.
- Concatenated all text fields into a single feature (`raw_text`).
- Cleaned text by:
  - Lowercasing
  - Removing extra spaces, tabs, and newlines
  - Removing stopwords

### Feature Engineering
The following features were explored:
- `word_count`
- `char_count`
- `digit_count`
- `symbol_count`
- `num_sample_testcases`
- Empty-field indicators: `input_empty`, `output_empty`, `description_empty`

### Feature Selection (via EDA)
- `char_count` removed due to high correlation with `word_count`
- `digit_count` and `symbol_count` dropped due to heavy overlap across classes
- `num_sample_testcases` dropped due to weak correlation with difficulty
- `input_empty` and `output_empty` dropped due to low predictive value
- `description_empty` retained **only for regression**

### Final Features
- **Classification:** `word_count`
- **Regression:** `word_count`, `description_empty`

---

## 📈 Exploratory Data Analysis (EDA)

- Class distribution observed as: **Hard > Medium > Easy**
- Word count increases monotonically with difficulty
- Moderate positive correlation between word count and problem score
- Problems with empty descriptions tend to have lower difficulty scores



---

## 🛠 Feature Preprocessing Pipeline

A unified preprocessing pipeline was constructed using `ColumnTransformer`:

- **TF-IDF Vectorization** applied to `raw_text`
- **Standard Scaling** applied to numerical features
- Combined into a single feature matrix using `Pipeline`

Separate preprocessors were used for classification and regression tasks.

---

## 🤖 Models Used

### 🔹 Classification Models
1. **Logistic Regression** *(final choice)*
2. Linear Support Vector Machine (Linear SVM)
3. Random Forest Classifier

**Observations:**
- Logistic Regression handled sparse TF-IDF features most effectively.
- Medium and Hard classes showed overlap due to semantic similarity.
- Random Forest showed bias toward the majority class (Hard), reducing F1-score.
- Tree-based models struggled with high-dimensional sparse text representations.

---

### 🔹 Regression Models
1. Linear Regression
2. **Ridge Regression** *(final choice)*
3. Random Forest Regressor
4. Gradient Boosting Regressor

**Observations:**
- Linear Regression performed poorly due to lack of regularization.
- Ridge Regression handled high-dimensional TF-IDF features more stably.
- Tree-based regressors tended to predict values near the mean.
- Ridge Regression provided the best balance of stability and performance.

---

## 🔁 Ordinal Classification Attempt

Since problem difficulty is **ordinal (Easy < Medium < Hard)**, an ordinal approach was explored:

1. Predict `problem_score` using regression.
2. Map score to class using optimized thresholds.
3. Tune thresholds to maximize F1-score.

**Result:**  
Performance was inferior to direct classification due to noisy score predictions (MAE ≈ 1.7).  
Hence, direct classification was retained.

---

## ✅ Final Model Selection

| Task | Final Model |
|----|----|
| Problem Class | Logistic Regression |
| Problem Score | Ridge Regression |

---

## 🖥 User Interface

A **Streamlit web application** enables users to:
- Paste a programming problem description
- Receive predicted **Problem Class** and **Problem Score**


  <img width="1917" height="910" alt="image" src="https://github.com/user-attachments/assets/2606a24c-97c7-417d-9abc-258a86b9ae0b" />


---
## 📸 Screenshots

I have not put screenshots of EDA here instead i have put them on my report 

Below are the screenshots of accuracy metrics so obtained

**Logistic Regression**


<img width="703" height="551" alt="image" src="https://github.com/user-attachments/assets/f685fdbc-a476-4191-8121-a5261d09120e" />


**Linear SVMs**


<img width="641" height="570" alt="image" src="https://github.com/user-attachments/assets/262de7ef-ccd5-4f94-8cdc-131558330f32" />


**Random Forest**


<img width="468" height="435" alt="image" src="https://github.com/user-attachments/assets/c983e5b5-9947-4a46-90fa-b7de5fa72960" />


**Linear Regression**


<img width="328" height="109" alt="image" src="https://github.com/user-attachments/assets/451b388d-18bf-41be-9711-f61f11efb445" />



**Ridge Regression**


<img width="365" height="134" alt="image" src="https://github.com/user-attachments/assets/94e314b9-4d6a-42af-86ae-28e1348b0086" />



**Random Forest**


<img width="376" height="109" alt="image" src="https://github.com/user-attachments/assets/bf3f68c7-ce4c-4a42-a1b6-776474acafd5" />



**Gradient Boosting**


<img width="335" height="121" alt="image" src="https://github.com/user-attachments/assets/8479f50f-7143-4525-88f6-019026637b78" />



## Running the Project Locally
```bash
git clone https://github.com/aditijoshi786/auto-judge-problem-classifier
```

```bash
cd auto-judge-problem-classifier
```
```bash
python -m venv venv
```

**for windows**

```bash
venv\Scripts\activate
```
**for mac**
```bash
source venv/bin/activate
```
```bash

pip install -r requirements.txt
```
```bash
streamlit run app.py
```

## DEMO VIDEO

https://drive.google.com/file/d/15KLi2WuiEnZ-IlR6Wt4G6Ko4SDRKxA5h/view?usp=sharing








