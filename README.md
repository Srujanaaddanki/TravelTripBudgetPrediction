# ✈️ Travel Trip Budget Prediction  
### A Smart Machine Learning–Based Travel Cost Estimation System

Travel planning often involves rough guesses and fixed per-day assumptions, which may not reflect real-world expenses.  
This project solves that problem by using **machine learning and data analysis** to predict a realistic **total travel budget** based on multiple influencing factors.

---

<img width="959" height="495" alt="image" src="https://github.com/user-attachments/assets/3bba5cec-cfcd-4670-b330-a675b4d3001d" />

## 📌 Project Overview

**Travel Trip Budget Prediction** is a machine learning project that estimates the **total cost of a trip** using historical travel data.  
Instead of relying on assumptions, the system uses patterns learned from data to provide accurate and practical budget predictions.

The project also includes an **interactive web application** that allows users to input their travel details and instantly receive an estimated budget.

---

<img width="960" height="499" alt="image" src="https://github.com/user-attachments/assets/f6204be9-8c7d-443d-a317-a36976e5a7b3" />


## 🎯 Objectives

- To predict total travel budget using machine learning techniques  
- To analyze historical travel data and identify cost-influencing factors  
- To compare multiple regression models and select the best-performing one  
- To deploy the trained model as a user-friendly web application  
- To assist users in planning trips with realistic budget estimates  

---

## 📂 Dataset Description

The dataset contains historical travel information with the following key features:

- **Source & Destination**
- **Season**
- **Month**
- **Trip Duration (Days)**
- **Trip Type** (Solo, Family, Friends, etc.)
- **Hotel Quality**
- **Target Variable:** Total Trip Cost (₹)

---

## 🧹 Data Cleaning & Preprocessing

- Removed missing and inconsistent values  
- Standardized categorical features  
- Handled outliers to avoid skewed predictions  
- Applied label encoding for categorical variables  
- Prepared data for model training  

---

## 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand cost patterns and relationships:
- Distribution of total trip cost  
- Season-wise impact on travel budget  
- Hotel quality vs cost comparison  
- Trip duration vs total cost relationship  
- Monthly average travel cost analysis  

These insights helped in feature selection and model choice.

---

## 🤖 Machine Learning Models Used

The following regression models were trained and compared:

1. **Linear Regression** – Baseline model  
2. **Decision Tree Regressor** – Captures non-linear patterns  
3. **Random Forest Regressor** – Ensemble model with high stability  

Models were evaluated using the **R² score** metric.

---

## 🏆 Final Model Selection

**Random Forest Regressor** was selected as the final model because:
- It captures complex, non-linear relationships  
- Reduces overfitting using ensemble learning  
- Produces stable and reliable predictions  
- Performed best compared to other models  

---

## 🌐 Web Application

The final model is deployed using **Streamlit**, allowing users to:
- Select destination, season, month, trip type, hotel quality  
- Adjust trip duration using a slider  
- Instantly view the estimated total travel budget  

This makes the project practical and user-friendly.

---

## 🚀 Future Scope

- Include distance-based travel cost estimation  
- Integrate real-time transport and hotel pricing APIs  
- Improve location detection accuracy  
- Expand dataset with real-world booking data  

---

## 🧠 Key Learnings

- Practical application of data preprocessing and EDA  
- Understanding model comparison and selection  
- Importance of visualization in decision-making  
- Deploying machine learning models as real-world applications  

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  

---

## 🔗 Access Links

| Resource | Link |
|----------|------|
| 💻 GitHub Repository | https://github.com/Srujanaaddanki/TravelTripBudgetPrediction |
| 📁 Dataset | https://docs.google.com/spreadsheets/d/1j_kxCGl5NBICDFKOrWRxWOOJ3uc6MGg5/edit?gid=159424351#gid=159424351 |
| 💼 LinkedIn Post | https://www.linkedin.com/feed/update/urn:li:activity:7409296852480606208/ |

---


## 👩‍💻 Author

**Srujana Addanki**  
B.Tech (CSE) | Machine Learning & Data Analytics Enthusiast  

---

⭐ *If you like this project, feel free to star the repository!*
