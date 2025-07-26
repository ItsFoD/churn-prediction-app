# 📊 Churn Prediction Streamlit App

## 🐧 Overview

This Streamlit app predicts **telecom customer churn** using a machine learning model trained on customer behavior data. Built with ease-of-use in mind, it lets users:

* Input customer details manually or generate random examples
* View predictions on churn likelihood
* Track all past predictions in a real-time log

Developed by **Hassan Ahmed**, this project showcases applied machine learning wrapped in an interactive web interface.

---

## 🚀 Features

* 🎛️ **Interactive UI**: Easily input customer details via sliders, dropdowns, and text fields
* 🧠 **ML-Powered Predictions**: A trained classification model predicts whether a customer will churn
* 📜 **Prediction History**: View all previous predictions in a scrollable log
* 🧪 **Random Data Generation**: Auto-fill random input values to quickly test the model

---

## 📊 Machine Learning Workflow

The churn prediction model was trained and evaluated using multiple classifiers on the Telco Customer Churn dataset. Below are the top-performing models along with their recall scores:

| Model                                | Recall | Notes                                       |
| ------------------------------------ | ------ | ------------------------------------------- |
| **K-Nearest Neighbors (basic)**      | 0.8832 | Simple implementation, strong recall        |
| **Best KNN (Grid Search)**           | 0.9305 | Highest recall among all models             |
| **Random Forest**                    | 0.9093 | Strong performance across multiple metrics  |
| **Random Forest (500 trees)**        | 0.9160 | Boosted recall through increased tree depth |
| **Best Random Forest (Grid Search)** | 0.9112 | Optimized using GridSearchCV                |

✅ The model with the **highest recall** was **Best KNN (Grid Search)** with **0.9305**, making it the preferred choice for minimizing false negatives in churn prediction.

### 📈 Accuracy Comparison

| Model                                | Accuracy |
| ------------------------------------ | -------- |
| **K-Nearest Neighbors (basic)**      | 0.7926   |
| **Best KNN (Grid Search)**           | 0.8164   |
| **Random Forest**                    | 0.7945   |
| **Random Forest (500 trees)**        | 0.8012   |
| **Best Random Forest (Grid Search)** | 0.8076   |

While **Best KNN** led in recall, **Best Random Forest** and **KNN** also demonstrated strong overall accuracy.

The selected model is saved as:

📁 `best_telco_model.pkl`

---

## ⚙️ Installation & Running

```bash
# Clone the repository
https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run streamlit_churn.py
```

---

## 📁 File Structure

```
├── stremlit_churn.py       # Streamlit app script
├── best_telco_model.pkl    # Trained ML model (Best KNN)
├── telco.html              # Notebook export (EDA + training)
├── requirements.txt        # Python dependencies
└── README.md               # You're here
```

---

## 📬 Contact

**Hassan Ahmed**
Teaching Assistant, Faculty of Computers and Information Systems
Egyptian Chinese University

---

Enjoy predicting churn! 🐧
