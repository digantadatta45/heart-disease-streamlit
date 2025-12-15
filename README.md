🎓 Heart Disease Prediction Streamlit App  
This is an interactive Streamlit app built to demonstrate machine learning models on a heart disease dataset. It predicts the likelihood of heart disease based on user-selected patient features.

🚀 Live Demo  
[Open the app](https://YOUR_STREAMLIT_APP_URL)

💻 Features  
- Dataset preview (first 10 rows shown for convenience)  
- Data cleaning: handles invalid values, outliers, and categorical encoding  
- Model evaluation metrics: accuracy, precision, recall, F1-score, ROC AUC  
- Confusion matrix with TP, FP, FN, TN visualized  
- Feature importance analysis with Random Forest  
- Interactive Sigmoid curve and probability visualization  
- Multiple ML models: Logistic Regression, Random Forest, Gradient Boosting, SVM  

🧠 How It Works  
- ML models predict the probability of heart disease for each patient.  
- Probability is a number between 0 and 1, showing the model's confidence.  
- Class prediction converts probability to 0 (healthy) or 1 (disease) using a threshold (0.5).  
- Interactive plots and EDA allow exploration of features and correlations.  

🛠️ Run Locally  
1. Download or clone this repository  
```bash
git clone https://github.com/digantadatta45/heart-disease-streamlit.git
cd heart-disease-streamlit
