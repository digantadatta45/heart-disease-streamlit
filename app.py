import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, classification_report)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #ff7f0e;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("❤ Heart Disease Prediction System")
st.markdown("### An End-to-End Machine Learning Application")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", 
                        ["📊 Data Overview", 
                         "🔍 EDA", 
                         "🤖 Model Training", 
                         "📈 Model Evaluation",
                         "🔮 Prediction"])

# Load and prepare data
@st.cache_data
def load_data():
    # Simulating the cleaned dataset structure
    np.random.seed(42)
    n_samples = 918
    
    data = {
        'Age': np.random.randint(28, 78, n_samples),
        'Sex': np.random.randint(0, 2, n_samples),
        'RestingBP': np.random.randint(90, 171, n_samples),
        'Cholesterol': np.random.randint(135, 347, n_samples),
        'FastingBS': np.random.randint(0, 2, n_samples),
        'MaxHR': np.random.randint(66, 203, n_samples),
        'ExerciseAngina': np.random.randint(0, 2, n_samples),
        'Oldpeak': np.random.uniform(0, 3.75, n_samples),
        'HeartDisease': np.random.randint(0, 2, n_samples),
        'ChestPainType_ATA': np.random.randint(0, 2, n_samples),
        'ChestPainType_NAP': np.random.randint(0, 2, n_samples),
        'ChestPainType_TA': np.random.randint(0, 2, n_samples),
        'RestingECG_Normal': np.random.randint(0, 2, n_samples),
        'RestingECG_ST': np.random.randint(0, 2, n_samples),
        'ST_Slope_Flat': np.random.randint(0, 2, n_samples),
        'ST_Slope_Up': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

df = load_data()

# Page 1: Data Overview
if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1] - 1)
    with col3:
        st.metric("Heart Disease Cases", df['HeartDisease'].sum())
    with col4:
        st.metric("Healthy Cases", len(df) - df['HeartDisease'].sum())
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), width="stretch")
    
    st.subheader("Data Types and Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        st.write("*Data Types:*")
        st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']), width="stretch")
    with col2:
        st.write("*Missing Values:*")
        st.dataframe(pd.DataFrame(df.isnull().sum(), columns=['Missing']), width="stretch")

# Page 2: EDA
elif page == "🔍 EDA":
    st.header("🔍 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Target Distribution", "Feature Distributions", "Correlations", "Feature Importance"])
    
    with tab1:
        st.subheader("Heart Disease Distribution")
        
        fig = make_subplots(rows=1, cols=2, 
                           specs=[[{"type": "pie"}, {"type": "bar"}]],
                           subplot_titles=("Distribution", "Count"))
        
        disease_counts = df['HeartDisease'].value_counts()
        
        fig.add_trace(go.Pie(labels=['No Disease', 'Disease'], 
                            values=disease_counts.values,
                            marker=dict(colors=['#2ecc71', '#e74c3c']),
                            hole=0.4), row=1, col=1)
        
        fig.add_trace(go.Bar(x=['No Disease', 'Disease'], 
                            y=disease_counts.values,
                            marker=dict(color=['#2ecc71', '#e74c3c'])), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Disease Percentage", f"{disease_counts[1]/len(df)*100:.1f}%")
        with col2:
            st.metric("Healthy Percentage", f"{disease_counts[0]/len(df)*100:.1f}%")
    
    with tab2:
        st.subheader("Feature Distributions")
        
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=(f"{selected_feature} Distribution", 
                                         f"{selected_feature} by Heart Disease"))
        
        fig.add_trace(go.Histogram(x=df[selected_feature], 
                                  name=selected_feature,
                                  marker_color='#3498db',
                                  nbinsx=30), row=1, col=1)
        
        fig.add_trace(go.Box(y=df[df['HeartDisease']==0][selected_feature], 
                            name='No Disease',
                            marker_color='#2ecc71'), row=1, col=2)
        fig.add_trace(go.Box(y=df[df['HeartDisease']==1][selected_feature], 
                            name='Disease',
                            marker_color='#e74c3c'), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, width="stretch")
    
    with tab3:
        st.subheader("Correlation Heatmap")
        
        correlation = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(height=700, title="Feature Correlation Matrix")
        st.plotly_chart(fig, width="stretch")
        
        st.subheader("Top Correlations with Heart Disease")
        disease_corr = correlation['HeartDisease'].sort_values(ascending=False)[1:]
        
        fig = go.Figure(go.Bar(
            x=disease_corr.values,
            y=disease_corr.index,
            orientation='h',
            marker=dict(color=disease_corr.values, 
                       colorscale='RdYlGn',
                       reversescale=True)
        ))
        fig.update_layout(height=500, title="Feature Correlation with Heart Disease")
        st.plotly_chart(fig, width="stretch")
    
    with tab4:
        st.subheader("Feature Importance Analysis")
        
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=feature_importance['Importance'],
            y=feature_importance['Feature'],
            orientation='h',
            marker=dict(color=feature_importance['Importance'],
                       colorscale='Viridis')
        ))
        fig.update_layout(height=600, title="Feature Importance (Random Forest)")
        st.plotly_chart(fig, width="stretch")

# Page 3: Model Training
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    st.sidebar.subheader("Model Configuration")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Train",
        ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"],
        default=["Logistic Regression", "Random Forest"]
    )
    
    if st.sidebar.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            X = df.drop('HeartDisease', axis=1)
            y = df['HeartDisease']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {}
            results = {}
            
            if "Logistic Regression" in selected_models:
                models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=random_state)
            if "Random Forest" in selected_models:
                models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=random_state)
            if "Gradient Boosting" in selected_models:
                models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
            if "SVM" in selected_models:
                models['SVM'] = SVC(probability=True, random_state=random_state)
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                results[name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
            
            st.session_state['models'] = models
            st.session_state['results'] = results
            st.session_state['X_test'] = X_test_scaled
            st.session_state['y_test'] = y_test
            st.session_state['scaler'] = scaler
            
            st.success("✅ Models trained successfully!")
            
            st.subheader("Model Performance Comparison")
            
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [r['accuracy'] for r in results.values()],
                'Precision': [r['precision'] for r in results.values()],
                'Recall': [r['recall'] for r in results.values()],
                'F1 Score': [r['f1_score'] for r in results.values()],
                'ROC AUC': [r['roc_auc'] for r in results.values()]
            })
            
            st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']), 
                        width="stretch")
            
            fig = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
            
            for model_name in results.keys():
                values = [results[model_name]['accuracy'],
                         results[model_name]['precision'],
                         results[model_name]['recall'],
                         results[model_name]['f1_score'],
                         results[model_name]['roc_auc']]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model_name
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500,
                title="Model Performance Radar Chart"
            )
            st.plotly_chart(fig, width="stretch")

# Page 4: Model Evaluation
elif page == "📈 Model Evaluation":
    st.header("📈 Model Evaluation")
    
    if 'results' not in st.session_state:
        st.warning("⚠ Please train models first in the 'Model Training' page.")
    else:
        results = st.session_state['results']
        y_test = st.session_state['y_test']
        
        selected_model = st.selectbox("Select Model for Detailed Evaluation", list(results.keys()))
        
        result = results[selected_model]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{result['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{result['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{result['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{result['f1_score']:.3f}")
        with col5:
            st.metric("ROC AUC", f"{result['roc_auc']:.3f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, result['y_pred'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                height=400,
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            st.plotly_chart(fig, width="stretch")
            
            tn, fp, fn, tp = cm.ravel()
            st.write(f"*True Negatives:* {tn} | *False Positives:* {fp}")
            st.write(f"*False Negatives:* {fn} | *True Positives:* {tp}")
        
        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, result['y_pred_proba'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{selected_model} (AUC = {result["roc_auc"]:.3f})',
                line=dict(color='#e74c3c', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                height=400,
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig, width="stretch")
        
        st.subheader("Sigmoid Function Visualization")
        
        X_range = np.linspace(-10, 10, 300)
        sigmoid = 1 / (1 + np.exp(-X_range))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=X_range,
            y=sigmoid,
            mode='lines',
            name='Sigmoid Function',
            line=dict(color='#3498db', width=4)
        ))
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Decision Boundary (0.5)")
        fig.add_vline(x=0, line_dash="dash", line_color="green",
                     annotation_text="x = 0")
        
        fig.add_trace(go.Scatter(
            x=[-5, 5],
            y=[1/(1+np.exp(5)), 1/(1+np.exp(-5))],
            mode='markers',
            name='Key Points',
            marker=dict(size=12, color='red')
        ))
        
        fig.update_layout(
            height=500,
            title="Sigmoid Activation Function: σ(x) = 1 / (1 + e^(-x))",
            xaxis_title="Input (x)",
            yaxis_title="Probability Output σ(x)",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        st.info("📊 The sigmoid function maps any input value to a probability between 0 and 1, making it ideal for binary classification. Values above 0.5 typically indicate class 1 (Disease), while values below 0.5 indicate class 0 (No Disease).")
        
        st.subheader("Classification Report")
        report = classification_report(y_test, result['y_pred'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0), width="stretch")

# Page 5: Prediction
elif page == "🔮 Prediction":
    st.header("🔮 Make Predictions")
    
    if 'models' not in st.session_state:
        st.warning("⚠ Please train models first in the 'Model Training' page.")
    else:
        models = st.session_state['models']
        scaler = st.session_state['scaler']
        
        st.subheader("Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            resting_bp = st.number_input("Resting BP", 80, 200, 120)
            cholesterol = st.number_input("Cholesterol", 100, 400, 200)
        
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
            exercise_angina = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0, 0.1)
        
        with col3:
            chest_pain_ata = st.selectbox("Chest Pain Type: ATA", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            chest_pain_nap = st.selectbox("Chest Pain Type: NAP", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            chest_pain_ta = st.selectbox("Chest Pain Type: TA", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            resting_ecg_normal = st.selectbox("Resting ECG: Normal", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            resting_ecg_st = st.selectbox("Resting ECG: ST", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col2:
            st_slope_flat = st.selectbox("ST Slope: Flat", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col3:
            st_slope_up = st.selectbox("ST Slope: Up", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        if st.button("🔮 Predict", type="primary"):
            input_data = np.array([[age, sex, resting_bp, cholesterol, fasting_bs, max_hr,
                                   exercise_angina, oldpeak, chest_pain_ata, chest_pain_nap,
                                   chest_pain_ta, resting_ecg_normal, resting_ecg_st,
                                   st_slope_flat, st_slope_up]])
            
            input_scaled = scaler.transform(input_data)
            
            st.subheader("Prediction Results")
            
            cols = st.columns(len(models))
            
            for idx, (name, model) in enumerate(models.items()):
                with cols[idx]:
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0]
                    
                    if prediction == 1:
                        st.error(f"{name}")
                        st.error("⚠ Heart Disease Detected")
                        st.metric("Disease Probability", f"{probability[1]*100:.1f}%")
                    else:
                        st.success(f"{name}")
                        st.success("✅ No Heart Disease")
                        st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
            
            st.subheader("Prediction Confidence Comparison")
            
            prob_data = []
            for name, model in models.items():
                probability = model.predict_proba(input_scaled)[0]
                prob_data.append({
                    'Model': name,
                    'No Disease': probability[0] * 100,
                    'Disease': probability[1] * 100
                })
            
            prob_df = pd.DataFrame(prob_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='No Disease',
                x=prob_df['Model'],
                y=prob_df['No Disease'],
                marker_color='#2ecc71'
            ))
            fig.add_trace(go.Bar(
                name='Disease',
                x=prob_df['Model'],
                y=prob_df['Disease'],
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                barmode='group',
                height=400,
                title="Prediction Confidence by Model (%)",
                yaxis_title="Probability (%)",
                xaxis_title="Model"
            )
            st.plotly_chart(fig, width="stretch")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>💡 Heart Disease Prediction System | Built with Streamlit & Scikit-learn</p>
        <p>⚕ For educational purposes only - Not for clinical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)
