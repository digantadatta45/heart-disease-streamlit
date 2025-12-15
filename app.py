import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
                         "🧹 Data Cleaning",
                         "🔍 EDA", 
                         "🤖 Model Training", 
                         "📈 Model Evaluation",
                         "🔮 Prediction"])

# Load data function
@st.cache_data
def load_raw_data():
    """Load the raw heart.csv dataset from GitHub"""
    try:
        # Replace with your GitHub raw URL
        url = "https://raw.githubusercontent.com/digantadatta45/heart-disease-streamlit/main/heart.csv"
        df = pd.read_csv(url)
        return df
    except:
        st.error("Error loading data from GitHub. Please check the URL.")
        return None

@st.cache_data
def clean_data(df):
    """Clean the raw dataset"""
    df_clean = df.copy()
    
    # Fix invalid values
    df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, df_clean['Cholesterol'].median())
    df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, df_clean['RestingBP'].median())
    df_clean.loc[df_clean['Oldpeak'] < 0, 'Oldpeak'] = 0
    
    # Handle outliers using IQR method (capping)
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    for col in num_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.where(df_clean[col] > upper, upper,
                                 np.where(df_clean[col] < lower, lower, df_clean[col]))
    
    # Encode categorical variables
    # Binary encoding
    df_clean['Sex'] = df_clean['Sex'].map({'M': 1, 'F': 0})
    df_clean['ExerciseAngina'] = df_clean['ExerciseAngina'].map({'Y': 1, 'N': 0})
    
    # One-hot encoding
    df_clean = pd.get_dummies(df_clean, 
                              columns=['ChestPainType', 'RestingECG', 'ST_Slope'],
                              drop_first=True, 
                              dtype=int)
    
    return df_clean

# Load data
df_raw = load_raw_data()

if df_raw is None:
    st.stop()

df_clean = clean_data(df_raw)

# Page 1: Data Overview
if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df_raw.shape[0])
    with col2:
        st.metric("Total Features", df_raw.shape[1] - 1)
    with col3:
        st.metric("Heart Disease Cases", df_raw['HeartDisease'].sum())
    with col4:
        st.metric("Healthy Cases", len(df_raw) - df_raw['HeartDisease'].sum())
    
    st.subheader("Raw Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df_raw.describe(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame(df_raw.dtypes, columns=['Type']), use_container_width=True)
    with col2:
        st.subheader("Missing Values")
        missing_df = pd.DataFrame(df_raw.isnull().sum(), columns=['Missing'])
        st.dataframe(missing_df, use_container_width=True)
        if missing_df['Missing'].sum() == 0:
            st.success("✅ No missing values found!")

# Page 2: Data Cleaning
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning Process")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Invalid Values", "Outliers", "Encoding", "Final Dataset"])
    
    with tab1:
        st.subheader("Checking Invalid Values")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            zero_chol = (df_raw['Cholesterol'] == 0).sum()
            st.metric("Cholesterol = 0", zero_chol, 
                     delta=f"-{zero_chol} fixed" if zero_chol > 0 else "None")
        with col2:
            zero_bp = (df_raw['RestingBP'] == 0).sum()
            st.metric("RestingBP = 0", zero_bp,
                     delta=f"-{zero_bp} fixed" if zero_bp > 0 else "None")
        with col3:
            neg_oldpeak = (df_raw['Oldpeak'] < 0).sum()
            st.metric("Oldpeak < 0", neg_oldpeak,
                     delta=f"-{neg_oldpeak} fixed" if neg_oldpeak > 0 else "None")
        
        st.info("🔧 *Fix Applied:* Invalid values replaced with median (Cholesterol, RestingBP) or converted to 0 (negative Oldpeak)")
        
        # Show before/after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("*Before Cleaning:*")
            st.dataframe(df_raw[['Cholesterol', 'RestingBP', 'Oldpeak']].describe())
        with col2:
            st.write("*After Cleaning:*")
            st.dataframe(df_clean[['Cholesterol', 'RestingBP', 'Oldpeak']].describe())
    
    with tab2:
        st.subheader("Outlier Detection & Treatment")
        
        num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        
        # Count outliers
        outlier_counts = {}
        for col in num_cols:
            Q1 = df_raw[col].quantile(0.25)
            Q3 = df_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df_raw[(df_raw[col] < lower) | (df_raw[col] > upper)]
            outlier_counts[col] = len(outliers)
        
        # Display outlier counts
        st.write("*Outliers Detected (IQR Method):*")
        outlier_df = pd.DataFrame(list(outlier_counts.items()), 
                                  columns=['Feature', 'Outlier Count'])
        st.dataframe(outlier_df, use_container_width=True)
        
        st.info("🔧 *Treatment Applied:* Capping method - outliers replaced with upper/lower bounds")
        
        # Visualize outliers
        selected_col = st.selectbox("Select feature to visualize", num_cols)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Before Cleaning", "After Cleaning"))
        
        fig.add_trace(go.Box(y=df_raw[selected_col], name="Before", 
                            marker_color='#e74c3c'), row=1, col=1)
        fig.add_trace(go.Box(y=df_clean[selected_col], name="After",
                            marker_color='#2ecc71'), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Categorical Encoding")
        
        st.write("*Binary Encoding:*")
        st.code("""
Sex: M → 1, F → 0
ExerciseAngina: Y → 1, N → 0
        """)
        
        st.write("*One-Hot Encoding:*")
        st.code("""
ChestPainType: ATA, NAP, TA (drop first to avoid dummy trap)
RestingECG: Normal, ST (drop first)
ST_Slope: Flat, Up (drop first)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("*Before Encoding:*")
            st.write(f"Columns: {df_raw.shape[1]}")
            st.dataframe(df_raw[['Sex', 'ChestPainType', 'ExerciseAngina']].head())
        with col2:
            st.write("*After Encoding:*")
            st.write(f"Columns: {df_clean.shape[1]}")
            encoded_cols = [col for col in df_clean.columns if any(x in col for x in ['ChestPainType', 'RestingECG', 'ST_Slope'])]
            display_cols = ['Sex', 'ExerciseAngina'] + encoded_cols[:3]
            st.dataframe(df_clean[display_cols].head())
    
    with tab4:
        st.subheader("Final Cleaned Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Shape", f"{df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        with col2:
            st.metric("Cleaned Shape", f"{df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
        
        st.dataframe(df_clean.head(20), use_container_width=True)
        
        st.subheader("Cleaned Dataset Statistics")
        st.dataframe(df_clean.describe(), use_container_width=True)
        
        if st.button("📥 Download Cleaned Dataset"):
            csv = df_clean.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_heart.csv",
                mime="text/csv"
            )

# Page 3: EDA
elif page == "🔍 EDA":
    st.header("🔍 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Target Distribution", "Feature Distributions", "Correlations", "Feature Importance"])
    
    with tab1:
        st.subheader("Heart Disease Distribution")
        
        fig = make_subplots(rows=1, cols=2, 
                           specs=[[{"type": "pie"}, {"type": "bar"}]],
                           subplot_titles=("Distribution", "Count"))
        
        disease_counts = df_clean['HeartDisease'].value_counts()
        
        fig.add_trace(go.Pie(labels=['No Disease', 'Disease'], 
                            values=disease_counts.values,
                            marker=dict(colors=['#2ecc71', '#e74c3c']),
                            hole=0.4), row=1, col=1)
        
        fig.add_trace(go.Bar(x=['No Disease', 'Disease'], 
                            y=disease_counts.values,
                            marker=dict(color=['#2ecc71', '#e74c3c'])), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Disease Cases", disease_counts[1], 
                     delta=f"{disease_counts[1]/len(df_clean)*100:.1f}%")
        with col2:
            st.metric("Healthy Cases", disease_counts[0],
                     delta=f"{disease_counts[0]/len(df_clean)*100:.1f}%")
    
    with tab2:
        st.subheader("Feature Distributions")
        
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=(f"{selected_feature} Distribution", 
                                         f"{selected_feature} by Heart Disease"))
        
        fig.add_trace(go.Histogram(x=df_clean[selected_feature], 
                                  name=selected_feature,
                                  marker_color='#3498db',
                                  nbinsx=30), row=1, col=1)
        
        fig.add_trace(go.Box(y=df_clean[df_clean['HeartDisease']==0][selected_feature], 
                            name='No Disease',
                            marker_color='#2ecc71'), row=1, col=2)
        fig.add_trace(go.Box(y=df_clean[df_clean['HeartDisease']==1][selected_feature], 
                            name='Disease',
                            marker_color='#e74c3c'), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("*No Disease:*")
            st.write(df_clean[df_clean['HeartDisease']==0][selected_feature].describe())
        with col2:
            st.write("*Disease:*")
            st.write(df_clean[df_clean['HeartDisease']==1][selected_feature].describe())
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        correlation = df_clean.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(height=700, title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
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
        fig.update_layout(height=500, title="Feature Correlation with Heart Disease",
                         xaxis_title="Correlation Coefficient")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Importance Analysis")
        
        with st.spinner("Calculating feature importance..."):
            X = df_clean.drop('HeartDisease', axis=1)
            y = df_clean['HeartDisease']
            
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
        fig.update_layout(height=600, title="Feature Importance (Random Forest)",
                         xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("*Top 5 Most Important Features:*")
        st.dataframe(feature_importance.head(), use_container_width=True)

# Page 4: Model Training
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
    
    if st.sidebar.button("🚀 Train Models", type="primary"):
        if not selected_models:
            st.warning("⚠ Please select at least one model to train.")
        else:
            with st.spinner("Training models... This may take a moment."):
                X = df_clean.drop('HeartDisease', axis=1)
                y = df_clean['HeartDisease']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                models = {}
                results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                model_list = []
                if "Logistic Regression" in selected_models:
                    model_list.append(("Logistic Regression", LogisticRegression(max_iter=1000, random_state=random_state)))
                if "Random Forest" in selected_models:
                    model_list.append(("Random Forest", RandomForestClassifier(n_estimators=100, random_state=random_state)))
                if "Gradient Boosting" in selected_models:
                    model_list.append(("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=random_state)))
                if "SVM" in selected_models:
                    model_list.append(("SVM", SVC(probability=True, random_state=random_state)))
                
                for idx, (name, model) in enumerate(model_list):
                    status_text.text(f"Training {name}...")
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    models[name] = model
                    results[name] = {
                        'model': model,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba)
                    }
                    
                    progress_bar.progress((idx + 1) / len(model_list))
                
                status_text.empty()
                progress_bar.empty()
                
                st.session_state['models'] = models
                st.session_state['results'] = results
                st.session_state['X_test'] = X_test_scaled
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = X.columns.tolist()
                
                st.success("✅ Models trained successfully!")
                
                st.subheader("Model Performance Comparison")
                
                metrics_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [r['accuracy'] for r in results.values()],
                    'Precision': [r['precision'] for r in results.values()],
                    'Recall': [r['recall'] for r in results.values()],
                    'F1 Score': [r['f1'] for r in results.values()],
                    'ROC AUC': [r['roc_auc'] for r in results.values()]
                })
                
                st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']).format("{:.4f}", subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']), 
                            use_container_width=True)
                
                # Bar chart comparison
                fig = go.Figure()
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
                
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=list(results.keys()),
                        y=[results[m][metric.lower().replace(' ', '_')] for m in results.keys()]
                    ))
                
                fig.update_layout(
                    barmode='group',
                    height=400,
                    title="Model Performance Metrics Comparison",
                    yaxis_title="Score",
                    xaxis_title="Model"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart
                fig = go.Figure()
                
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
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Configure your models in the sidebar and click 'Train Models' to begin.")

# Page 5: Model Evaluation
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
            st.metric("Accuracy", f"{result['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{result['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{result['recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{result['f1']:.4f}")
        with col5:
            st.metric("ROC AUC", f"{result['roc_auc']:.4f}")
        
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
            st.plotly_chart(fig, use_container_width=True)
            
            tn, fp, fn, tp = cm.ravel()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("True Negatives", tn)
                st.metric("False Negatives", fn, delta=f"-{fn}", delta_color="inverse")
            with col_b:
                st.metric("False Positives", fp, delta=f"-{fp}", delta_color="inverse")
                st.metric("True Positives", tp)
        
        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_test, result['y_pred_proba'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{selected_model} (AUC = {result["roc_auc"]:.4f})',
                line=dict(color='#e74c3c', width=3),
                fill='tonexty'
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
                yaxis_title="True Positive Rate (Recall)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📊 *AUC Score: {result['roc_auc']:.4f}* - The area under the ROC curve measures the model's ability to distinguish between classes. Higher is better (max = 1.0).")
        
        st.subheader("Sigmoid Function Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            X_range = np.linspace(-10, 10, 300)
            sigmoid = 1 / (1 + np.exp(-X_range))
            
            fig = go.Figure()
            
            # Sigmoid curve
            fig.add_trace(go.Scatter(
                x=X_range,
                y=sigmoid,
                mode='lines',
                name='Sigmoid Function',
                line=dict(color='#3498db', width=4)
            ))
            
            # Decision boundary
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Decision Boundary (0.5)",
                         annotation_position="right")
            
            # Zero line
            fig.add_vline(x=0, line_dash="dash", line_color="green",
                         annotation_text="x = 0",
                         annotation_position="top")
            
            # Key points
            fig.add_trace(go.Scatter(
                x=[-5, 0, 5],
                y=[1/(1+np.exp(5)), 0.5, 1/(1+np.exp(-5))],
                mode='markers',
                name='Key Points',
                marker=dict(size=12, color='red'),
                text=['Low Probability', 'Threshold (0.5)', 'High Probability'],
                textposition='top center'
            ))
            
            # Shaded regions
            fig.add_shape(type="rect",
                x0=-10, y0=0, x1=0, y1=0.5,
                fillcolor="lightblue", opacity=0.2,
                layer="below", line_width=0)
            
            fig.add_shape(type="rect",

                x0=0, y0=0.5, x1=10)


