import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

@st.cache
def load_data():
    data = pd.read_csv('DATA SET.csv')
    data.columns = data.columns.str.strip() 
    return data

def preprocess_data(data):
    X = data.drop(columns=['day', 'month', 'year', 'Classes'])
    y = data['Classes'].str.strip()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X = X.apply(pd.to_numeric, errors='coerce')
    return X, y

@st.cache
def train_model(X_train, y_train):
    dt_classifier = DecisionTreeClassifier()

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def create_streamlit_ui():
    st.set_page_config(page_title="Predicción de Incendios Forestales", layout="wide")

    st.title("Predicción de Incendios Forestales")
    st.subheader("Análisis de Riesgo de Incendios con Machine Learning")

    data = load_data()
    st.write("Datos originales:")
    st.dataframe(data.head())

    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.header("Entrenamiento del modelo")
    if st.button("Entrenar modelo"):
        best_dt = train_model(X_train, y_train)
        y_pred = best_dt.predict(X_test)

        accuracyDT = accuracy_score(y_test, y_pred)
        precisionDT = precision_score(y_test, y_pred)
        recallDT = recall_score(y_test, y_pred)
        f1DT = f1_score(y_test, y_pred)

        st.write(f"Accuracy: {accuracyDT}")
        st.write(f"Precision: {precisionDT}")
        st.write(f"Recall: {recallDT}")
        st.write(f"F1 Score: {f1DT}")

        st.header("Predicción de incendios")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperatura (°C)", min_value=0.0, max_value=50.0, value=25.0)
            rh = st.slider("Humedad Relativa (%)", min_value=0, max_value=100, value=50)
            ws = st.slider("Velocidad del viento (km/h)", min_value=0.0, max_value=30.0, value=5.0)
            rain = st.slider("Lluvia (mm)", min_value=0.0, max_value=100.0, value=0.0)
        with col2:
            ffmc = st.slider("FFMC", min_value=0.0, max_value=100.0, value=85.0)
            dmc = st.slider("DMC", min_value=0.0, max_value=100.0, value=30.0)
            dc = st.slider("DC", min_value=0.0, max_value=500.0, value=150.0)
            isi = st.slider("ISI", min_value=0.0, max_value=100.0, value=10.0)
        
        bui = st.slider("BUI", min_value=0.0, max_value=100.0, value=60.0)
        fwi = st.slider("FWI", min_value=0.0, max_value=100.0, value=15.0)

        if st.button("Predecir"):
            input_data = pd.DataFrame({
                'Temperature': [temperature],
                'RH': [rh],
                'Ws': [ws],
                'Rain': [rain],
                'FFMC': [ffmc],
                'DMC': [dmc],
                'DC': [dc],
                'ISI': [isi],
                'BUI': [bui],
                'FWI': [fwi]
            })

            prediction = best_dt.predict(input_data)[0]
            st.subheader(f"Predicción de clase de incendio: {prediction}")

if __name__ == "__main__":
    create_streamlit_ui()
