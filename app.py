import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@st.cache_data
def load_data():
    data = pd.read_csv('DATA SET.csv')
    data.columns = data.columns.str.strip()  
    return data

data = load_data()

if "train_model" not in st.session_state:
    st.session_state.train_model = False

X = data.drop(columns=['day', 'month', 'year', 'Classes'])
y = data['Classes'].str.strip()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = X.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=X_train.shape[1])
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

st.title("Predicción de Incendios Forestales con Árboles de Decisión")
st.write("""
La aplicación utiliza un clasificador de árboles de decisión para predecir el nivel de riesgo de incendios forestales a partir de diferentes 
características como la temperatura, humedad, velocidad del viento, cantidad de lluvia.
""")

def entrenar_y_mostrar_modelo(best_dt):
    y_pred = best_dt.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    y_score = best_dt.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    st.write(f"AUC: {roc_auc:.4f}")
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve - Decision Tree')
    ax.legend(loc="lower right")
    st.pyplot(fig)

st.subheader("Entrenamiento Inicial")

dt_classifier = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

best_dt = grid_search.best_estimator_
entrenar_y_mostrar_modelo(best_dt)

st.write("---")

st.subheader("Selecciona Nuevos Parámetros para Entrenar")

with st.expander("Selecciona los Parámetros de Entrenamiento", expanded=not st.session_state.train_model):
    temperatura = st.slider("Temperatura", min_value=0.0, max_value=50.0, value=25.0)
    humedad = st.slider("Humedad Relativa (%)", min_value=0, max_value=100, value=50)
    velocidad_viento = st.slider("Velocidad del Viento (km/h)", min_value=0, max_value=100, value=10)
    lluvia = st.slider("Lluvia (mm)", min_value=0.0, max_value=50.0, value=0.0)

    if st.button("Entrenar Modelo"):
        st.session_state.train_model = True 
        st.success("Entrenando el modelo con nuevos parámetros...")

if st.session_state.train_model:
    grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
    grid_search.fit(X_train_pca, y_train)
    best_dt = grid_search.best_estimator_

    st.subheader("Resultados del entrenamiento con nuevos parámetros")
    entrenar_y_mostrar_modelo(best_dt)
