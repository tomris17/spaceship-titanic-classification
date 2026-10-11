import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Spaceship Titanic Dashboard",
    page_icon="🚀",
    layout="wide"
)

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('test (1).csv')
    target = df['Transported']
    
    columns_to_drop = ['Age', 'VIP', 'Cabin_side_P', 'Cabin_side_S', 
                       'Cabin_side_blank', 'PassengerId_group', 
                       'PassengerId_num', 'Cabin_num']
    
    df_model = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    if 'TotalSpent' in df_model.columns:
        df_model['TotalSpent'] = df_model['TotalSpent'].fillna(0).astype(np.int64)
        
    X = df_model.drop(columns=['Transported'], errors='ignore')
    y = target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = load_and_train_model()

st.title("🚀 Spaceship Titanic - Sınıflandırma ve Performans Paneli")
st.write("Bu panel, Multinomial Naive Bayes modeli ile yolcu taşıma tahminlerini ve hata matrislerini analiz eder.")

# Metrikler
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

col1, col2 = st.columns(2)
col1.metric("Eğitim Verisi Doğruluğu", f"%{train_acc * 100:.2f}")
col2.metric("Test Verisi Doğruluğu", f"%{test_acc * 100:.2f}")

st.markdown("---")

def plot_confusion_matrix(y_true, y_pred, cf_title):
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f'{value:0.0f}' for value in cf_matrix.flatten()]
    group_percentages = [f'{value:.2%}' for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', ax=ax)
    ax.set_title(cf_title)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek Değer")
    st.pyplot(fig)

# Sekmeler veya Yan Yana Matrisler
st.subheader("📊 Model Karmaşıklık Matrisi (Confusion Matrix Analizi)")
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

tab1, tab2 = st.tabs(["Eğitim Matrisi", "Test Matrisi"])

with tab1:
    plot_confusion_matrix(y_train, pred_train, 'Eğitim Verisi Karmaşıklık Matrisi')

with tab2:
    plot_confusion_matrix(y_test, pred_test, 'Test Verisi Karmaşıklık Matrisi')