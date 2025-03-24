import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import os
import random
import gdown
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Google Drive File IDs
DATASET_FILE_ID = "1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_FILE_ID = "1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"

# Download & cache dataset
@st.cache_data
def load_dataset():
    dataset_url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
    output_zip = "customer_data.zip"
    dataset_folder = "dataset"

    if not os.path.exists(dataset_folder):
        with st.spinner("Downloading dataset..."):
            gdown.download(dataset_url, output_zip, quiet=True)
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
    
    csv_filename = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")][0]
    df = pd.read_csv(f"{dataset_folder}/{csv_filename}")
    return df

# Download & cache model
@st.cache_resource
def load_model():
    model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    output_model = "customer_churn_model.h5"

    if not os.path.exists(output_model):
        with st.spinner("Downloading pre-trained model..."):
            gdown.download(model_url, output_model, quiet=True)
    
    return tf.keras.models.load_model(output_model)

# Random sample function with seed
def get_random_sample(df, n=10000, seed=42):
    return df.sample(n=n, random_state=seed)

# Preprocess categorical variables
@st.cache_data
def preprocess_data(df):
    df = df.copy()
    categorical_columns = ['gender', 'marital_status', 'education_level', 'occupation', 'preferred_store', 'payment_method', 
                           'store_city', 'store_state', 'season', 'product_color', 'product_material', 'promotion_channel', 
                           'promotion_type', 'promotion_target_audience']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# Sidebar Hyperparameters
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=30, value=3)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05], index=0)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"], index=0)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0)
dense_layers = st.sidebar.slider("Dense Layers", min_value=2, max_value=4, value=2)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", min_value=32, max_value=256, step=32, value=128)
sample_size = st.sidebar.slider("Sample Size", min_value=5000, max_value=50000, step=5000, value=10000)

# Load dataset
df = load_dataset()

# Load pre-trained model (optional use later)
pretrained_model = load_model()

# Train Model Section
if st.button("🚀 Train Model"):
    st.subheader(f"📥 Extracting {sample_size} Random Samples")
    df_sample = get_random_sample(df, n=sample_size)
    df_sample = preprocess_data(df_sample)
    
    X = df_sample.drop(columns=['churned'])
    y = df_sample['churned']
    
    # Define Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    
    optimizer_dict = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    }
    model.compile(optimizer=optimizer_dict[optimizer_choice], loss="binary_crossentropy", metrics=["accuracy"])
    
    with st.spinner("Training in progress..."):
        history = model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    
    st.success("✅ Training Completed!")
    
    # Plot Training Progress
    st.subheader("📊 Training Metrics")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='Train Acc')
    ax[1].plot(history.history['val_accuracy'], label='Val Acc')
    ax[1].set_title("Accuracy")
    ax[1].legend()
    st.pyplot(fig)
    
    # Evaluation
    y_pred = model.predict(X) > 0.5
    st.subheader("📋 Classification Report")
    st.text(classification_report(y, y_pred))
    
    # Save model
    model.save("updated_customer_churn_model.h5")
    with open("updated_customer_churn_model.h5", "rb") as f:
        st.download_button("📥 Download Updated Model", f, file_name="updated_customer_churn_model.h5")

# Visualizations
st.subheader("📈 Data Insights")

# Churn Distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["churned"], palette="coolwarm", ax=ax)
ax.set_title("Customer Churn Distribution")
st.pyplot(fig)

# Age vs Churn
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df, x="age", hue="churned", kde=False, element="step", palette="coolwarm", ax=ax)
ax.set_title("Churn Distribution by Age")
st.pyplot(fig)

# Income vs Churn
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x="churned", y="income_bracket", data=df, palette="coolwarm", ax=ax)
ax.set_title("Income Bracket vs Churn")
st.pyplot(fig)

st.success("✅ Dashboard Ready!")

