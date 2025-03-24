import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile
import os
import random
import gdown
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Google Drive File IDs
DATASET_FILE_ID = "1X40NeGmYe0epMXrewSVlbQscLaX4u9qT"
MODEL_FILE_ID = "1o02g0r4xjlhWDUewEAlGb-kFQU9QcCqP"

# Function to download and extract dataset
@st.cache_data
def load_dataset():
    dataset_url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
    output_zip = "customer_data.zip"
    gdown.download(dataset_url, output_zip, quiet=True)

    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall("dataset")

    csv_filename = [f for f in os.listdir("dataset") if f.endswith(".csv")][0]
    # FIXED Encoding issue
    df = pd.read_csv(f"dataset/{csv_filename}", encoding='latin1')
    return df

# Function to download and load model
@st.cache_resource
def load_model():
    model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    output_model = "customer_churn_model.h5"
    gdown.download(model_url, output_model, quiet=True)
    return tf.keras.models.load_model(output_model)

# Random sample selector
def get_random_sample(df):
    return df.sample(n=50000, random_state=random.randint(1, 1000))

# Preprocessing function
def preprocess_data(df):
    categorical_columns = ['gender', 'marital_status', 'education_level', 'occupation', 'preferred_store', 
                           'payment_method', 'store_city', 'store_state', 'season', 'product_color', 
                           'product_material', 'promotion_channel', 'promotion_type', 'promotion_target_audience']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df

# Streamlit Page Config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# Sidebar Hyperparameters
st.sidebar.header("🔧 Model Hyperparameters")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=5)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05])
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.slider("Dense Layers", min_value=2, max_value=5, value=3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", min_value=32, max_value=512, step=32, value=128)

# Load dataset once
df = load_dataset()

# "Train Model" Button OUTSIDE sidebar
st.subheader("🚀 Model Training Section")
if st.button("Start Training Model"):
    st.info("Extracting 50,000 Random Data Points...")
    df_sample = get_random_sample(df)
    df_sample = preprocess_data(df_sample)

    X = df_sample.drop(columns=['churned'])
    y = df_sample['churned']

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation_function))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer_config = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    }
    optimizer = optimizer_config[optimizer_choice]
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Training
    history = model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

    # Plot
    st.subheader("📊 Training Progress")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss Over Epochs")
    ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title("Accuracy Over Epochs")
    ax[1].legend()
    st.pyplot(fig)

    # Save updated model
    model.save("updated_customer_churn_model.h5")
    with open("updated_customer_churn_model.h5", "rb") as f:
        st.download_button("📥 Download Updated Model", f, file_name="updated_customer_churn_model.h5")

# Data Insights Section
st.subheader("📈 Data Insights & Visualization")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["churned"], palette="coolwarm", ax=ax)
ax.set_title("Customer Churn Distribution")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df, x="age", hue="churned", kde=True, element="step", palette="coolwarm", ax=ax)
ax.set_title("Churn Distribution by Age")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="churned", y="income_bracket", data=df, palette="coolwarm", ax=ax)
ax.set_title("Income Bracket vs Churn")
st.pyplot(fig)

st.success("✅ Dashboard Ready!")
