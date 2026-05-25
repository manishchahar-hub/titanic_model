import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load trained model
model = pickle.load(open(r"C:\Users\Manish Chahar\Desktop\internship\TitanicProject\model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Interactive Dashboard", layout="wide")

st.title("Interactive Data Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    dataset = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    # Dataset info
    st.subheader("Dataset Shape")
    st.write(f"Rows: {dataset.shape[0]}")
    st.write(f"Columns: {dataset.shape[1]}")

    # Select columns
    numeric_columns = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()

    st.sidebar.header("Chart Options")

    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        ["Histogram", "Boxplot", "Scatter Plot", "Count Plot", "Correlation Heatmap"]
    )

    # Histogram
    if chart_type == "Histogram":
        col = st.sidebar.selectbox("Select Numeric Column", numeric_columns)

        fig, ax = plt.subplots()
        ax.hist(dataset[col], bins=20)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    # Boxplot
    elif chart_type == "Boxplot":
        col = st.sidebar.selectbox("Select Numeric Column", numeric_columns)

        fig, ax = plt.subplots()
        sns.boxplot(y=dataset[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    # Scatter Plot
    elif chart_type == "Scatter Plot":
        x_col = st.sidebar.selectbox("Select X Column", numeric_columns)
        y_col = st.sidebar.selectbox("Select Y Column", numeric_columns)

        fig, ax = plt.subplots()
        ax.scatter(dataset[x_col], dataset[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig)

    # Count Plot
    elif chart_type == "Count Plot":
        if len(categorical_columns) > 0:
            col = st.sidebar.selectbox("Select Categorical Column", categorical_columns)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=dataset[col], ax=ax)
            plt.xticks(rotation=45)
            ax.set_title(f"Count Plot of {col}")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns found.")

    # Correlation Heatmap
    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = dataset[numeric_columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

else:
    st.info("Please upload a CSV dataset to continue.")

# =============================
# Titanic Survival Prediction
# =============================

st.title("Titanic Survival Prediction")

st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)

embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encoding
sex = 1 if sex == 'male' else 0

embarked_C = 1 if embarked == 'C' else 0
embarked_Q = 1 if embarked == 'Q' else 0

# Feature array
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_C, embarked_Q]])

# Prediction button
if st.button("Predict Survival"):

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Passenger is likely to Survive ✅")
    else:
        st.error("Passenger is likely to Not Survive ❌")


