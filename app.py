import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Insurance Claim Analysis: Demographic and Health-EDA",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Insurance Claim Analysis: Demographic and Health-EDA")
st.write(" "
         )
st.subheader("Infos about the dataset:")
st.write(
    """
    Our dataset provides in-depth insights into the demographic patterns and health characteristics of insurance claimants. It encompasses key information such as age, gender, BMI (Body Mass Index), blood pressure levels, diabetic status, number of children, smoking status, and geographical region of patients. These data points allow us to identify critical factors influencing who is likely to file insurance claims.
Analyzing these variables enables us to uncover patterns across geographical areas and demographic groups. This knowledge is not only crucial for tailoring our services but can also guide targeted support for the most needy and vulnerable groups, and make public policy more effective.
       """
)
a1, a2 = st.columns(2)
with a1:
    df = pd.read_csv("df_clean.csv")
    st.subheader("Data Preview:")
    st.write(df.head())
with a2:
    st.subheader("Data Description:")
    st.write(df.describe())
a11, a22, a33 = st.columns(3)
with a11:
    st.subheader("Data types:")
    st.write(df.dtypes.astype(str))
with a22:
    st.subheader("Data shape:")
    st.write(df.shape)
with a33:
    st.subheader("Missing values:")
    st.write(df.isnull().sum())

a111, a222 = st.columns(2)
with a111:
    st.subheader("Data Correlation:")
    st.write(df.corr())
with a222:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="viridis")
    plt.title("The correlation between the features", fontsize=15)
    st.pyplot()

count_data = df[["gender","diabetic", "children", "smoker"]]
col1, col2 = st.columns(2)
with col1:
    for col in count_data.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, data=count_data, hue=df["region"])
        plt.title(f"The count of {col} by region", fontsize=15)
        st.pyplot()
with col2:
    for col in df.columns:
        if df[col].dtype != "object" and col != "index":
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f"The distribution of {col}", fontsize=15)
            st.pyplot()

col4, col5 = st.columns(2)
with col4:
    for col in df.columns:
        if df[col].dtype != "object" and col != "index":
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=col, data=df)
            plt.title(f"The distribution of {col}", fontsize=15)
            st.pyplot()
with col5:
    for col in count_data.columns:
        plt.figure(figsize=(10, 10))
        plt.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct="%.1f%%")
        plt.title(f"The count of {col}", fontsize=15)
        st.pyplot()

st.subheader("Exploring Relationships Between Features by Region")
st.pyplot(sns.pairplot(df, hue="region"))

st.write("Thanks....!!!")
st.write("Done by [Ahmad](https://github.com/Ahmad1993Aj)")
