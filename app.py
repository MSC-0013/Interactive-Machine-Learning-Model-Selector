import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Machine Learning Model Explorer", layout="wide")
st.title("Machine Learning Model Explorer")

st.sidebar.header("Upload your CSV dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Here is a preview of your dataset:")
    st.write(df.head())
    
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns", [col for col in df.columns if col != target_column])

    if target_column and feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
            task_type = "regression"
        else:
            task_type = "classification"
        
        model_choice = st.sidebar.selectbox("Choose Model", 
                                            ["Logistic Regression", "Random Forest", 
                                             "Support Vector Machine", "K-Means Clustering", 
                                             "Linear Regression", "Decision Tree"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = None
        if model_choice == "Logistic Regression" and task_type == "classification":
            model = LogisticRegression()
        elif model_choice == "Random Forest":
            if task_type == "classification":
                model = RandomForestClassifier()
            else:
                model = RandomForestRegressor()
        elif model_choice == "Support Vector Machine" and task_type == "classification":
            model = SVC()
        elif model_choice == "K-Means Clustering":
            model = KMeans(n_clusters=st.sidebar.slider("Number of Clusters", 2, 10, 2))
        elif model_choice == "Linear Regression" and task_type == "regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            if task_type == "classification":
                model = DecisionTreeClassifier()
            else:
                model = DecisionTreeRegressor()
        
        if model:
            model.fit(X_train, y_train)
            
            if model_choice == "K-Means Clustering":
                y_pred = model.predict(X_test)
                st.write("Cluster Centers:", model.cluster_centers_)
            elif task_type == "classification":
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write("Accuracy:", accuracy)
                
                conf_matrix = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt="d", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)
            elif task_type == "regression":
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                st.write("Mean Squared Error:", mse)
            
            if model_choice in ["Random Forest", "Decision Tree"] and task_type == "classification":
                feature_importances = model.feature_importances_
                feature_df = pd.DataFrame({"Feature": feature_columns, "Importance": feature_importances})
                feature_df = feature_df.sort_values(by="Importance", ascending=False)
                st.bar_chart(feature_df.set_index("Feature"))
            
            st.write("Feature Distributions:")
            for feature in feature_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[feature], kde=True, ax=ax)
                ax.set_title(f"Distribution of {feature}")
                st.pyplot(fig)
else:
    st.write("Upload a CSV file to get started.")
