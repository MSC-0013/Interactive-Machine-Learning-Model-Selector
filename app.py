import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

# Set the page title and icon
st.set_page_config(page_title="ML Model Selector", page_icon=":chart_with_upwards_trend:", layout="wide")

# Title and subtitle
st.title("📊 ML Model Selector")
st.markdown("Designed by MSC")  # Footer with designer's name
st.markdown("""
Select a model and upload your dataset to explore various machine learning techniques.
""")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV or SQL file", type=["csv", "sql"])

if uploaded_file is not None:
    # Load data based on file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".sql"):
        # Connect to SQL database and load data
        engine = create_engine('sqlite:///my_database.db')  # Replace with your SQL file or connection string
        df = pd.read_sql(uploaded_file, engine)
    
    # Display data preview and description
    st.subheader("Data Preview")
    st.write(df.head())
    st.subheader("Data Description")
    st.write(df.describe())

    # Data Cleaning Options
    st.subheader("Data Cleaning")
    if st.checkbox("Remove rows with missing values"):
        df = df.dropna()
        st.write("Removed rows with missing values.")
    if st.checkbox("Remove duplicate rows"):
        df = df.drop_duplicates()
        st.write("Removed duplicate rows.")
    
    # Convert categorical columns to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Data Visualization
    st.subheader("Data Visualization")
    
    # Filter numeric columns for correlation heatmap
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Feature Correlation Heatmap
        st.write("### Feature Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig_corr = plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig_corr)
    
        # Histograms
        st.write("### Histograms")
        selected_hist_cols = st.multiselect("Select columns for histograms", numeric_cols, default=numeric_cols)
        if selected_hist_cols:
            for col in selected_hist_cols:
                fig_hist = plt.figure(figsize=(10, 4))
                plt.hist(df[col], bins=30, edgecolor='black')
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                st.pyplot(fig_hist)

        # Scatter Plots
        st.write("### Scatter Plots")
        if len(numeric_cols) > 1:
            x_col = st.selectbox("Select X-axis column for scatter plot", numeric_cols)
            y_col = st.selectbox("Select Y-axis column for scatter plot", numeric_cols)
            if x_col and y_col:
                fig_scatter = plt.figure(figsize=(10, 6))
                plt.scatter(df[x_col], df[y_col], alpha=0.5)
                plt.title(f'Scatter Plot: {x_col} vs {y_col}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                st.pyplot(fig_scatter)

    # Model Selection
    model = st.selectbox(
        "Select a model",
        ["Linear Regression", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Clustering"]
    )
    
    if model == "Linear Regression":
        st.subheader("Linear Regression")
        st.write("Select target and features")
        target = st.selectbox("Select target variable", numeric_cols)
        features = st.multiselect("Select features", numeric_cols)
        
        if target and features:
            # Prepare data for training
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")
            
            # Visualization
            st.write("### Predictions vs Actual")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, color='blue', label='Predicted values')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal fit')
            ax.set_title('Actual vs Predicted Values')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()
            st.pyplot(fig)

    elif model == "Logistic Regression":
        st.subheader("Logistic Regression")
        st.write("Select target and features")
        target = st.selectbox("Select target variable", numeric_cols)
        features = st.multiselect("Select features", numeric_cols)
        
        if target and features:
            # Prepare data for training
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("### Classification Report")
            st.write(classification_report(y_test, predictions))
            
            # Visualization
            st.write("### Confusion Matrix")
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(pd.DataFrame(confusion_matrix(y_test, predictions)), annot=True, fmt='d')
            st.pyplot(fig)

    elif model == "Decision Tree":
        st.subheader("Decision Tree")
        st.write("Select target and features")
        target = st.selectbox("Select target variable", numeric_cols)
        features = st.multiselect("Select features", numeric_cols)
        
        if target and features:
            # Prepare data for training
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy:.2f}")
            
            # Visualization
            st.write("### Decision Tree Visualization")
            fig = plt.figure(figsize=(12, 8))
            plot_tree(clf, feature_names=features, class_names=df[target].unique(), filled=True)
            st.pyplot(fig)

    elif model == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors")
        st.write("Select target and features")
        target = st.selectbox("Select target variable", numeric_cols)
        features = st.multiselect("Select features", numeric_cols)
        
        if target and features:
            # Prepare data for training
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = KNeighborsClassifier()
            n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5)
            clf.n_neighbors = n_neighbors
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy:.2f}")
            
            # Visualization
            st.write("### KNN Visualization")
            if len(features) >= 2:
                fig = plt.figure(figsize=(10, 6))
                scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=predictions, cmap='viridis')
                plt.colorbar(scatter)
                plt.title('KNN Prediction Visualization')
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                st.pyplot(fig)
            else:
                st.write("Select at least 2 features for visualization.")

    elif model == "Naive Bayes":
        st.subheader("Naive Bayes")
        st.write("Select target and features")
        target = st.selectbox("Select target variable", numeric_cols)
        features = st.multiselect("Select features", numeric_cols)
        
        if target and features:
            # Prepare data for training
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Accuracy: {accuracy:.2f}")
            
            # Visualization
            st.write("### Naive Bayes Visualization")
            if len(features) >= 2:
                fig = plt.figure(figsize=(10, 6))
                scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=predictions, cmap='viridis')
                plt.colorbar(scatter)
                plt.title('Naive Bayes Prediction Visualization')
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                st.pyplot(fig)
            else:
                st.write("Select at least 2 features for visualization.")

    elif model == "Clustering":
        st.subheader("Clustering")
        st.write("Select number of clusters")
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        st.write("Select features for clustering")
        features = st.multiselect("Select features", numeric_cols)
        
        if features:
            X = df[features]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            df['Cluster'] = clusters
            
            # Visualization
            if len(features) >= 2:
                st.write("### Clustering Visualization")
                fig = px.scatter(df, x=features[0], y=features[1], color='Cluster', title='Clustering Visualization')
                st.plotly_chart(fig)
            else:
                st.write("Select at least 2 features for visualization.")


