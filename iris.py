import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset
def load_data():
    return pd.read_csv("Iris.csv")

iris = load_data()

# Displaying the first few rows of the dataset
st.title("Iris Dataset Exploration")
st.subheader("First Few Rows of the Dataset")
st.write(iris.head())

# Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(iris.describe())

# Checking for missing values
st.subheader("Missing Values Check")
st.write(iris.isnull().sum())

# Class distribution visualization
st.subheader("Class Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='Species', data=iris, ax=ax)
plt.title('Class Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
st.pyplot(fig)

# Pairplot
# Pairplot
st.subheader("Pairplot")
pairplot = sns.pairplot(iris, hue='Species')
st.pyplot(pairplot)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numerical_iris = iris.drop('Species', axis=1)
corr = numerical_iris.corr()
heatmap_fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(heatmap_fig)

# Boxplots
st.subheader("Boxplots")
plt.figure(figsize=(10, 6))
for i, feature in enumerate(iris.columns[:-1]):
    if i < 4:
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='Species', y=feature, data=iris)
        plt.title(f'{feature} Distribution by Species')
plt.tight_layout()
st.pyplot(plt)

# Model training and evaluation
st.subheader("Model Training and Evaluation")

# Splitting the data
X = iris.drop('Species', axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing models
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'K Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Training and evaluating models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write(f'**{name}**')
    st.write(f'Accuracy: {accuracy}')
    st.write(f'Precision: {precision}')
    st.write(f'Recall: {recall}')
    st.write(f'F1 Score: {f1}')
