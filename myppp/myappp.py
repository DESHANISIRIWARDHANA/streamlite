# Import statements
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"D:\myppp\diabetes.csv")  # Use raw string to avoid path issues

# Headings
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X and Y data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function for user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Patient data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# Visualizations
st.title('Visualised Patient Report')

# Color function
color = 'blue' if user_result[0] == 0 else 'red'

# Visualization plots
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
titles = ['Pregnancy count', 'Glucose Value', 'Blood Pressure Value', 'Skin Thickness Value', 'Insulin Value', 'BMI Value', 'DPF Value']

for feature, title in zip(features, titles):
    st.header(f'{title} Graph (Others vs Yours)')
    fig = plt.figure()
    ax1 = sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome', palette='Greens')
    ax2 = sns.scatterplot(x=user_data['Age'], y=user_data[feature].values, s=150, color=color)
    plt.xticks(np.arange(10, 100, 5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig)

# Output
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.header(output)

# Accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader('Model Accuracy:')
st.write(f'Accuracy of the model: {accuracy:.2f}%')
