import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("student_data.csv")

# Train the model
X = df[['Hours_Studied', 'Previous_Score', 'Attendance']]
y = df['Exam_Score']
model = LinearRegression()
model.fit(X, y)

# Streamlit Interface
st.title("📘 Student Exam Score Predictor")
st.write("Enter student details to predict exam score:")

hours = st.slider("Hours Studied", 0.0, 15.0, 5.0)
previous = st.slider("Previous Score", 0.0, 100.0, 70.0)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0)

if st.button("Predict Score"):
    input_data = np.array([[hours, previous, attendance]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Exam Score: **{prediction:.2f}**")
