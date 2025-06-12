#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import statsmodels.api as sm

# Load and prepare the data
df = pd.read_csv('data.csv')
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
df = df.join(pd.get_dummies(df.region, dtype=int)).drop('region', axis=1)

# Train the model
X = df[['age', 'bmi', 'smoker', 'southeast']]
y = df['charges']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Streamlit UI
st.title("💰 Insurance Charge Predictor")
st.write("Estimate your insurance charges based on your age, BMI, smoking status, and region.")

# User inputs
age = st.number_input("What is your age?", min_value=22, max_value=100, value=30)
bmi = st.number_input("What is your BMI?", min_value=10.0, max_value=65.0, value=25.0)
smoker_input = st.selectbox("Are you a smoker?", ['no', 'yes'])
region_input = st.selectbox("What region do you reside in?", ['northeast', 'northwest', 'southeast', 'southwest'])

# Convert user inputs to model features
smoker = 1 if smoker_input == 'yes' else 0
southeast = 1 if region_input == 'southeast' else 0

# Prepare input DataFrame
new_user_df = pd.DataFrame({
    'const': [1],
    'age': [age],
    'bmi': [bmi],
    'smoker': [smoker],
    'southeast': [southeast]
})

# Predict
predicted_charges = model.predict(new_user_df)[0]

# Output
st.subheader("💵 Estimated Annual Insurance Premium:")
st.success(f"${predicted_charges:,.2f}")


# In[ ]:




