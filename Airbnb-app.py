#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np


# In[ ]:


st.markdown("""
### *Welcome to the Airbnb Property Price Predictor App*

This app helps you **predict the price of a property in Cape Town** based on several input features.
""")


# In[ ]:


model = joblib.load("gbr_model.pkl")
feature_means = joblib.load("feature_means.pkl")
feature_names = list(feature_means.keys())


# In[ ]:


st.markdown("""
**The features required for this model to make a prediction are:**

- `Ward Number(neighborhood)`, `Latitude`, `Longitude`
- `Minimum Number of Nights`, `Number of Reviews`, `Number of Reviews Per Month`
- `Number of Host Listings`, `Availability In A Year (Number of Days)`
- `number_of_reviews_ltm`, `Last Review Missing (Y/N)`
- `Room Type: Entire home/apt (Y/N)`, `Room Type: Hotel Room (Y/N)`
- `Room Type: Private Room (Y/N)`, `Room Type: Shared Room (Y/N)`

*Please enter numeric values for each.*
N.B: for (Y/N), use: 1-Yes; 0-No
""")


# In[ ]:


cape_town_coords = {
    "lat": -33.9249,
    "lon": 18.4241}

df = pd.DataFrame({
    'lat': [cape_town_coords["lat"]],
    'lon': [cape_town_coords["lon"]]})

st.map(df, zoom=10) 


# In[ ]:


user_input = {}
for feat in feature_names:
    val = st.text_input(f"{feat} (leave blank to use default: {round(feature_means[feat], 2)})")
    if val.strip() == "":
        user_input[feat] = feature_means[feat]
    else:
        try:
            user_input[feat] = float(val)
        except ValueError:
            st.error(f"Invalid input for {feat}. Must be numeric.")
            st.stop()

if st.button("Predict"):
    input_array = pd.DataFrame([user_input])[feature_names]
    prediction = model.predict(input_array)[0]
    st.success(f"The predicted price is: R{round(prediction, 2)}")


# In[ ]:


# st.title("GBR Predictor")

# f1 = st.number_input("Feature 1")
# f2 = st.number_input("Feature 2")
# f3 = st.number_input("Feature 3")

# if st.button("Predict"):
#     X = np.array([[f1, f2, f3]])
#     y_pred = model.predict(X)
#     st.write(f"Prediction: {y_pred[0]}")


# In[ ]:


# !streamlit run C:\Users\Lenovo\Desktop\Projects\Datasets\venv\Lib\site-packages\ipykernel_launcher.py [ARGUMENTS]


# In[ ]:




