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

- `neighbourhood`, `latitude`, `longitude`
- `minimum_nights`, `number_of_reviews`, `reviews_per_month`
- `calculated_host_listings_count`, `availability_365`
- `number_of_reviews_ltm`, `last_review_missing`
- `room_type_Entire home/apt`, `room_type_Hotel room`
- `room_type_Private room`, `room_type_Shared room`

*Please enter numeric values for each.*
""")


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




