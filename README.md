# Airbnb Price Predictor App

This is a simple and interactive **Streamlit web app** that predicts the nightly price of an Airbnb listing in **Cape Town** using a pre-trained **Gradient Boosting Regressor** model built with `scikit-learn`.

---

## Features

- Predicts Airbnb listing price based on:
  - `neighbourhood`
  - `latitude`, `longitude`
  - `minimum_nights`
  - `number_of_reviews`, `reviews_per_month`
  - `calculated_host_listings_count`, `availability_365`
  - `number_of_reviews_ltm`, `last_review_missing`
  - Encoded room types: `Entire home/apt`, `Hotel room`, `Private room`, `Shared room`
- Simple and responsive UI built with **Streamlit**
- Real-time predictions using a pre-trained regression model

---

## Model Information

- Model: `GradientBoostingRegressor` (from `scikit-learn`)
- Trained on Airbnb listing data for **Cape Town**
- Saved using `joblib` and loaded at runtime in the app

---

## Project Structure

airbnb-price-predictor/
├── Airbnb-app.py # Streamlit app script
├── gbr_model.pkl # Trained model file
├── requirements.txt # Python dependencies
└── README.md # Project overview


---

## How to Run Locally

1. **Clone the repository:**

```
git clone https://github.com/your-username/airbnb-price-predictor.git
cd airbnb-price-predictor
(Optional) Create and activate a virtual environment:


python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

Install required packages:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run Airbnb-app.py

Live App
Check out the live version here:
https://your-deployment-link.streamlit.app
(Replace with your actual Streamlit Cloud URL)

Built With
Streamlit
scikit-learn
pandas
joblib

License
This project is licensed under the MIT License. Feel free to use or modify it.