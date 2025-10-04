import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ------------------------------
# Load model from Hugging Face Hub
# ------------------------------
model_path = hf_hub_download(
    repo_id="Vignesh-vigu/Tourism-Package-Prediction",  # replace with your actual repo_id
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧳 Tourism Wellness Package Prediction App")
st.write("""
This app predicts whether a customer is likely to purchase the new **Wellness Tourism Package**.
Please fill in the details below:
""")

# ------------------------------
# Input Form
# ------------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer"])
gender = st.radio("Gender", ["Male", "Female"])
num_person_visiting = st.slider("Number of Persons Visiting", 1, 5, 2)
num_followups = st.slider("Number of Followups", 0, 10, 2)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe"])
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
passport = st.radio("Passport", [0, 1])
pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.radio("Own Car", [0, 1])
num_children_visiting = st.slider("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

# ------------------------------
# Prepare Input Data
# ------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "✅ Likely to Purchase Package" if prediction == 1 else "❌ Not Likely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
