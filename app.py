import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64

# ---------------- Page Config ---------------- #
st.set_page_config(page_title="Breast Cancer Detector", layout="wide")

# ---------------- Background Styling ---------------- #
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.75);
            padding: 2rem;
            border-radius: 1rem;
        }}
        .block-container {{
            padding-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")

# ---------------- Load Model ---------------- #
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

weight = model["weight"]
bias = model["bias"]

# ---------------- Load Data ---------------- #
data = pd.read_csv("data.csv")
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
x_data = data.drop(['diagnosis'], axis=1)
feature_names = x_data.columns
min_vals = x_data.min().values
max_vals = x_data.max().values

# ---------------- Sidebar ---------------- #
st.sidebar.title("🔬 Breast Cancer Detection")
st.sidebar.markdown("""
This app uses a custom **Logistic Regression** model to predict if a tumor is:

- 🟢 **Benign**
- 🔴 **Malignant**

Trained on the **Wisconsin Breast Cancer Dataset** using 30 input features.

**Accuracy:**
- Train: ~98%  
- Test: ~96%

📌 Fill out all fields and click **Predict**.
""")

# ---------------- Input Section ---------------- #
st.title("📋 Enter Tumor Feature Values")

input_data = []
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.text_input(f"{feature}", value="0.0")
        try:
            val = float(val)
        except ValueError:
            st.warning(f"Invalid input for {feature}. Please enter a number.")
            st.stop()
        input_data.append(val)

# ---------------- Prediction ---------------- #
if st.button("🔎 Predict"):
    input_array = np.array(input_data)
    norm_input = (input_array - min_vals) / (max_vals - min_vals)
    norm_input = norm_input.reshape(-1, 1)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z = np.dot(weight.T, norm_input) + bias
    prediction = sigmoid(z)
    prediction_class = 1 if prediction > 0.5 else 0

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    if prediction_class == 1:
        st.error("🔴 The tumor is likely **Malignant (Cancerous)**.")
    else:
        st.success("🟢 The tumor is likely **Benign (Non-cancerous)**.")

    st.markdown(f"**Confidence Score:** {prediction[0,0]*100:.2f}%")

    # ---------------- Feature Descriptions (Plain Text) ---------------- #
st.markdown("---")
st.subheader("📌 Input Feature Descriptions")

st.markdown("""
**Mean Features (first 10)**  
- `radius_mean`: Average distance from center to edge  
- `texture_mean`: Variation in pixel intensities  
- `perimeter_mean`: Average perimeter of tumor  
- `area_mean`: Size of tumor area  
- `smoothness_mean`: Edge smoothness  
- `compactness_mean`: Compact shape of tumor  
- `concavity_mean`: Depth of concave portions  
- `concave points_mean`: Number of concave points  
- `symmetry_mean`: Symmetry of tumor  
- `fractal_dimension_mean`: Complexity of tumor boundary  

**Standard Error Features (middle 10)**  
- Measure of uncertainty/variance in the above features  
- Example: `radius_se`, `area_se`, etc.

**Worst Features (last 10)**  
- Maximum abnormal values across all feature types  
- Represents the most severe observations seen in the tumor
""")
