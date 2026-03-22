import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===========================
# Load Models & Data
# ===========================
pipe_rf = joblib.load("pipe_joblib.pkl")
df = joblib.load("df_joblib.pkl")

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="Laptop Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Custom UI Styling
# ===========================
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #1c1c2b, #2b2b3f);
    color: white;
}
h1 {
    color: #FFD700;
    text-align:center;
}
.stButton>button {
    background-color: #FF4500;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #FFA500;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ===========================
# Header
# ===========================
st.markdown("<h1>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.write("### Enter laptop specifications to predict price")
st.write("---")

# ===========================
# Sidebar Inputs
# ===========================
st.sidebar.header("Configure Laptop")

company = st.sidebar.selectbox("Brand", df['Company'].unique())
type_laptop = st.sidebar.selectbox("Type", df['TypeName'].unique())

ram = st.sidebar.selectbox("RAM (GB)", [2,4,6,8,12,16,24,32,64])
weight = st.sidebar.number_input("Weight (kg)", 0.5, 5.0, 2.0)

touchscreen = st.sidebar.selectbox("Touchscreen", ["No","Yes"])
ips = st.sidebar.selectbox("IPS Panel", ["No","Yes"])

screen_size = st.sidebar.slider("Screen Size", 10.0, 18.0, 13.0)

resolution = st.sidebar.selectbox("Resolution", [
    '1920x1080','1366x768','1600x900','3840x2160',
    '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
])

cpu = st.sidebar.selectbox("CPU", df['Cpu brand'].unique())

hdd = st.sidebar.selectbox("HDD (GB)", [0,128,256,512,1024,2048])
ssd = st.sidebar.selectbox("SSD (GB)", [0,8,128,256,512,1024])

gpu = st.sidebar.selectbox("GPU", df['Gpu brand'].unique())
os = st.sidebar.selectbox("OS", df['os'].unique())

# ===========================
# Prediction
# ===========================
if st.sidebar.button("🚀 Predict Price"):

    # Encode categorical
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Calculate PPI
    X_res, Y_res = map(int, resolution.split("x"))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_laptop],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen_val],
        'Ips': [ips_val],
        'ppi': [ppi],
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })
    
    predicted_price = np.exp(pipe_rf.predict(query)[0])

    # ===========================
    # Output UI
    # ===========================
    st.markdown(f"""
    <div style='
        background-color: rgba(255,69,0,0.9);
        padding: 25px;
        border-radius: 15px;
        text-align:center;
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin: 20px auto;
        width: 60%;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    '>
    💰 Predicted Price: ₹ {int(predicted_price):,}
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

# ===========================
# Footer
# ===========================
st.markdown("---")
st.markdown("### 🚀 Tips")
st.markdown("""
- Use realistic configurations  
- Higher SSD & RAM → higher price  
- IPS & Touchscreen increase cost  
""")
