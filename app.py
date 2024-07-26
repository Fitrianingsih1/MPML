import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input
# CSS untuk gaya inovatif dan modern
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #F3F4F6, #E2E2E2); /* Soft Gradient Background */
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 900px;
        margin: 30px auto;
        padding: 20px;
        background-color: #FFFFFF; /* White Background */
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .container::before {
        content: "";
        position: absolute;
        top: -10%;
        left: -10%;
        width: 120%;
        height: 120%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 10%, rgba(255, 255, 255, 0) 40%);
        z-index: 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 0.2; }
        100% { transform: scale(1); opacity: 0.5; }
    }
    h1 {
        color: #333333; /* Dark Gray */
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 2.5em;
        z-index: 1;
        position: relative;
    }
    h3 {
        color: #555555; /* Medium Gray */
        text-align: center;
        margin-bottom: 15px;
        font-weight: normal;
        font-size: 1.5em;
        z-index: 1;
        position: relative;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF6F61, #FF3F3F); /* Gradient Button */
        color: #FFFFFF; /* White */
        padding: 12px 24px;
        border: none;
        border-radius: 30px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF4C4C, #FF1A1A); /* Darker Gradient on Hover */
        transform: translateY(-3px);
    }
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .stButton>button svg {
        margin-right: 8px;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
        border: 2px solid #DDDDDD;
        border-radius: 15px;
        padding: 12px;
        font-size: 16px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s, box-shadow 0.3s;
        z-index: 1;
        position: relative;
    }
    .stNumberInput:hover, .stSelectbox:hover {
        border-color: #FF6F61; /* Coral */
        box-shadow: 0 0 8px rgba(255, 111, 97, 0.3);
    }
    .prediction-output {
        color: #28A745; /* Green */
        font-size: 22px;
        text-align: center;
        font-weight: bold;
        padding: 20px;
        border: 2px solid #28A745;
        border-radius: 15px;
        background-color: #E9FBE9; /* Light Green */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        margin-top: 20px;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    .info-text {
        color: #6C757D; /* Gray */
        text-align: center;
        font-size: 14px;
        padding: 10px;
        z-index: 1;
        position: relative;
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
st.markdown('<div class="container">', unsafe_allow_html=True)
st.title("Prediksi Feedback Pelanggan")

st.markdown("<h3>Masukkan Data Pelanggan</h3>", unsafe_allow_html=True)

# Membagi input form menjadi dua kolom
col1, col2 = st.columns(2)

# Input pengguna di kolom kiri
with col1:
    age = st.number_input('Age', min_value=18, max_value=100)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
    occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Employed'])
    monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])

# Input pengguna di kolom kanan
with col2:
    educational_qualifications = st.selectbox('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate'])
    family_size = st.number_input('Family size', min_value=1, max_value=20)
    latitude = st.number_input('Latitude', format="%f")
    longitude = st.number_input('Longitude', format="%f")
    pin_code = st.number_input('Pin code', min_value=100000, max_value=999999)

user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code
}

if st.button('Predict'):
    with st.spinner('Memproses...'):
        user_input_processed = preprocess_input(user_input)
        try:
            prediction = model.predict(user_input_processed)
            result = 'Terdapat feedback' if prediction[0] == 1 else 'Tidak terdapat feedback'
            st.markdown(f"<h3 class='prediction-output'>{result}</h3>", unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Error in prediction: {e}")

st.markdown('</div>', unsafe_allow_html=True)
