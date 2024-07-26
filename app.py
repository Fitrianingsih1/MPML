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

# CSS untuk gaya dengan warna baby pink dan biru tua
st.markdown("""
    <style>
    .main {
        background-color: #FDE2E4; /* Baby Pink */
        font-family: 'Baloo 2', cursive;
    }
    h1, h3, label, .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #003366; /* Dark Blue */
        font-family: 'Baloo 2', cursive;
        text-align: center; /* Memusatkan teks */
    }
    .stButton>button {
        background-color: #FFC1CC; /* Light Pink */
        color: #003366; /* Dark Blue */
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-family: 'Baloo 2', cursive;
    }
    .stButton>button:hover {
        background-color: #FFA6C9; /* Darker Pink */
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    .stSelectbox, .stNumberInput {
        padding: 6px 12px; /* Mengurangi padding di dalam input box */
        font-size: 14px; /* Ukuran teks lebih besar */
    }
    .prediction-output {
        color: #003366; /* Dark Blue */
        font-size: 18px; /* Ukuran teks lebih besar */
        text-align: center; /* Memusatkan teks */
    }
    .info-text {
        color: #003366; /* Dark Blue */
        text-align: center; /* Memusatkan teks */
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
st.title("Feedback Pelanggan Online Food")

st.markdown("<h3 style='text-align: center;'>Masukkan Data Pelanggan</h3>", unsafe_allow_html=True)

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
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        # Pemetaan hasil model ke "Pelanggan terdaftar" atau "Pelanggan tidak terdaftar"
        result = 'Terdapat feedback' if prediction[0] == 1 else 'Tidak terdapat feedback'
        st.markdown(f"<h3> {result}</h3>", unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error in prediction: {e}")