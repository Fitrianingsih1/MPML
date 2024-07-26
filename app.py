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
# CSS untuk gaya dengan desain modern dan menarik
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5; /* Light Gray */
        font-family: 'Poppins', sans-serif;
    }
    h1, h3 {
        color: #333333; /* Dark Gray */
        font-family: 'Poppins', sans-serif;
        text-align: center;
        padding: 20px;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #007BFF; /* Blue */
        color: #FFFFFF; /* White */
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-family: 'Poppins', sans-serif;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s, transform 0.2s;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker Blue */
        transform: scale(1.05); /* Zoom effect on hover */
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    .stSelectbox, .stNumberInput {
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
        border: 1px solid #DDDDDD;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .prediction-output {
        color: #28A745; /* Green */
        font-size: 20px;
        text-align: center;
        font-weight: 600;
        padding: 20px;
        border: 2px solid #28A745;
        border-radius: 8px;
        background-color: #E9FBE9; /* Light Green */
    }
    .info-text {
        color: #6C757D; /* Gray */
        text-align: center;
        font-size: 14px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
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
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        result = 'Terdapat feedback' if prediction[0] == 1 else 'Tidak terdapat feedback'
        st.markdown(f"<h3 class='prediction-output'> {result}</h3>", unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
