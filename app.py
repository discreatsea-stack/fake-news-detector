import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
import pandas as pd
import plotly.express as px

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Data Science Project UAS - Fake News Detection", page_icon="🎓", layout="wide")

# --- DOWNLOAD NLTK RESOURCE ---
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- FUNGSI PREPROCESSING DARI COLAB ---
@st.cache_resource
def setup_nlp():
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor', 'never'}
    stop_words = stop_words - negation_words
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = setup_nlp()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_colab.pkl')
    model = joblib.load('model_lr_colab.pkl')
    return tfidf, model

try:
    tfidf, model = load_models()
except FileNotFoundError:
    st.error("⚠️ Model belum ada! Jalankan 'python train_model.py' terlebih dahulu.")
    st.stop()

# --- LOAD DATA UNTUK EDA (Ambil sampel agar web tidak lambat) ---
@st.cache_data
def load_data_for_eda():
    # Kita menggunakan data yang sudah bersih dari Colab jika ada, atau data asli
    try:
        # Cobalah membaca file clean jika sudah kamu download dari Colab
        df = pd.read_csv('fake_news_clean.csv')
        return df.sample(5000, random_state=42) # Ambil 5000 sampel agar grafik cepat dirender
    except FileNotFoundError:
        # Jika file clean tidak ada, kita gabung dataset asli secara dinamis (contoh kecil)
        df_true = pd.read_csv('True.csv')[['title', 'text']].sample(2500)
        df_fake = pd.read_csv('Fake.csv')[['title', 'text']].sample(2500)
        df_true['label'] = 1 # 1 untuk Real
        df_fake['label'] = 0 # 0 untuk Fake
        df_combined = pd.concat([df_true, df_fake], ignore_index=True)
        # Hitung panjang teks untuk visualisasi
        df_combined['text_len'] = df_combined['text'].apply(lambda x: len(str(x).split()))
        return df_combined

# ==========================================
# SIDEBAR NAVIGASI
# ==========================================
st.sidebar.title("📌 Navigasi Menu")
st.sidebar.markdown("Silakan pilih halaman di bawah ini:")
menu = st.sidebar.radio("", ["📊 Eksplorasi Data (EDA)", "🤖 Prediksi Model AI"])

st.sidebar.write("---")
st.sidebar.info("Proyek UAS Sains Data\n\nTopik: Fake News Detection")

# ==========================================
# HALAMAN 1: EKSPLORASI DATA (EDA)
# ==========================================
if menu == "📊 Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data & Karakteristik")
    st.markdown("Halaman ini menampilkan karakteristik dari dataset berita yang digunakan untuk melatih model Machine Learning.")
    
    # Load data
    df_eda = load_data_for_eda()
    
    st.subheader("1. Cuplikan Data Raw")
    st.dataframe(df_eda.head(10), use_container_width=True)
    
    # --- VISUALISASI INTERAKTIF (PLOTLY) ---
    st.write("---")
    st.subheader("2. Visualisasi Interaktif")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribusi Label Berita**")
        # Visualisasi 1: Pie Chart Distribusi Kelas
        label_counts = df_eda['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Jumlah']
        label_counts['Kategori'] = label_counts['Label'].map({1: 'Real News', 0: 'Fake News'})
        
        fig_pie = px.pie(label_counts, values='Jumlah', names='Kategori', 
                         color='Kategori', color_discrete_map={'Real News':'#2ecc71', 'Fake News':'#e74c3c'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("**Distribusi Panjang Teks Berita**")
        # Visualisasi 2: Histogram Panjang Kata
        if 'text_len' not in df_eda.columns:
             df_eda['text_len'] = df_eda['text'].apply(lambda x: len(str(x).split()))
             
        # Menambahkan kolom nama kategori untuk legend
        df_eda['Kategori'] = df_eda['label'].map({1: 'Real News', 0: 'Fake News'})
        
        fig_hist = px.histogram(df_eda, x="text_len", color="Kategori", 
                                marginal="box", nbins=50,
                                color_discrete_map={'Real News':'#2ecc71', 'Fake News':'#e74c3c'},
                                labels={"text_len": "Jumlah Kata dalam Berita"})
        # Batasi sumbu X agar lebih rapi (menghindari outlier teks yang terlalu panjang)
        fig_hist.update_xaxes(range=[0, 1500])
        st.plotly_chart(fig_hist, use_container_width=True)


# ==========================================
# HALAMAN 2: PREDIKSI MODEL AI
# ==========================================
elif menu == "🤖 Prediksi Model AI":
    st.title("🤖 Prediksi Berita Menggunakan AI")
    st.markdown("Masukkan artikel atau teks berita berbahasa Inggris pada form di bawah ini. Model *Logistic Regression* akan memprediksi apakah berita tersebut asli atau palsu berdasarkan pola kata.")
    st.write("---")

    user_input = st.text_area("📄 Form Input Teks Berita:", height=250, placeholder="Paste your English news content here...")

    if st.button("Jalankan Prediksi 🔍", type="primary", use_container_width=True):
        if len(user_input.strip()) < 10:
            st.warning("Mohon masukkan teks yang lebih panjang untuk dianalisis!")
        else:
            with st.spinner('Menganalisis pola kalimat menggunakan Machine Learning...'):
                time.sleep(1) 
                
                # Preprocessing dan Prediksi
                cleaned_input = clean_text(user_input)
                tfidf_input = tfidf.transform([cleaned_input])
                prediction = model.predict(tfidf_input)[0]
                probabilities = model.predict_proba(tfidf_input)[0]
                
                prob_fake = probabilities[0] * 100
                prob_real = probabilities[1] * 100
                
            st.write("---")
            st.subheader("Hasil Analisis:")
            
            # Layout Hasil
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediction == 0:
                    st.error("🚨 **FAKE NEWS**")
                else:
                    st.success("✅ **REAL NEWS**")
                    
            with res_col2:
                if prediction == 0:
                    st.progress(int(prob_fake))
                    st.caption(f"Tingkat Keyakinan Model: **{prob_fake:.2f}%** (Berita Palsu)")
                else:
                    st.progress(int(prob_real))
                    st.caption(f"Tingkat Keyakinan Model: **{prob_real:.2f}%** (Berita Asli)")
                
            with st.expander("Klik untuk melihat teks yang telah di-preprocessing (Input Model)"):
                st.write(f"*{cleaned_input}*")