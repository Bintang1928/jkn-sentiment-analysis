# 🔮 JKN Mobile Sentiment Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)
![Google AI](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?style=for-the-badge&logo=google&logoColor=white)

**Aplikasi Berbasis Web untuk Analisis Sentimen Ulasan Aplikasi JKN Mobile menggunakan Multi-Model AI**

</div>

---

## 📌 Deskripsi Proyek

**JKN Mobile Sentiment Analyzer** adalah aplikasi web interaktif yang dibangun menggunakan **Python** dan **Streamlit**. Aplikasi ini dirancang untuk menganalisis opini dan masukan pengguna (sentimen) terhadap aplikasi JKN Mobile secara otomatis, tepat, dan mendalam.

Aplikasi ini mendukung **3 Pendekatan Model AI/ML**:
1. **Naive Bayes + TF-IDF**: Pendekatan Machine Learning klasik yang ringan dan cepat berdasarkan bobot probabilitas kata.
2. **IndoBERT**: Pendekatan Deep Learning menggunakan model Transformer (*pre-trained*) khusus bahasa Indonesia untuk memahami representasi kontekstual mendalam.
3. **Google Gemini (LLM)**: Pendekatan Generative AI (*Large Language Model*) langsung melalui API untuk klasifikasi sentimen disertai **alasan (*reasoning*)** berbasis konteks.

---

## ✨ Fitur Utama

### 1. 📁 Batch Excel Analysis (Analisis Massal)
- **Unggah Dokumen Excel (.xlsx)**: Mendukung dokumen Excel yang berisi kolom `id` dan `comment`.
- **Pilihan Model ML/DL**: Pilih model **Naive Bayes** atau **IndoBERT** untuk memproses ribuan baris ulasan sekaligus dengan cepat.
- **Preview Data**: Tampilan pratinjau tabel interaktif sebelum dan sesudah analisis.
- **Visualisasi Interaktif (Plotly)**:
  - **Bar Chart**: Menampilkan jumlah ulasan pada setiap kategori sentimen (`Positif`, `Netral`, `Negatif`).
  - **Pie Chart**: Menampilkan proporsi persentase distribusi sentimen secara keseluruhan.
- **Export / Unduh Hasil**: Hasil analisis otomatis ditambahkan ke dalam kolom baru bernama `sentiment_result` dan dapat langsung diunduh dalam format `.xlsx`.

### 2. 💬 Single Review Analysis (Analisis Ulasan Tunggal)
- **Pengecekan Teks Langsung**: Uji coba kalimat ulasan atau masukan pengguna secara *real-time*.
- **Dukungan Multi-Model**: Dapat diuji menggunakan **Naive Bayes**, **IndoBERT**, maupun **Google Gemini API**.
- **💡 AI Reasoning (Khusus Gemini)**: Ketika menggunakan model Gemini, aplikasi tidak hanya memberikan label sentimen (`Positif`, `Netral`, `Negatif`), tetapi juga menghasilkan **penjelasan/alasan ilmiah singkat** mengapa ulasan tersebut diklasifikasikan ke dalam label tersebut.

---

## 🛠️ Arsitektur & Struktur Direktori

```text
SKRIPSI/
│
├── app.py                     # Antarmuka web utama (Streamlit UI & visualisasi)
├── models.py                  # Logika pemuatan model (Naive Bayes, IndoBERT, Gemini API)
├── requirements.txt           # Daftar dependensi perpustakaan Python
├── run_st.sh                  # Skrip bash alternatif untuk menjalankan aplikasi
│
├── naive_bayes_model.pkl      # Model trained Naive Bayes (joblib/pickle)
├── tfidf_vectorizer.pkl       # TF-IDF Vectorizer pasangan Naive Bayes
│
└── indobert_sentiment_model_full/  # Folder model trained IndoBERT
    ├── model/
    │   ├── config.json
    │   └── model.safetensors
    └── tokenizer/
        ├── tokenizer.json
        └── tokenizer_config.json
```

---

## 🚀 Cara Instalasi & Menjalankan Aplikasi

### 1. Prasyarat Sistem
Pastikan Anda telah menginstal **Python 3.10 atau yang lebih baru** di sistem komputer/laptop Anda.

### 2. Clone / Siapkan Direktori
Masuk ke folder proyek ini melalui terminal/command prompt:
```bash
cd path/to/SKRIPSI
```

### 3. Instalasi Dependensi
Sangat disarankan menggunakan virtual environment (*opsional*). Instal seluruh library yang dibutuhkan dengan perintah berikut:
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Aplikasi Web
Jalankan perintah berikut di terminal Anda:
```bash
streamlit run app.py
```
Atau jika menggunakan skrip bash yang tersedia:
```bash
bash run_st.sh
```

Setelah dijalankan, aplikasi akan otomatis terbuka di peramban web (*browser*) pada tautan:
👉 `http://localhost:8501`

---

## 📄 Format Dokumen Excel Input & Output

### Format File Input (`.xlsx`)
Pastikan file Excel yang diunggah pada tab **Batch Excel Analysis** memiliki minimal 2 kolom berikut:

| id | comment |
| :--- | :--- |
| 101 | Aplikasi JKN Mobile sangat membantu untuk daftar antrean faskes 1! |
| 102 | Aplikasi sering error dan loading lama saat jam sibuk. |
| 103 | Fitur lengkap, cukup standar untuk penggunaan sehari-hari. |

### Format File Output yang Diunduh (`.xlsx`)
Setelah proses analisis selesai, file Excel keluaran akan memiliki tambahan 1 kolom baru (`sentiment_result`):

| id | comment | sentiment_result |
| :--- | :--- | :--- |
| 101 | Aplikasi JKN Mobile sangat membantu untuk daftar antrean faskes 1! | **Positif** |
| 102 | Aplikasi sering error dan loading lama saat jam sibuk. | **Negatif** |
| 103 | Fitur lengkap, cukup standar untuk penggunaan sehari-hari. | **Netral** |

---

## 🔑 Penggunaan Google Gemini API
Untuk menggunakan fitur **Gemini Flash** pada tab **Single Review Analysis**:
1. Dapatkan API Key secara gratis dari [Google AI Studio](https://aistudio.google.com/).
2. Masukkan API Key Anda pada kolom *Enter Gemini API Key* di antarmuka aplikasi sebelum menekan tombol analisis.
3. Aplikasi secara otomatis akan mendeteksi model Flash terbaik yang tersedia pada akun API Anda.

---

## 👥 Kontributor / Pembuat
Dikembangkan untuk keperluan Penelitian / Skripsi mengenai Analisis Sentimen Pengguna JKN Mobile.
