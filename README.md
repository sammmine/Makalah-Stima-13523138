# 📖 Analisis Cross-Reference dalam Teks Alkitab (World English Bible) 📚

Makalah IF2211 Strategi Algoritma  
Samantha Laqueenna Ginting - 13523138  
Institut Teknologi Bandung  
Semester II Tahun 2024/2025

## ✨ Deskripsi Proyek

Repositori ini berisi implementasi sistem analisis teks berbasis Python untuk mendeteksi dan menganalisis **rujukan silang (cross-reference)** dalam teks **Alkitab versi World English Bible (WEB)**. Tiga fokus utama penelitian ini:

1. **Deteksi Tema Tematik Penting**  
   Menggunakan *Boyer-Moore Algorithm* dan *Regular Expressions* untuk mendeteksi tema besar seperti:
   - Perjanjian: “covenant”, “promise”, dll
   - Penebusan: “sin”, “redeem”, dll
   - Nubuat: “prophecy”, “Messiah”, dll

2. **Deteksi Referensi Silang Antar Kitab**  
   Menggunakan *Regex Pattern* + *Fuzzy Matching* (difflib) untuk mengenali frasa seperti “as it is written in Isaiah”.

3. **Analisis Frekuensi Nama Penting**  
   Mendeteksi nama tokoh/lokasi yang sering muncul dengan penyaringan *stopwords* dan analisis kapitalisasi.

## 🧠 Algoritma yang Digunakan

- **Boyer-Moore**: Efisien untuk pencarian kata/frasa dari kanan ke kiri menggunakan *bad character heuristic*.
- **Regular Expression**: Untuk mencocokkan pola frasa yang fleksibel dan kompleks.
- **Fuzzy Matching**: Menggunakan Levenshtein similarity untuk mencocokkan nama kitab yang tidak eksplisit.
- **Word Frequency Counter**: Menghitung dan mengurutkan kata penting dalam teks berdasarkan frekuensi kemunculan.