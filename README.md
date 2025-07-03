# Laporan Proyek Machine Learning
**Domain Proyek:** Prediksi Attrition Karyawan untuk Manajemen Sumber Daya Manusia

**Nama:** Muhamad Rajwa Athoriq

Dalam era persaingan bisnis yang ketat, manajemen sumber daya manusia (SDM) menjadi faktor krusial yang menentukan kesuksesan perusahaan. Tingkat turnover karyawan yang tinggi dapat berdampak signifikan terhadap operasional dan profitabilitas perusahaan. Menurut penelitian dari Society for Human Resource Management (SHRM), biaya untuk mengganti satu karyawan dapat mencapai 50-200% dari gaji tahunan karyawan tersebut, tergantung pada tingkat posisi dan keahlian yang dibutuhkan.

Sistem evaluasi karyawan tradisional seringkali bergantung pada penilaian subjektif dan indikator performa yang terbatas, sehingga sulit untuk memprediksi secara akurat karyawan mana yang berpotensi untuk resign. Hal ini mengakibatkan dua masalah utama: pertama, kehilangan talenta berharga secara mendadak yang mengganggu kontinuitas bisnis; dan kedua, biaya rekrutmen dan pelatihan yang membengkak akibat tingkat turnover yang tinggi.

Oleh karena itu, diperlukan pendekatan yang lebih akurat dan efisien dalam memprediksi attrition karyawan dengan memanfaatkan teknologi machine learning dan data analytics. Proyek ini akan membangun model prediktif untuk mengidentifikasi karyawan yang berpotensi resign, sehingga dapat membantu perusahaan dalam mengambil tindakan preventif yang tepat waktu.

**Referensi:**
- Society for Human Resource Management. (2023). Employee Turnover and Retention Statistics
- Harvard Business Review. (2022). The Cost of Employee Turnover
- McKinsey & Company. (2023). The Future of Work: Talent Retention in the Digital Age

## Business Understanding

### Problem Statements
1. Bagaimana cara mengidentifikasi karyawan yang berpotensi resign berdasarkan karakteristik personal dan performa kerja mereka?
2. Faktor-faktor apa saja yang paling signifikan memengaruhi keputusan karyawan untuk meninggalkan perusahaan?
3. Bagaimana mengoptimalkan strategi retensi karyawan untuk meminimalkan tingkat turnover dan memaksimalkan produktivitas tim?

### Goals
1. Mengembangkan model prediktif yang dapat mengklasifikasikan karyawan menjadi kategori "Stay" (bertahan) atau "Resign" (keluar) dengan akurasi yang tinggi
2. Mengidentifikasi faktor-faktor atau fitur yang paling berpengaruh dalam menentukan keputusan karyawan untuk resign
3. Menyediakan solusi analitik yang dapat membantu pengambilan keputusan manajemen SDM yang lebih objektif dan konsisten

### Solution Statements
1. Melakukan eksplorasi dan analisis mendalam terhadap dataset historis karyawan untuk memahami pola dan tren yang memengaruhi attrition
2. Membangun dan membandingkan beberapa algoritma klasifikasi seperti Logistic Regression, Decision Tree, Random Forest, dan AdaBoost untuk memprediksi attrition karyawan
3. Menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk mengatasi masalah ketidakseimbangan kelas pada data
4. Melakukan tuning hyperparameter untuk model terbaik guna meningkatkan performa prediksi
5. Menganalisis feature importance untuk mengidentifikasi faktor-faktor yang paling signifikan dalam prediksi attrition karyawan

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data historis karyawan yang mencakup informasi demografis, performa kerja, kepuasan karyawan, dan status resign. Dataset ini bersumber dari perusahaan yang ingin memahami pola attrition karyawan untuk meningkatkan strategi retensi.

- **Jumlah data:** 287 baris
- **Jumlah fitur awal:** 25 kolom
- **Jumlah fitur setelah praproses:** 14 kolom (termasuk target)
- **Tipe data:** Gabungan antara data numerik dan kategorikal
- **Target variable:** Resign (Stay/Resign)

### Variabel-variabel pada Dataset

| No. | Nama Kolom | Deskripsi |
|-----|------------|-----------|
| 1 | StatusPernikahan | Status pernikahan karyawan |
| 2 | JenisKelamin | Jenis kelamin karyawan |
| 3 | StatusKepegawaian | Status kepegawaian (tetap/kontrak) |
| 4 | Pekerjaan | Jenis pekerjaan atau posisi |
| 5 | JenjangKarir | Tingkat karir (Fresh Graduate/Mid Level/Senior) |
| 6 | PerformancePegawai | Rating performa karyawan |
| 7 | SkorSurveyEngagement | Skor keterlibatan karyawan |
| 8 | SkorKepuasanPegawai | Skor kepuasan kerja karyawan |
| 9 | JumlahKeikutsertaanProjek | Jumlah proyek yang diikuti |
| 10 | JumlahKeterlambatanSebulanTerakhir | Jumlah keterlambatan dalam sebulan terakhir |
| 11 | TingkatPendidikan | Tingkat pendidikan karyawan |
| 12 | LengthWorked | Lama bekerja di perusahaan |
| 13 | AgeAtResign | Usia saat resign (0 jika masih bekerja) |
| 14 | Resign | Status resign (Stay/Resign) - Target Variable |

### Distribusi Target Variable (Resign)
Analisis distribusi variabel target menunjukkan ketidakseimbangan kelas yang signifikan:

- **Kelas 'Stay' (karyawan bertahan):** ~70%
- **Kelas 'Resign' (karyawan keluar):** ~30%

Ketidakseimbangan ini akan diatasi dengan teknik SMOTE pada tahap pra-pemodelan.

### Analisis Univariat

**1. Jenjang Karir:** Mayoritas karyawan berada pada level Mid Level, diikuti oleh Senior Level dan Fresh Graduate.

**2. Jenis Kelamin:** Distribusi cukup seimbang antara pria dan wanita dengan sedikit dominasi pria.

**3. Status Pernikahan:** Mayoritas karyawan sudah menikah, diikuti oleh yang belum menikah.

**4. Performa Karyawan:** Sebagian besar karyawan memiliki performa "Bagus" dan "Sangat Bagus".

**5. Tingkat Pendidikan:** Didominasi oleh lulusan Sarjana, diikuti Magister dan Doktor.

### Analisis Multivariat

**1. Usia dan Resign:** Karyawan yang lebih tua cenderung memiliki tingkat resign yang lebih tinggi, terutama yang mendekati usia pensiun.

**2. Lama Bekerja:** Karyawan dengan masa kerja yang lebih lama cenderung lebih stabil dan jarang resign.

**3. Skor Kepuasan:** Karyawan dengan skor kepuasan rendah memiliki probabilitas resign yang jauh lebih tinggi.

**4. Performa vs Resign:** Karyawan dengan performa rendah lebih cenderung resign, namun ada juga karyawan berprestasi tinggi yang resign karena faktor lain.

**5. Jenjang Karir:** Fresh Graduate memiliki tingkat turnover tertinggi, diikuti Mid Level, sedangkan Senior Level paling stabil.

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data agar siap digunakan untuk pemodelan. Berikut adalah langkah-langkah yang dilakukan:

### 1. Data Cleaning
- **Penanganan Missing Values:**
  - Kolom numerik: Diisi menggunakan nilai median
  - Kolom kategorikal: Diisi menggunakan nilai modus (nilai yang paling sering muncul)
- **Penghapusan Kolom:** Menghapus fitur yang tidak relevan
  - `IkutProgramLOP` (90% missing values)
  - `Username`, `EnterpriseID`, `NomorHP`, `Email` (identifier non-prediktif)
  - `PernahBekerja` (tidak relevan untuk analisis)

### 2. Data Correction
- **StatusPernikahan** dan **AlasanResign**: Memperbaiki nilai invalid menggunakan penggantian modus

### 3. Feature Engineering
- **YearBirth, YearHiring, YearPenilaian, YearResign**: Ekstraksi tahun dari kolom tanggal
- **AgeAtResign**: Menghitung usia saat resign
- **LengthWorked**: Menghitung durasi kerja
- **Resign**: Membuat variabel target biner (Stay/Resign)

### 4. Encoding Values
- **Label Encoding**: Diterapkan pada fitur ordinal (JenjangKarir, TingkatPendidikan, PerformancePegawai)
- **One-Hot Encoding**: Diterapkan pada fitur kategorikal nominal
- **Target Encoding**: Mengkonversi status resign ke biner (0: Stay, 1: Resign)

### 5. Transformation Values
- **StandardScaler**: Menormalkan skala nilai pada setiap fitur agar memiliki mean=0 dan std=1

### 6. Splitting Data & Data Balancing
- **Train-Test Split**: Membagi dataset dengan rasio 80:20
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Menyeimbangkan kelas target untuk mengatasi ketidakseimbangan data
  - Sebelum SMOTE: Data tidak seimbang
  - Setelah SMOTE: Data seimbang untuk training

## Modeling

Pada proyek ini, dilakukan perbandingan beberapa algoritma klasifikasi untuk memprediksi attrition karyawan. Berikut adalah beberapa mode yang digunakan:

### 1. Logistic Regression
Logistic Regression adalah algoritma klasifikasi yang memodelkan probabilitas kelas target menggunakan fungsi logistik sigmoid. Model ini cocok untuk masalah klasifikasi biner dan memberikan output probabilitas yang dapat diinterpretasikan.

**Kelebihan:**
- Interpretabilitas tinggi, koefisien model menunjukkan pentingnya setiap fitur
- Pemodelan yang efisien dengan kompleksitas komputasi rendah
- Memberikan probabilitas yang terkalibrasi dengan baik

**Kekurangan:**
- Asumsi hubungan linear antara fitur dan log-odds
- Performa mungkin tidak sebaik model kompleks pada data non-linear
- Sensitif terhadap multikolinearitas

### 2. Random Forest
Random Forest adalah ensemble method yang menggabungkan banyak decision trees untuk meningkatkan akurasi dan mengurangi overfitting. Model ini menggunakan teknik bootstrap aggregating dan random feature selection.

**Kelebihan:**
- Mengurangi overfitting dibandingkan single decision tree
- Memberikan feature importance yang berguna
- Robust terhadap outlier dan noise

**Kekurangan:**
- Kurang interpretable dibandingkan single decision tree
- Membutuhkan memory yang lebih besar
- Dapat overfitting pada data yang sangat noisy

### 3. AdaBoost
AdaBoost adalah teknik ensemble yang menggabungkan beberapa weak learners (biasanya Decision Tree sederhana) menjadi model yang kuat. AdaBoost bekerja dengan memberikan bobot lebih pada data yang salah diklasifikasikan pada iterasi sebelumnya.

**Kelebihan:**
- Mampu mengurangi bias dan variance
- Cenderung tidak overfitting pada dataset besar
- Performa baik pada berbagai jenis data

**Kekurangan:**
- Sensitif terhadap noise dan outlier
- Komputasi lebih intensif dibandingkan model tunggal
- Parameter boosting perlu dituning dengan hati-hati

## Evaluation

Untuk mengevaluasi kinerja model dalam memprediksi attrition karyawan, digunakan berbagai metrik evaluasi yang relevan dengan masalah klasifikasi yang memiliki ketidakseimbangan kelas. Berikut adalah metrik-metrik yang digunakan:

### Metrik Evaluasi
- **Accuracy**: Persentase prediksi benar dari total prediksi
- **Precision**: Proporsi prediksi positif yang benar; penting untuk menghindari false positive
- **Recall**: Proporsi kasus positif yang berhasil dikenali; penting untuk menangkap semua karyawan berisiko
- **F1-Score**: Rata-rata harmonik precision dan recall; cocok saat keduanya sama penting
- **ROC AUC**: Mengukur kemampuan model membedakan kelas; semakin tinggi, semakin baik performa

Fokus utama evaluasi dalam proyek ini berada pada **F1-Score**, karena data yang tidak seimbang membutuhkan keseimbangan antara precision dan recall untuk memastikan model tidak hanya akurat dalam mengenali karyawan berisiko resign, tetapi juga tidak melewatkan terlalu banyak kasus penting.

### Hasil Evaluasi Model

#### Perbandingan Model dengan SMOTE

| Model              | acc_train | acc_test | prec_train | prec_test | rec_train | rec_test | f1_train | f1_test | roc_train | roc_test |
|--------------------|-----------|----------|------------|-----------|-----------|----------|----------|---------|-----------|----------|
| Logistic Regression ✨| 0.99      | 0.98     | 0.99       | 0.95      | 1.00      | 1.00     | 0.99     | 0.97    | 0.99      | 0.99     |
| Random Forest      | 1.00      | 0.98     | 1.00       | 0.95      | 1.00      | 1.00     | 1.00     | 0.97    | 1.00      | 0.99     |
| Ada Boost          | 1.00      | 0.98     | 1.00       | 0.95      | 1.00      | 1.00     | 1.00     | 0.97    | 1.00      | 0.99     |


#### Kesimpulan Model Terbaik
Dari beberapa model yang diuji, **Logistic Regression dengan SMOTE** menunjukkan performa terbaik dengan keseimbangan yang optimal antara semua metrik evaluasi. Model ini dipilih karena:

1. **Performa Konsisten**: Mencapai <95% di semua metrik evaluasi
2. **Minimalisasi Overfitting**: Gap kecil antara skor training dan testing
3. **Interpretabilitas Tinggi**: Mudah dipahami dan dijelaskan untuk stakeholder bisnis
4. **Efisiensi Komputasi**: Waktu training yang relatif cepat

### Hyperparameter Tuning

Setelah menentukan algoritma terbaik yaitu Logistic Regression, dilakukan Hyperparameter Tuning menggunakan RandomizedSearchCV untuk memaksimalkan performa model. Parameter yang diuji meliputi:

- **C**: Mengatur kekuatan regularisasi (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
- **Penalty**: Jenis regularisasi ('l1', 'l2', 'elasticnet', 'none')
- **Solver**: Algoritma optimasi ('lbfgs', 'liblinear', 'saga', 'newton-cg')
- **Max Iter**: Jumlah iterasi maksimum (100, 200, 500, 1000)

#### Hasil Hyperparameter Tuning:

Parameter terbaik yang dihasilkan adalah:

- **C**: 1.0
- **Penalty**: 'l1'
- **Solver**: 'saga'
- **Max Iter**: 1000
- **l1_ratio**: 1.0


| Model                          | f1_train | f1_test |
|--------------------------------|----------|---------|
| Logistic Regression (Default)  | 0.99     | 0.97    |
| Logistic Regression (Tuning)   | 0.99     | 0.97    |


### Cross-Validation Results
Model yang dibangun divalidasi menggunakan 5-fold cross-validation:

| Fold | Score             |
|------|------------------|
| 1    | 0.9846153846153847 |
| 2    | 0.9846153846153847 |
| 3    | 0.9850746268656716 |
| 4    | 1.0                |
| 5    | 1.0                |

**Mean cross-validation score:** 0.9908610792192881

## Feature Importance

Menggunakan analisis **SHAP (SHapley Additive exPlanations)**, faktor-faktor yang paling berpengaruh dalam memprediksi attrition karyawan adalah: 

1. **Usia Resign**: Karyawan yang lebih tua cenderung resign lebih sering → Tawarkan fleksibilitas kerja dan program pensiun.
2. **Lama Bekerja**: Masa kerja lebih lama → Risiko resign lebih kecil → Berikan penghargaan dan jalur karier.
3. **Tingkat Pendidikan**: Pendidikan tinggi → Cenderung lebih mudah resign → Libatkan dalam proyek menantang & pengembangan karier.
4. **Kepuasan Kerja**: Kepuasan rendah → Risiko resign tinggi → Tingkatkan kompensasi, lingkungan kerja, dan keseimbangan hidup.

## Kesimpulan

### 🚪 Alasan Umum Karyawan Resign & Solusi
Kurangnya Fleksibilitas → Tawarkan jam kerja fleksibel & opsi remote.
1. Tidak Jelasnya Jalur Karir → Sediakan pelatihan & pengembangan yang terstruktur.
2. Budaya Kerja Negatif → Bangun lingkungan kerja yang sehat & suportif.
3. Kepemimpinan Lemah → Latih manajer untuk jadi pemimpin yang inspiratif.
4. Kurangnya Apresiasi → Terapkan sistem penghargaan untuk karyawan berprestasi.

### Dampak Bisnis
Tanpa menggunakan machine learning, perusahaan mungkin hanya dapat mempertahankan sekitar 70% karyawan berdasarkan intuisi dan pengalaman. Namun, dengan penerapan machine learning, perusahaan dapat:

1. **Meningkatkan prediksi akurasi** hingga 97% dalam mengidentifikasi karyawan berisiko resign
2. **Menghemat biaya rekrutmen** dengan tindakan preventif yang tepat waktu
3. **Meningkatkan strategi retensi** berdasarkan faktor-faktor yang paling berpengaruh
4. **Optimalisasi alokasi sumber daya** untuk program pengembangan karyawan

## Sumber Dataset

**Link Dataset:** [Google Drive](https://drive.google.com/file/d/1YyZfVX2xakEH_YDei-Uoefsihg7Kr7eK/view?usp=sharing)

## Technical Stack

- **Python 3.x**
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Balancing:** Imbalanced-learn (SMOTE)
- **Model Interpretation:** SHAP
- **Environment:** Jupyter Notebook

## Cara Menjalankan Proyek

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost imbalanced-learn shap
```

### Langkah Eksekusi
1. Clone repository ini
2. Letakkan dataset (`Improving Employee Retention by Predicting Employee Attrition Using Machine Learning.xlsx - hr_data.csv`) dalam direktori proyek
3. Buka dan jalankan `PredictingEmployeeAttrition.ipynb`
4. Ikuti analisis step-by-step

### Struktur File
```
├── PredictingEmployeeAttrition.ipynb    # Notebook analisis utama
├── Improving Employee Retention by Predicting Employee Attrition Using Machine Learning.xlsx - hr_data.csv                          # File dataset
├── README.md                            # Dokumentasi proyek
└── Portofolio_TrunoverEmploye.pdf      # Portfolio proyek
```

## Future Improvements

1. **Fitur Tambahan:** Data gaji, lokasi kerja, dinamika tim
2. **Model Lanjutan:** Ensemble methods, deep learning approaches
3. **Aplikasi Real-time:** Pengembangan web application
4. **Analisis Longitudinal:** Time-series analysis tren kepuasan
5. **Analisis Cost-Benefit:** Perhitungan ROI strategi retensi

## Kontak

Untuk pertanyaan atau kolaborasi:
**Muhamad Rajwa Athoriq**

---

*Proyek ini mendemonstrasikan penerapan praktis teknik machine learning untuk menyelesaikan tantangan nyata dalam manajemen SDM dan memberikan insight yang dapat ditindaklanjuti untuk meningkatkan strategi retensi karyawan.*
