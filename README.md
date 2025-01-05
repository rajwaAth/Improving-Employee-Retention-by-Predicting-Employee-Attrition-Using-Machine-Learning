# Improving-Employee-Retention-by-Predicting-Employee-Attrition-Using-Machine-Learning

## Business Understanding
Sebuah perusahaan teknologi menghadapi **masalah turnover karyawan** yang tinggi, terutama dalam tim pengembang perangkat lunak. **Tingkat turnover yang tinggi ini menyebabkan peningkatan biaya rekrutmen, penundaan proyek, dan penurunan produktivitas**. Perusahaan ingin memahami faktor-faktor yang menyebabkan karyawan keluar dan mengambil langkah preventif untuk meningkatkan retensi.

### Problem:
Turnover karyawan yang tinggi menyebabkan peningkatan biaya rekrutmen, penundaan proyek, dan penurunan produktivitas.

### Objective:
Menganalisis alasan utama terjadinya turnover karyawan dan membangun model prediktif untuk mengidentifikasi karyawan dengan potensi resign yang tinggi.

### Goal:
Mengusulkan strategi preventif untuk mengurangi turnover karyawan.

## Dataset
Dataset ini berisi kolom-kolom berikut:

1. Username  
2. EnterpriseID  
3. StatusPernikahan  
4. JenisKelamin  
5. StatusKepegawaian  
6. Pekerjaan  
7. JenjangKarir  
8. PerformancePegawai  
9. AsalDaerah  
10. HiringPlatform  
11. SkorSurveyEngagement  
12. SkorKepuasanPegawai  
13. JumlahKeikutsertaanProjek  
14. JumlahKeterlambatanSebulanTerakhir  
15. JumlahKetidakhadiran  
16. NomorHP  
17. Email  
18. TingkatPendidikan  
19. PernahBekerja  
20. IkutProgramLOP  
21. AlasanResign  
22. TanggalLahir  
23. TanggalHiring  
24. TanggalPenilaianKaryawan  
25. TanggalResign  

## Data Cleansing

### Data Summary
- Total baris data: 287

### Imputation of Missing Values
- **Numerical Columns**: Missing value diisi menggunakan median.
- **Categorical Columns**: Missing value diisi menggunakan modus (nilai yang paling sering muncul).

### Column Removal
Kolom-kolom berikut dihapus:
- IkutProgramLOP
- Username
- PernahBekerja
- EnterpriseID
- NomorHP
- Email

### Data Correction
- **Kolom StatusPernikahan** dan **AlasanResign**: Kesalahan nilai diperbaiki dengan mengganti nilai yang salah menggunakan nilai yang paling sering muncul pada masing-masing kolom.

## Analytics
