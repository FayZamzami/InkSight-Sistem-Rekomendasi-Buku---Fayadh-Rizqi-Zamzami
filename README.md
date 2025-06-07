# InkSight-Sistem-Rekomendasi-Buku

## **Domain Proyek**

Di era digital saat ini, industri buku mengalami transformasi signifikan dengan meningkatnya akses terhadap buku digital dan platform membaca online. Berdasarkan laporan "Global eBook Market Report 2023", pasar buku digital global diproyeksikan mencapai USD 18.7 miliar pada tahun 2023, dengan tingkat pertumbuhan tahunan (CAGR) sebesar 4.9% dari 2023 hingga 2028. Pertumbuhan ini menciptakan tantangan baru dalam hal bagaimana membantu pembaca menemukan buku yang sesuai dengan preferensi mereka di tengah jutaan pilihan yang tersedia.

**Mengapa Masalah Ini Penting untuk Diselesaikan?**

- **Information Overload :** Ledakan konten digital telah menghasilkan lebih dari 4 juta buku yang diterbitkan setiap tahunnya secara global, menciptakan fenomena "paradox of choice" dimana pembaca justru kesulitan menentukan pilihan di tengah banyaknya opsi. Hal ini terbukti dari data yang menunjukkan 67% pembaca mengalami kesulitan menemukan buku berikutnya, dengan 45% dari mereka menghabiskan lebih dari 30 menit hanya untuk mencari buku yang sesuai minat mereka.
- **Damppak Ekonomi :** Ketidakmampuan pembaca dalam menemukan buku yang sesuai telah menciptakan dampak signifikan pada industri perbukuan, dimana 35% potensi pembelian buku gagal terjadi. Namun, platform yang telah mengimplementasikan sistem rekomendasi yang baik melaporkan peningkatan penjualan hingga 50%, dengan retensi pengguna platform membaca online meningkat hingga 40% berkat adanya rekomendasi yang personal.
- **Pengembangan Literasi :** Sistem rekomendasi yang efektif terbukti berperan penting dalam meningkatkan budaya literasi, dimana pembaca yang berhasil menemukan buku sesuai minat memiliki 73% kemungkinan lebih tinggi untuk membaca secara rutin. Hal ini diperkuat dengan data yang menunjukkan bahwa kualitas rekomendasi buku berkorelasi positif dengan peningkatan minat baca, serta personalisasi rekomendasi yang berhasil meningkatkan engagement pembaca hingga 58%.
**Hasil Riset Terkait:**



**Referensi :**
- https://www.researchgate.net/publication/360772285_An_Enhanced_Book_Recommendation_System_Using_Hybrid_Machine_Learning_Techniques

- https://esomar.org/uploads/attachments/clpv6d0qh09v2h53v27nptwra-esomar-global-market-research-2023-chapter-1.pdf


## **Business Understanding**

### Problem Statements

- Bagaimana mengembangkan sistem rekomendasi Neural Collaborative Filtering yang dapat memprediksi preferensi pembaca dengan akurat berdasarkan pola interaksi pengguna-buku untuk membantu pembaca menemukan buku yang sesuai dengan selera mereka?

- Bagaimana mengimplementasikan dan mengevaluasi model deep learning untuk sistem rekomendasi buku yang dapat memberikan prediksi rating yang akurat dan rekomendasi yang terpersonalisasi berdasarkan data interaksi historis pengguna?

- Bagaimana mengukur efektivitas sistem rekomendasi collaborative filtering melalui berbagai metrik evaluasi yang komprehensif untuk memastikan kualitas prediksi rating dan keberagaman rekomendasi yang dihasilkan?

### Goals

- Membangun model Neural Collaborative Filtering yang dapat memprediksi rating buku dengan tingkat akurasi minimal 85% (measured by accuracy within 0.5 stars) dan Mean Absolute Error (MAE) di bawah 0.3.

- Mengembangkan sistem rekomendasi yang dapat memberikan rekomendasi buku yang beragam dengan diversity score minimal 0.95 dan mampu menghasilkan rekomendasi personal berdasarkan embedding pengguna dan buku yang dipelajari oleh model.

- Mengimplementasikan framework evaluasi yang komprehensif meliputi offline evaluation (MAE, RMSE, accuracy metrics), recommendation quality evaluation (precision, recall, NDCG, diversity), dan qualitative evaluation untuk mengukur performa model secara menyeluruh.

### Solution statements

- Mengimplementasikan arsitektur Neural Collaborative Filtering menggunakan TensorFlow/Keras yang menggabungkan embedding layers untuk pengguna dan buku, diikuti dengan dense layers untuk mempelajari pola interaksi kompleks. Model ini akan dilatih menggunakan data interaksi pengguna-buku untuk memprediksi rating dengan akurasi tinggi dan menghasilkan rekomendasi yang terpersonalisasi.

- Membangun sistem preprocessing data yang komprehensif meliputi feature engineering, data cleaning, dan normalisasi rating untuk mempersiapkan dataset yang optimal untuk training model collaborative filtering. Sistem ini akan menangani encoding ID pengguna dan buku, serta normalisasi rating ke skala yang sesuai untuk output sigmoid.

## Data Understanding

Dalam proyek ini, saya menggunakan Goodreads Books Dataset yang berisi informasi komprehensif mengenai buku-buku beserta rating dari pembaca. Dataset ini diambil dari [Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks), yang merupakan kumpulan data dari platform Goodreads, salah satu platform review buku terbesar dengan lebih dari 90 juta pengguna.

Link Dataset : https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel

Dataset books.csv memiliki 11,127 entries dengan variabel sebagai berikut:
- bookID (int64): ID unik untuk mengidentifikasi setiap buku
- title (object): Judul buku
- authors (object): Nama penulis buku
- average_rating (float64): Rating rata-rata buku (skala 1-5)
- isbn (object): Nomor ISBN (International Standard Book Number) 10 digit
- isbn13 (object): Nomor ISBN 13 digit
- language_code (object): Kode bahasa buku (contoh: 'eng' untuk bahasa Inggris)
- num_pages (int64): Jumlah halaman buku
- ratings_count (int64): Jumlah rating yang diberikan untuk buku
- text_reviews_count (int64): Jumlah review teks yang ditulis untuk buku
- publication_date (object): Tanggal publikasi buku
- publisher (object): Nama penerbit buku

Dataset ini memiliki keunggulan karena sudah mencakup informasi rating agregat (average_rating) dan jumlah rating (ratings_count) dalam satu file, sehingga tidak memerlukan file terpisah untuk data rating dan pengguna. Hal ini memudahkan dalam pembuatan sistem rekomendasi karena data sudah terintegrasi dan memiliki informasi yang lengkap untuk implementasi content-based filtering.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

#### 1. LOAD DATASET
Pada bagian ini melakukan import library yang akan di gunakan dan melakukan read pada datasetnya dan melihat hed dan describ nya

**1. Informasi Dataset**
- Dataset memiliki 11,123 baris dan 12 kolom
- Semua kolom memiliki data lengkap (non-null)
- Memory usage dataset adalah 1.8+ MB

**2. Tipe Data pada Dataset**
- **Integer (int64)**:
  * bookID: ID unik buku
  * isbn13: Nomor ISBN 13 digit
  * num_pages: Jumlah halaman
  * ratings_count: Jumlah rating
  * text_reviews_count: Jumlah review teks
- **Float (float64)**:
  * average_rating: Rating rata-rata buku
- **Object**:
  * title: Judul buku
  * authors: Nama penulis
  * isbn: Nomor ISBN 10 digit
  * language_code: Kode bahasa
  * publication_date: Tanggal publikasi
  * publisher: Penerbit

**3. Analisis Statistik Deskriptif**
- **Count**: Semua kolom memiliki 11,123 entri, menunjukkan tidak ada missing values
- **Mean (Rata-rata)**:
  * average_rating: 3.93 (skala 1-5)
  * num_pages: 336.40 halaman
  * ratings_count: 17,942.85 rating per buku
  * text_reviews_count: 542.04 review per buku
- **Std (Standar Deviasi)**:
  * average_rating: 0.35, menunjukkan variasi rating yang relatif kecil
  * num_pages: 241.15, menunjukkan variasi yang cukup besar dalam jumlah halaman
- **Min dan Max**:
  * average_rating: min 1.0, max 5.0
  * num_pages: min 0, max 6,576 halaman
  * ratings_count: min 0, max 4,597,666 rating
  * text_reviews_count: min 0, max 94,265 review

**4. Sample Data**
Dari 20 baris pertama yang ditampilkan, terlihat beberapa pola:
- Banyak buku dari penulis terkenal seperti J.K. Rowling dan Douglas Adams
- Rating rata-rata berkisar antara 3.74 hingga 4.78
- Bahasa dominan adalah bahasa Inggris (eng)
- Jumlah halaman bervariasi dari 6 hingga 3,342
- Publisher termasuk penerbit besar seperti Scholastic dan Random House

Dataset ini menunjukkan koleksi buku yang komprehensif dengan informasi detail tentang rating dan review pembaca, yang sangat cocok untuk sistem rekomendasi buku.

#### 2. Check Missing Value dan Duplikasi

**Check Missing Value :**
   - Hasil menunjukkan bahwa tidak ada missing values (nilai 0) pada semua kolom
   - Ini berarti dataset sangat lengkap dan tidak memerlukan penanganan untuk missing values
   - Semua 11,123 baris data memiliki nilai yang valid untuk setiap kolomnya

**Check Duplikasi :**
   - Tidak ditemukan data duplikat dalam dataset
   - Setiap baris data adalah unik
   - Hal ini menunjukkan kualitas data yang baik karena tidak ada redundansi

#### 3. Boxplot Visualization
1. Memilih kolom-kolom numerik yang relevan untuk dianalisis

2. **Pengaturan Plot**
- sns.set(style="whitegrid"): Mengatur tampilan plot dengan grid putih
- plt.figure(figsize=(10, 4)): Membuat figure baru dengan ukuran 10x4 inci
**Pembuatan Box Plot**
- Menggunakan skala logaritmik untuk 'ratings_count' dan 'text_reviews_count' karena memiliki range nilai yang besar
- Box plot menunjukkan:
    - Garis tengah: median
    - Kotak: kuartil 1 (Q1) hingga kuartil 3 (Q3)
    - Whisker: range data normal
    - Titik: outlier
**Pengaturan Tampilan**
- Title: Judul plot dengan format yang rapi
- xlabel: Label sumbu x
- Grid: Menambahkan grid pada sumbu x
- Ticks: Mengatur ukuran font pada sumbu x
- Layout: Mengatur agar plot tidak tumpang tindih

3. **Hasil Visualisasi :**

1. Boxplot of Average Rating
- Mayoritas buku memiliki rating antara 3.7 (Q1) hingga 4.1 (Q3)
- Median rating sekitar 3.96
- Terdapat outlier di bagian bawah (rating 0-2)
- Rating maksimum adalah 5.0 dan minimum 0.0
- Distribusi cenderung miring ke kiri (left-skewed), menunjukkan kebanyakan buku mendapat rating yang cukup tinggi
 2. Boxplot of Num Pages
- Sebagian besar buku memiliki jumlah halaman antara 192 (Q1) hingga 416 (Q3) halaman
- Median jumlah halaman sekitar 299
- Terdapat banyak outlier di atas 1000 halaman
- Beberapa buku sangat tebal mencapai 6,576 halaman (outlier ekstrem)
- Distribusi miring ke kanan (right-skewed)
3. Boxplot of Ratings Count (Skala Logaritmik)
- Median jumlah rating sekitar 745 ratings
- Range interquartil antara 100 hingga 1000 ratings
- Terdapat outlier ekstrem dengan lebih dari 100,000 ratings
- Nilai maksimum mencapai 4,597,666 ratings
- Distribusi sangat miring ke kanan (heavily right-skewed)
4. Boxplot of Text Reviews Count (Skala Logaritmik)
- Median jumlah review sekitar 47 reviews
- Mayoritas buku memiliki antara 10-1000 reviews
- Beberapa buku populer memiliki lebih dari 10,000 reviews
- Nilai maksimum mencapai 94,265 reviews
- Distribusi sangat miring ke kanan (heavily right-skewed)


4. **Analisis Statistik Deskriptif**

Text Reviews Count :
- Menunjukkan distribusi jumlah review teks
- Menggunakan skala log karena range yang besar
- Mengindikasikan tingkat engagement pembaca

# 📊 Exploratory Data Analysis (EDA) - Univariate Analysis

Bagian ini menjelaskan analisis univariat yang dilakukan untuk memahami karakteristik individual dari setiap variabel dalam dataset buku Goodreads.

## 📈 1. Analisis Distribusi Rating Buku

### 🔧 Implementasi

```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df_books, x='average_rating', bins=30, color='skyblue')
plt.title('Distribusi Rating Buku', fontsize=12)
plt.xlabel('Rating')
plt.ylabel('Jumlah Buku')

plt.subplot(1, 2, 2)
sns.kdeplot(data=df_books, x='average_rating', fill=True, color='skyblue')
plt.title('Density Plot Rating Buku', fontsize=12)
plt.xlabel('Rating')
plt.ylabel('Density')
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Figure size**: `(12, 5)` untuk visualisasi optimal
- **Bins**: `30` untuk detail distribusi yang memadai
- **Subplot layout**: `(1, 2)` untuk perbandingan histogram dan density plot
- **Color scheme**: `skyblue` untuk konsistensi visual

### 📊 Hasil Analisis

- **Rata-rata rating**: 3.93 dari skala 5.0
- **Standar deviasi**: 0.35 (variasi rendah)
- **Distribusi**: Mayoritas buku memiliki rating 3.5-4.5
- **Puncak distribusi**: Sekitar rating 4.0
- **Insight**: Dataset menunjukkan bias positif dalam penilaian buku

***

## 🌍 2. Analisis Distribusi Bahasa

### 🔧 Implementasi

```python
plt.figure(figsize=(12, 5))
language_counts = df_books['language_code'].value_counts().head(10)
sns.barplot(x=language_counts.index, y=language_counts.values, color='skyblue')
plt.title('10 Bahasa Paling Umum', fontsize=12)
plt.xlabel('Kode Bahasa')
plt.ylabel('Jumlah Buku')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Top languages**: `head(10)` untuk fokus pada bahasa dominan
- **Rotation**: `45°` untuk keterbacaan label
- **Bar orientation**: Vertikal untuk perbandingan mudah

### 📊 Hasil Analisis

| Bahasa | Jumlah Buku | Persentase |
| --- | --- | --- |
| English (eng) | 8,908 | 80.1% |
| English-US (en-US) | 1,408 | 12.7% |
| Spanish (spa) | 218 | 2.0% |
| English-GB (en-GB) | 214 | 1.9% |
| French (fre) | 144 | 1.3% |

**Key Insights:**

- 🔹 Dominasi bahasa Inggris: **94.7%** total
- 🔹 Representasi bahasa non-Inggris terbatas
- 🔹 Dataset optimal untuk rekomendasi buku berbahasa Inggris

***

## 📖 3. Analisis Distribusi Jumlah Halaman

### 🔧 Implementasi

```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=df_books, x='num_pages', bins=50, color='skyblue')
plt.title('Distribusi Jumlah Halaman', fontsize=12)
plt.xlabel('Jumlah Halaman')
plt.ylabel('Jumlah Buku')

plt.subplot(1, 2, 2)
sns.histplot(data=df_books[df_books['num_pages'] < 1000],
             x='num_pages', bins=50, color='skyblue')
plt.title('Distribusi Jumlah Halaman (<1000)', fontsize=12)
plt.xlabel('Jumlah Halaman')
plt.ylabel('Jumlah Buku')
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Dual view**: Semua data vs. filtered (<1000 halaman)
- **Bins**: `50` untuk detail distribusi
- **Filter threshold**: `1000` halaman untuk menghilangkan outlier

### 📊 Hasil Analisis

- **Range mayoritas**: 200-600 halaman
- **Puncak distribusi**: 300-400 halaman
- **Distribusi**: Right-skewed dengan long tail
- **Outlier**: Beberapa buku >1000 halaman
- **Insight**: Mayoritas buku memiliki panjang "standar"

***

## 🏢 4. Analisis Distribusi Penerbit

### 🔧 Implementasi

```python
plt.figure(figsize=(12, 5))
publisher_counts = df_books['publisher'].value_counts().head(10)
sns.barplot(x=publisher_counts.values, y=publisher_counts.index, color='skyblue')
plt.title('10 Penerbit Terbesar', fontsize=12)
plt.xlabel('Jumlah Buku')
plt.ylabel('Penerbit')
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Horizontal layout**: Untuk keterbacaan nama penerbit
- **Top publishers**: `head(10)` untuk fokus pada dominan
- **Value-based sorting**: Otomatis melalui `value_counts()`

### 📊 Hasil Analisis

| Rank | Penerbit | Jumlah Buku |
| --- | --- | --- |
| 1 | Vintage | 318 |
| 2 | Penguin Books | 261 |
| 3 | Penguin Classics | 184 |
| 4 | Mariner Books | 150 |
| 5 | Ballantine Books | 144 |

**Key Insights:**

- 🔹 **Penguin Group** dominan (445 buku total)
- 🔹 Konsentrasi pada penerbit besar
- 🔹 Distribusi relatif merata untuk posisi 4-9

***

## 📅 5. Analisis Distribusi Tahun Publikasi

### 🔧 Implementasi

```python
def extract_year(date_str):
    try:
        return pd.to_datetime(date_str, format='mixed').year
    except:
        try:
            year = ''.join(filter(str.isdigit, date_str))[-4:]
            return int(year) if len(year) == 4 else None
        except:
            return None

df_books['publication_year'] = df_books['publication_date'].apply(extract_year)

plt.figure(figsize=(12, 5))
sns.histplot(data=df_books, x='publication_year', bins=50, color='skyblue')
plt.title('Distribusi Tahun Publikasi', fontsize=12)
plt.xlabel('Tahun')
plt.ylabel('Jumlah Buku')
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Date parsing**: Robust extraction dengan fallback
- **Bins**: `50` untuk detail temporal
- **Range**: 1900-2020 (120 tahun)

### 📊 Hasil Analisis

- **Periode dominan**: 1996-2020 (75% data)
- **Puncak publikasi**: 2000-2005
- **Median tahun**: 2003
- **Range**: 1900-2020
- **Insight**: Dataset bias pada buku modern/kontemporer

***

## 📋 6. Ringkasan Statistik Deskriptif

### 🔧 Implementasi

```python
print("Statistik Deskriptif Rating:")
print(df_books['average_rating'].describe())

print("Top 5 Bahasa:")
print(df_books['language_code'].value_counts().head())

print("Top 5 Penerbit:")
print(df_books['publisher'].value_counts().head())
```

### 📊 Summary Statistics

#### Rating Distribution

- **Count**: 11,123 buku
- **Mean**: 3.93 ± 0.35
- **Range**: 0.0 - 5.0
- **IQR**: 3.77 - 4.14

#### Language Distribution

- **English variants**: 94.7%
- **Other languages**: 5.3%
- **Total languages**: 25+ kode bahasa

#### Publisher Concentration

- **Top 5 publishers**: 1,057 buku (9.5%)
- **Penguin Group**: 445 buku (4.0%)
- **Market fragmentation**: Moderate

***

## 🎯 Key Findings & Implications

### ✅ Dataset Strengths

1. **High-quality ratings**: Konsisten dan bias positif
2. **Comprehensive coverage**: 11K+ buku dari penerbit ternama
3. **Modern focus**: Representasi baik untuk buku kontemporer
4. **Language consistency**: Dominasi bahasa Inggris memudahkan analisis

### ⚠️ Potential Limitations

1. **Language bias**: Terbatas untuk analisis multilingual
2. **Temporal bias**: Kurang representatif untuk buku klasik
3. **Publisher concentration**: Dominasi penerbit besar
4. **Rating inflation**: Bias positif dalam penilaian



# 📊 Exploratory Data Analysis (EDA) - Multivariate Analysis

Bagian ini menjelaskan analisis multivariat untuk memahami hubungan dan pola antar variabel dalam dataset buku Goodreads.

## 🔗 1. Analisis Hubungan Rating vs Jumlah Halaman

### 🔧 Implementasi

```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_books, x='num_pages', y='average_rating', alpha=0.5, color='skyblue')
plt.title('Hubungan Rating dan Jumlah Halaman', fontsize=12)
plt.xlabel('Jumlah Halaman')
plt.ylabel('Rating')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_books[df_books['num_pages'] < 1000],
                x='num_pages', y='average_rating',
                alpha=0.5, color='skyblue')
plt.title('Hubungan Rating dan Jumlah Halaman (<1000)', fontsize=12)
plt.xlabel('Jumlah Halaman')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Figure layout**: `(1, 2)` untuk perbandingan dual-view
- **Alpha transparency**: `0.5` untuk visualisasi overlapping points
- **Outlier filtering**: `<1000` halaman untuk analisis detail
- **Color consistency**: `skyblue` untuk kohesi visual

### 📊 Hasil Analisis

#### Plot Kiri (Semua Data)

- **Range halaman**: 0-6,000 halaman
- **Konsentrasi data**: 0-1,000 halaman (mayoritas)
- **Rating range**: 0-5 dengan dominasi 3.5-4.5
- **Outlier**: Beberapa buku >3,000 halaman

#### Plot Kanan (Filtered <1000)

- **Focus range**: 200-600 halaman (mayoritas)
- **Rating consistency**: Stabil di 3.5-4.5
- **Pattern**: Tidak ada tren linear yang jelas

### 🎯 Key Insights

- ❌ **Korelasi lemah**: Jumlah halaman tidak mempengaruhi rating secara signifikan
- ✅ **Bias positif**: Mayoritas buku mendapat rating >3.5 terlepas dari panjangnya
- ⚠️ **Anomali data**: Rating 0 perlu investigasi lebih lanjut

***

## 📈 2. Analisis Hubungan Rating vs Popularitas

### 🔧 Implementasi

```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_books,
                x='ratings_count', y='average_rating',
                alpha=0.5, color='skyblue')
plt.title('Hubungan Rating dan Jumlah Ratings', fontsize=12)
plt.xlabel('Jumlah Ratings')
plt.ylabel('Rating')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_books,
                x='ratings_count', y='average_rating',
                alpha=0.5, color='skyblue')
plt.xscale('log')
plt.title('Hubungan Rating dan Jumlah Ratings (Log Scale)', fontsize=12)
plt.xlabel('Jumlah Ratings (Log)')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Dual scaling**: Normal vs logarithmic untuk data dengan range lebar
- **Log scale range**: 10⁰ hingga 10⁶ (1 - 1,000,000 ratings)
- **Transparency**: `alpha=0.5` untuk density visualization
- **Data range**: 0 - 4,597,666 ratings

### 📊 Hasil Analisis

#### Plot Kiri (Skala Normal)

- **Konsentrasi**: Mayoritas buku di bagian kiri (rating rendah)
- **Visibility issue**: Sulit melihat pola karena data clustering
- **Outlier**: Beberapa buku dengan >2 juta ratings

#### Plot Kanan (Skala Log)

- **Pattern revelation**: "Triangular" distribution terlihat jelas
- **Wisdom of crowds**: Rating konvergen ke 3.5-4.5 pada popularitas tinggi
- **Variance reduction**: Variasi rating menurun seiring bertambahnya jumlah rating

### 🎯 Key Insights

- 📊 **Wisdom of Crowds Effect**: Buku populer memiliki rating yang lebih stabil (3.5-4.5)
- 📉 **Variance Pattern**: Rating variance menurun seiring bertambahnya jumlah rating
- 🎯 **Quality Filter**: Buku berkualitas rendah jarang mendapat banyak rating
- ⚖️ **Reliability**: Rating dari buku dengan banyak review lebih reliable

***

## 🌍 3. Analisis Rating Berdasarkan Bahasa

### 🔧 Implementasi

```python
plt.figure(figsize=(12, 5))
top_languages = df_books['language_code'].value_counts().head(10).index
sns.boxplot(data=df_books[df_books['language_code'].isin(top_languages)],
            x='language_code', y='average_rating',
            color='skyblue')
plt.title('Distribusi Rating berdasarkan Bahasa', fontsize=12)
plt.xlabel('Kode Bahasa')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Language selection**: Top 10 bahasa berdasarkan frekuensi
- **Box plot elements**: Median, quartiles, outliers, whiskers
- **Label rotation**: `45°` untuk keterbacaan optimal
- **Filter strategy**: `isin()` untuk multiple language selection

### 📊 Hasil Analisis

#### Kelompok Bahasa Inggris

| Bahasa | Median Rating | IQR Range | Outliers |
| --- | --- | --- | --- |
| eng | ~4.0 | 3.7-4.2 | Banyak (0-2) |
| en-US | ~4.0 | 3.7-4.2 | Sedang |
| en-GB | ~4.0 | 3.7-4.2 | Sedang |

#### Bahasa Eropa

- **Karakteristik**: Distribusi lebih sempit dan stabil
- **Median**: Konsisten ~4.0
- **Outlier**: Lebih sedikit dibanding bahasa Inggris

#### Bahasa Asia

- **jpn & zho**: Median tertinggi (>4.2)
- **Konsistensi**: Range sangat sempit, outlier minimal
- **Sample size**: Relatif kecil tapi konsisten

### 🎯 Key Insights

- 🔄 **Bias Positif Universal**: Semua bahasa memiliki median >3.8
- 📏 **Sample Size Effect**: Bahasa dengan sampel kecil lebih konsisten
- 🌏 **Cultural Differences**: Bahasa Asia menunjukkan rating lebih tinggi
- ⚖️ **Normalization Need**: Diperlukan normalisasi berdasarkan bahasa untuk sistem rekomendasi

***

## 🔥 4. Analisis Matriks Korelasi

### 🔧 Implementasi

```python
numeric_columns = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']
correlation_matrix = df_books[numeric_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f')
plt.title('Matriks Korelasi', fontsize=12)
plt.tight_layout()
plt.show()
```

### ⚙️ Parameter Kunci

- **Correlation method**: Pearson correlation coefficient
- **Heatmap colormap**: `coolwarm` (red=positive, blue=negative, white=zero)
- **Annotation**: `annot=True` untuk nilai numerik
- **Format**: `.2f` untuk 2 decimal places
- **Center**: `0` untuk color balance

### 📊 Hasil Analisis

#### Matriks Korelasi

| Variables | average_rating | num_pages | ratings_count | text_reviews_count |
| --- | --- | --- | --- | --- |
| **average_rating** | 1.00 | 0.15 | 0.04 | 0.03 |
| **num_pages** | 0.15 | 1.00 | 0.03 | 0.04 |
| **ratings_count** | 0.04 | 0.03 | 1.00 | 0.87 |
| **text_reviews_count** | 0.03 | 0.04 | 0.87 | 1.00 |

### 🎯 Key Insights

#### Korelasi Kuat (>0.7)

- 🔗 **Ratings ↔ Reviews**: 0.87 (sangat kuat)
- Buku yang banyak dirating juga banyak direview
- Menunjukkan engagement consistency

#### Korelasi Lemah (<0.2)

- 📖 **Rating ↔ Pages**: 0.15 (lemah positif)
- 📊 **Rating ↔ Popularity**: 0.04 (sangat lemah)
- 📏 **Pages ↔ Popularity**: 0.03 (sangat lemah)

#### Implikasi untuk Sistem Rekomendasi

- ✅ **Quality Independence**: Rating tidak bergantung pada popularitas atau panjang buku
- 🎯 **Engagement Indicator**: Ratings count dan reviews count saling mendukung
- 🔍 **Feature Selection**: Variabel relatif independen, cocok untuk model ML

***

## 📋 5. Statistik Deskriptif per Bahasa

### 🔧 Implementasi

```python
language_stats = df_books.groupby('language_code')['average_rating'].agg(['count', 'mean', 'std']).round(3)
print(language_stats.sort_values('count', ascending=False).head(10))
```

### ⚙️ Parameter Kunci

- **Grouping**: `groupby('language_code')` untuk analisis per bahasa
- **Aggregation**: `count`, `mean`, `std` untuk statistik komprehensif
- **Sorting**: Berdasarkan jumlah buku (descending)
- **Precision**: `round(3)` untuk 3 decimal places

### 📊 Hasil Analisis

| Language | Count | Mean Rating | Std Dev | Interpretation |
| --- | --- | --- | --- | --- |
| **eng** | 8,908 | 3.934 | 0.359 | Dominan, variasi sedang |
| **en-US** | 1,408 | 3.915 | 0.289 | Konsisten, variasi rendah |
| **spa** | 218 | 3.929 | 0.270 | Stabil, sampel sedang |
| **en-GB** | 214 | 3.943 | 0.264 | Konsisten, kualitas tinggi |
| **fre** | 144 | 3.915 | 0.283 | Stabil |
| **jpn** | 46 | 4.269 | 0.165 | Rating tertinggi, sangat konsisten |
| **zho** | 14 | 4.456 | 0.218 | Rating tertinggi, sampel kecil |
| **grc** | 11 | 3.707 | 1.336 | Variasi ekstrem, sampel kecil |

### 🎯 Key Insights

#### Pattern Analysis

- 📊 **Sample Size Effect**: Bahasa dengan sampel besar memiliki std dev lebih tinggi
- 🌏 **Cultural Rating Bias**: Bahasa Asia menunjukkan rating lebih tinggi
- ⚖️ **Consistency Paradox**: Sampel kecil bisa sangat konsisten atau sangat bervariasi
- 🎯 **Quality Threshold**: Semua bahasa memiliki mean rating >3.7

#### Implikasi Bisnis

- 🔍 **Market Focus**: 94.7% konten berbahasa Inggris
- 🌐 **Expansion Opportunity**: Potensi growth di bahasa Asia
- ⚖️ **Normalization Strategy**: Diperlukan adjustment berdasarkan bahasa
- 📈 **Quality Assurance**: Konsistensi rating tinggi across languages

***

## 🎯 Ringkasan Temuan Multivariate

### ✅ Pola yang Teridentifikasi

1. **Independence of Quality**

- Rating buku independen dari panjang dan popularitas
- Kualitas tidak berkorelasi dengan metrics fisik

2. **Wisdom of Crowds**

- Buku populer memiliki rating lebih stabil
- Variance menurun seiring bertambahnya jumlah rating

3. **Language Bias**

- Bahasa Asia menunjukkan rating sistematis lebih tinggi
- Diperlukan normalisasi cross-language

4. **Engagement Consistency**

- Strong correlation (0.87) antara ratings dan reviews
- Menunjukkan user engagement yang konsisten

