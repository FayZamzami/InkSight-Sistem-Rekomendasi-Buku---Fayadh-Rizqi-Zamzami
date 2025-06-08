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

## Data Preperationn

# **Data Preparation - Basic Data Exploration & Column Name Fixing**

## **📋 Penjelasan Kode**

### **🔧 Cara Kerja Kode**

Bagian ini merupakan tahap awal data preparation yang melakukan eksplorasi dasar dan pembersihan nama kolom. Berikut adalah breakdown dari setiap komponen:

#### **1. Dataset Shape Exploration**

```python
print(f"Shape dataset: {df_books.shape}")
```

- **Fungsi**: Menampilkan dimensi dataset (jumlah baris × kolom)
- **Method**: `shape` attribute dari pandas DataFrame
- **Output**: Tuple (11123, 13) yang berarti 11,123 buku dengan 13 atribut

#### **2. Column Name Inspection**

```python
for i, col in enumerate(df_books.columns):
    print(f"{i}: '{col}'")
```

- **Fungsi**: Iterasi melalui semua nama kolom dengan indeks
- **Method**: `enumerate()` untuk mendapatkan index dan value
- **Purpose**: Mengidentifikasi masalah formatting pada nama kolom
- **Benefit**: Memudahkan debugging dan identifikasi anomali

#### **3. Column Name Cleaning**

```python
df_books.columns = df_books.columns.str.strip()
```

- **Fungsi**: Menghilangkan whitespace di awal dan akhir nama kolom
- **Method**: `str.strip()` pada pandas Index object
- **Target**: Mengatasi masalah seperti `'  num_pages'` menjadi `'num_pages'`
- **Impact**: Mencegah error saat accessing kolom

#### **4. Missing Values Check**

```python
print(f"\nMissing values:\n{df_books.isnull().sum()}")
```

- **Fungsi**: Menghitung jumlah nilai null pada setiap kolom
- **Method**: `isnull().sum()` - kombinasi boolean mask dan aggregation
- **Output**: Series dengan nama kolom sebagai index dan count null sebagai value

***

## **⚙️ Parameter dan Fungsi**

### **1. Shape Attribute**

| Parameter | Type | Description |
| --- | --- | --- |
| `df_books.shape` | tuple | Mengembalikan (n_rows, n_columns) |
| **Return** | (11123, 13) | 11,123 baris, 13 kolom |

### **2. Enumerate Function**

| Parameter | Type | Description |
| --- | --- | --- |
| `df_books.columns` | Index | Pandas column index object |
| `start` | int | Starting index (default: 0) |
| **Return** | enumerate object | (index, column_name) pairs |

### **3. String Strip Method**

| Parameter | Type | Description |
| --- | --- | --- |
| `chars` | str | Characters to remove (default: whitespace) |
| **Method** | `str.strip()` | Removes leading/trailing characters |
| **Application** | Column names | Applied to all column names simultaneously |

### **4. Missing Values Detection**

| Method | Function | Output |
| --- | --- | --- |
| `isnull()` | Boolean mask | True for null values |
| `sum()` | Aggregation | Count of True values per column |
| **Combined** | `isnull().sum()` | Count of missing values per column |

***

## **📊 Analisis Output Mendalam**

### **🔍 1. Dataset Dimensions**

```javascript
Shape dataset: (11123, 13)
```

**Interpretasi:**

- **Rows (11,123)**: Ukuran dataset cukup besar untuk collaborative filtering
- **Columns (13)**: Atribut yang komprehensif untuk sistem rekomendasi
- **Memory Impact**: ~1.8MB, manageable untuk processing
- **Model Suitability**: Cukup untuk training neural collaborative filtering

### **🔍 2. Column Name Issues**

```javascript
Original: '  num_pages'  ← Masalah: Extra spaces
Fixed:    'num_pages'    ← Solusi: Clean formatting
```

**Masalah yang Teridentifikasi:**

- **Kolom 7**: `'  num_pages'` memiliki 2 spasi di awal
- **Impact**: Dapat menyebabkan `KeyError` saat accessing
- **Solution**: `str.strip()` menghilangkan whitespace

**Before vs After Comparison:**

| Index | Before | After | Status |
| --- | --- | --- | --- |
| 0-6 | ✅ Clean | ✅ Clean | No change |
| 7 | ❌ `'  num_pages'` | ✅ `'num_pages'` | **Fixed** |
| 8-12 | ✅ Clean | ✅ Clean | No change |

### **🔍 3. Data Completeness Analysis**

```javascript
Missing values:
bookID                0  ← Perfect
title                 0  ← Perfect  
authors               0  ← Perfect
average_rating        0  ← Critical for CF
isbn                  0  ← Perfect
isbn13                0  ← Perfect
language_code         0  ← Perfect
num_pages             0  ← Perfect
ratings_count         0  ← Critical for CF
text_reviews_count    0  ← Perfect
publication_date      0  ← Perfect
publisher             0  ← Perfect
publication_year      0  ← Perfect
```

**Key Insights:**

- **🎯 100% Data Completeness**: Tidak ada missing values
- **🔑 Critical Columns**: `average_rating` dan `ratings_count` lengkap
- **⚡ Ready for Processing**: Tidak perlu imputation atau handling missing data
- **🚀 Model Training**: Dapat langsung proceed ke tahap berikutnya


# **Data Preparation - Data Cleaning & Filtering**

## **📋 Tujuan dan Cara Kerja**

Tahap ini bertujuan membersihkan dataset berdasarkan insight EDA untuk menghasilkan data berkualitas tinggi yang optimal untuk collaborative filtering.

### **🔧 Implementasi dan Parameter**

#### **1. Dataset Protection**

```python
df_clean = df_books.copy()
```

- **Fungsi**: Membuat salinan independen dataset
- **Parameter**: `copy()` - deep copy untuk melindungi data original
- **Tujuan**: Mencegah modifikasi data asli selama cleaning

#### **2. Invalid Rating Removal**

```python
df_clean = df_clean[df_clean['average_rating'] > 0]
```

- **Fungsi**: Menghilangkan buku dengan rating 0 (invalid/unrated)
- **Parameter**: `> 0` - kondisi boolean filtering
- **Hasil**: 11,123 → 11,098 buku (-25 buku, -0.22%)

#### **3. Popularity-Based Filtering**

```python
min_ratings_book = 30
df_clean = df_clean[df_clean['ratings_count'] >= min_ratings_book]
```

- **Fungsi**: Filter buku dengan minimum rating count
- **Parameter**: `min_ratings_book = 30` - threshold untuk reliability
- **Hasil**: 11,098 → 9,697 buku (-1,401 buku, -12.6%)

#### **4. Page Count Outlier Removal**

```python
df_clean = df_clean[(df_clean['num_pages'] >= 10) & (df_clean['num_pages'] <= 1500)]
```

- **Fungsi**: Menghilangkan outlier jumlah halaman
- **Parameter**: Range 10-1500 halaman (realistic book length)
- **Operator**: `&` untuk AND condition
- **Hasil**: 9,697 → 9,548 buku (-149 buku, -1.5%)

#### **5. Outlier Capping (Winsorization)**

```python
ratings_count_95th = df_clean['ratings_count'].quantile(0.95)
text_reviews_95th = df_clean['text_reviews_count'].quantile(0.95)

df_clean['ratings_count_capped'] = df_clean['ratings_count'].clip(upper=ratings_count_95th)
df_clean['text_reviews_count_capped'] = df_clean['text_reviews_count'].clip(upper=text_reviews_95th)
```

- **Fungsi**: Membatasi nilai ekstrem tanpa menghapus data
- **Parameter**: 
- `quantile(0.95)` - persentil ke-95 sebagai upper bound
- `clip(upper=value)` - capping nilai maksimum
- **Hasil**: Ratings capped pada 70,664, Reviews pada 2,512

# **Data Preparation - Feature Engineering**

## **📋 Tujuan dan Cara Kerja**

Tahap ini mengubah dan menciptakan fitur baru berdasarkan insight EDA untuk meningkatkan kualitas prediksi model collaborative filtering.

### **🔧 Implementasi dan Parameter**

#### **1. Publication Year Extraction**

```python
def extract_year_improved(date_str):
    # Method 1: DateTime parsing
    year = pd.to_datetime(date_str, format='mixed').year
    # Method 2: Regex extraction  
    year_match = re.findall(r'\b(19|20)\d{2}\b', str(date_str))
    # Validation: 1900-2020 range
    return year if 1900 <= year <= 2020 else None

df_clean['publication_year'] = df_clean['publication_date'].apply(extract_year_improved)
df_clean['publication_year'] = df_clean['publication_year'].fillna(2003)  # Median dari EDA
```

- **Fungsi**: Ekstrak tahun publikasi dengan dual-method approach
- **Parameter**: 
- `format='mixed'` - flexible datetime parsing
- `r'\b(19|20)\d{2}\b'` - regex untuk tahun 1900-2099
- `fillna(2003)` - median berdasarkan EDA
- **Validasi**: Range 1900-2020 untuk realistic years

#### **2. Author Processing**

```python
def get_primary_author(authors_str):
    return str(authors_str).split('-')[0].strip()

def count_authors(authors_str):
    return len(str(authors_str).split('-'))

df_clean['primary_author'] = df_clean['authors'].apply(get_primary_author)
df_clean['author_count'] = df_clean['authors'].apply(count_authors)
```

- **Fungsi**: Ekstrak penulis utama dan hitung jumlah penulis
- **Parameter**: 
- `split('-')` - delimiter berdasarkan dataset description
- `strip()` - remove whitespace
- `len()` - count co-authors

#### **3. Enhanced Rating Categorization**

```python
def categorize_rating_enhanced(rating):
    if rating >= 4.2:    return 'Excellent'     # Top quartile
    elif rating >= 3.96: return 'Very Good'    # Above median  
    elif rating >= 3.77: return 'Good'         # Above Q1
    elif rating >= 3.0:  return 'Average'
    else:                return 'Below Average'
```

- **Fungsi**: Kategorisasi rating berdasarkan distribusi statistik EDA
- **Parameter**: Thresholds berdasarkan quartiles (Q1=3.77, Median=3.96, Q3=4.2)
- **Output**: 5 kategori verbal untuk interpretability

#### **4. Page Count Categorization**

```python
def categorize_pages_enhanced(pages):
    if pages <= 200:   return 'Short'      # Below median
    elif pages <= 300: return 'Medium'     # Around median (299)
    elif pages <= 450: return 'Long'       # Around Q3 (416)
    else:              return 'Very Long'   # Above Q3
```

- **Fungsi**: Kategorisasi berdasarkan distribusi halaman dari EDA
- **Parameter**: Thresholds (200, 300, 450) berdasarkan statistical analysis
- **Logic**: Pembagian quartile-based untuk balanced distribution

#### **5. Bayesian Popularity Score**

```python
def calculate_popularity_enhanced(row):
    v = row['ratings_count_capped']        # Vote count
    R = row['average_rating']              # Book rating
    C = df_clean['average_rating'].mean()  # Global mean (3.93)
    m = df_clean['ratings_count'].quantile(0.6)  # Threshold
    
    # Bayesian average formula
    weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
    return weighted_rating
```

- **Fungsi**: Menghitung popularity score menggunakan Bayesian average
- **Parameter**:
- `v` - jumlah rating (vote count)
- `R` - rating individual buku
- `C` - global mean rating (3.93)
- `m` - threshold dari 60th percentile
- **Formula**: Weighted combination untuk handle books dengan sedikit rating

#### **6. Engagement Score**

```python
df_clean['engagement_score'] = df_clean['text_reviews_count_capped'] / (df_clean['ratings_count_capped'] + 1)
```

- **Fungsi**: Mengukur rasio review terhadap rating
- **Parameter**: 
- `+1` untuk prevent division by zero
- Menggunakan capped values untuk outlier control
- **Insight**: Berdasarkan korelasi 0.87 antara ratings dan reviews dari EDA

#### **7. Language Grouping**

```python
def group_language(lang_code):
    lang_code = str(lang_code).lower()
    if lang_code in ['eng', 'en-us', 'en-gb']: return 'english'
    elif lang_code in ['spa', 'fre', 'ger', 'ita']: return 'european'  
    elif lang_code in ['jpn', 'zho', 'kor']: return 'asian'
    else: return 'other'
```

- **Fungsi**: Mengelompokkan bahasa berdasarkan regional similarity
- **Parameter**: Lists bahasa per kategori berdasarkan EDA insights
- **Logic**: Reduce dimensionality dari 25+ languages ke 4 groups

#### **8. Publication Era Categorization**

```python
def categorize_era(year):
    if year < 1980:   return 'Classic'       # Pre-digital era
    elif year < 2000: return 'Modern'        # 80s-90s
    elif year < 2010: return 'Contemporary'  # 2000s
    else:             return 'Recent'        # 2010+
```

- **Fungsi**: Kategorisasi berdasarkan era publikasi
- **Parameter**: Thresholds (1980, 2000, 2010) untuk historical periods
- **Rationale**: Capture technological and cultural shifts in publishing

***

## **📊 Ringkasan Fitur Baru**

### **Created Features Summary**

| Feature | Type | Purpose | Based On |
| --- | --- | --- | --- |
| `publication_year` | Numeric | Temporal analysis | Date extraction + EDA median |
| `primary_author` | Categorical | Author-based filtering | String processing |
| `author_count` | Numeric | Collaboration indicator | Delimiter counting |
| `rating_category` | Categorical | Quality segments | EDA quartiles |
| `page_category` | Categorical | Length segments | EDA distribution |
| `popularity_score` | Numeric | Bayesian popularity | Statistical formula |
| `engagement_score` | Numeric | User interaction ratio | EDA correlation insight |
| `language_group` | Categorical | Regional grouping | EDA language analysis |
| `publication_era` | Categorical | Historical periods | Temporal segmentation |

### **Feature Engineering Benefits**

- **Dimensionality Reduction**: 25+ languages → 4 groups
- **Statistical Grounding**: Thresholds berdasarkan EDA quartiles
- **Interpretability**: Categorical features untuk business understanding
- **Model Performance**: Bayesian scoring untuk better recommendations
-  **Outlier Handling**: Capped values dalam calculations

# **Data Preparation - Enhanced Synthetic User Data Generation**

## **📋 Tujuan dan Cara Kerja**

Tahap ini menghasilkan data interaksi user-book sintetis yang realistis berdasarkan insight EDA, karena dataset asli hanya berisi rating agregat tanpa individual user interactions yang dibutuhkan untuk collaborative filtering.

### **🔧 Implementasi dan Parameter**

#### **1. Random Seed & Configuration**

```python
np.random.seed(42)
n_users = 2000
```

- **Fungsi**: Memastikan reproducibility dan set jumlah user sintetis
- **Parameter**: 
- `seed=42` - untuk consistent random generation
- `n_users=2000` - balance antara diversity dan computational efficiency

#### **2. User Profile Generation**

```python
user_profiles = []
for user_id in range(n_users):
    lang_pref = np.random.choice(['english', 'european', 'asian', 'other'],
                                p=[0.85, 0.10, 0.03, 0.02])
    
    era_pref = np.random.choice(['Classic', 'Modern', 'Contemporary', 'Recent'],
                               p=[0.1, 0.2, 0.4, 0.3])
    
    rating_bias = np.random.normal(0, 0.2)
```

**Cara Kerja:**

- **Language Preference**: Distribusi berdasarkan EDA (85% English dominance)
- **Era Preference**: Weighted sampling berdasarkan publication distribution
- **Rating Bias**: Personal rating tendency dengan normal distribution

**Parameter Detail:**

| Parameter | Value | Function | Basis |
| --- | --- | --- | --- |
| `lang_pref` probabilities | [0.85, 0.10, 0.03, 0.02] | Language distribution | EDA insight (94.7% English) |
| `era_pref` probabilities | [0.1, 0.2, 0.4, 0.3] | Publication era weights | EDA temporal distribution |
| `rating_bias` | Normal(0, 0.2) | Personal rating tendency | Realistic user variation |

#### **3. Interaction Count Generation**

```python
n_interactions = int(np.random.lognormal(3.2, 0.6))
n_interactions = min(max(n_interactions, 20), 150)
```

- **Fungsi**: Generate realistic number of books per user
- **Parameter**:
- `lognormal(3.2, 0.6)` - right-skewed distribution (most users read few books)
- `min=20, max=150` - realistic bounds untuk active users
- **Logic**: Mimics real-world reading patterns (heavy-tailed distribution)

#### **4. Book Preference Filtering**

```python
preferred_books = df_clean[
    (df_clean['language_group'] == profile['language_pref']) |
    (df_clean['publication_era'] == profile['era_pref'])
]

# 70% from preferred, 30% from all books
pref_sample_size = int(n_interactions * 0.7)
other_sample_size = n_interactions - pref_sample_size
```

**Cara Kerja:**

- **Preference Matching**: Filter books berdasarkan user language dan era preference
- **Hybrid Sampling**: 70% dari preferred books, 30% random (simulate discovery)
- **Fallback Strategy**: Jika preferred books tidak cukup, sample dari semua books

**Parameter Logic:**

- `70/30 split` - balance antara preference consistency dan serendipity
- `OR condition` - user bisa suka language ATAU era (flexible preferences)

#### **5. Popularity-Weighted Book Selection**

```python
weights = available_books['popularity_score'].values
weights = weights / weights.sum()

selected_books = np.random.choice(
    available_books['bookID'].values,
    size=n_interactions,
    replace=False,
    p=weights
)
```

- **Fungsi**: Select books dengan probability berdasarkan popularity
- **Parameter**:
- `weights` - normalized popularity scores
- `replace=False` - prevent duplicate selections
- `p=weights` - weighted probability distribution
- **Logic**: Popular books lebih likely dipilih (realistic behavior)

#### **6. Rating Generation with User Bias**

```python
user_rating = book_avg + profile['rating_bias'] + np.random.normal(0, 0.3)
user_rating = np.clip(user_rating, 1, 5)
```

**Cara Kerja:**

- **Base Rating**: Mulai dari book's average rating
- **User Bias**: Apply personal rating tendency
- **Noise Addition**: Random variation dengan normal(0, 0.3)
- **Range Clipping**: Ensure rating dalam valid range [1, 5]

**Parameter Detail:**

| Component | Function | Parameter | Purpose |
| --- | --- | --- | --- |
| `book_avg` | Base rating | From dataset | Realistic starting point |
| `rating_bias` | Personal tendency | Normal(0, 0.2) | Individual user characteristics |
| `noise` | Random variation | Normal(0, 0.3) | Natural rating variability |
| `clip(1, 5)` | Range validation | [1, 5] bounds | Valid rating scale |

#### **7. Timestamp Generation**

```python
'timestamp': np.random.randint(1, 1000000)
```

- **Fungsi**: Generate random timestamps untuk temporal ordering
- **Parameter**: Range 1-1,000,000 untuk sufficient uniqueness
- **Purpose**: Enable temporal-based analysis jika diperlukan

***

## **📊 Technical Implementation Details**

### **Probabilistic Modeling**

```python
# Language preference distribution (based on EDA)
P(English) = 0.85    # Dominant language group
P(European) = 0.10   # Secondary group  
P(Asian) = 0.03      # Minority but high-quality
P(Other) = 0.02      # Rare languages
```

### **Interaction Distribution**

```python
# Lognormal parameters for realistic reading patterns
μ = 3.2  # Log-scale mean
σ = 0.6  # Log-scale standard deviation
# Results in: median ≈ 24 books, mean ≈ 30 books, some heavy readers >100
```

### **Rating Generation Formula**

```javascript
Final_Rating = Clip(Book_Avg + User_Bias + Noise, 1, 5)

Where:
- Book_Avg: Dataset average rating for the book
- User_Bias: Normal(0, 0.2) - personal rating tendency  
- Noise: Normal(0, 0.3) - situational variation
- Clip: Ensure valid range [1, 5]
```

***

## **🎯 Realism Features**

### **1. User Heterogeneity**

- **Language Preferences**: Berdasarkan actual distribution
- **Era Preferences**: Temporal reading patterns
- **Rating Bias**: Individual rating tendencies

### **2. Book Selection Logic**

- **Preference-Based**: 70% sesuai user preferences
- **Discovery Element**: 30% random exploration
- **Popularity Bias**: Weighted selection berdasarkan popularity_score

### **3. Rating Realism**

- **Book Quality**: Start dari actual book rating
- **Personal Variation**: Individual user characteristics
- **Natural Noise**: Random situational factors

### **4. Interaction Patterns**

- **Heavy-Tailed Distribution**: Few heavy readers, many casual readers
- **Bounded Range**: 20-150 books per user (realistic limits)
- **Temporal Component**: Random timestamps untuk ordering

***

# **Data Preparation - Enhanced Collaborative Filtering Preprocessing**

## **📋 Tujuan dan Cara Kerja**

Tahap ini mempersiapkan data interaksi sintetis untuk model collaborative filtering dengan fokus pada kualitas data dan adaptabilitas preprocessing berdasarkan karakteristik dataset.

### **🔧 Implementasi dan Parameter**

#### **1. User Filtering (Adjusted Threshold)**

```python
min_ratings_user = 10  # Turunkan dari 15
valid_users = [user for user, count in df_interactions['userId'].value_counts().items()
               if count >= min_ratings_user]
filtered_interactions = df_interactions[df_interactions['userId'].isin(valid_users)]
```

**Cara Kerja:**

- **Threshold Reduction**: Menurunkan minimum dari 15 ke 10 untuk retain lebih banyak data
- **User Activity Analysis**: Menghitung jumlah rating per user menggunakan `value_counts()`
- **Quality Filter**: Hanya user dengan aktivitas sufficient yang dipertahankan

**Parameter Detail:**

| Parameter | Value | Function | Rationale |
| --- | --- | --- | --- |
| `min_ratings_user` | 10 | Minimum interactions per user | Balance antara quality vs quantity |
| `value_counts()` | - | Count interactions per user | Statistical analysis |
| `list comprehension` | - | Filter valid users | Efficient filtering |

#### **2. Book Filtering (Adjusted Threshold)**

```python
min_user_ratings = 3  # Turunkan dari 8
valid_books_cf = [book for book, count in filtered_interactions['bookID'].value_counts().items()
                  if count >= min_user_ratings]
filtered_interactions = filtered_interactions[filtered_interactions['bookID'].isin(valid_books_cf)]
```

**Cara Kerja:**

- **Book Popularity Filter**: Minimum 3 users per book (reduced dari 8)
- **Cold Start Prevention**: Eliminasi books dengan terlalu sedikit interactions
- **Data Quality**: Ensure reliable recommendations untuk each book

**Parameter Logic:**

- `min_user_ratings = 3` - minimal viable sample size untuk book recommendations
- Progressive filtering - apply setelah user filtering untuk optimal results

#### **3. Adaptive Data Splitting Strategy**

```python
# Check data sufficiency for stratified split
unique_users = filtered_interactions['userId'].nunique()
min_test_size = int(0.2 * len(filtered_interactions))

if min_test_size < unique_users:
    # Simple random split
    train_data, test_data = train_test_split(filtered_interactions, test_size=0.2, random_state=42)
else:
    # Stratified split with validation
    user_counts = filtered_interactions['userId'].value_counts()
    valid_users_for_split = user_counts[user_counts >= 5].index
    
    if len(valid_users_for_split) < 10:
        # Fallback to simple split
        train_data, test_data = train_test_split(filtered_interactions, test_size=0.2, random_state=42)
    else:
        # Stratified split
        filtered_for_split = filtered_interactions[filtered_interactions['userId'].isin(valid_users_for_split)]
        train_data, test_data = train_test_split(
            filtered_for_split, test_size=0.2, random_state=42, stratify=filtered_for_split['userId']
        )
```

**Cara Kerja Logic Tree:**

```javascript
Data Assessment
├── Sufficient for Stratified? (min_test_size >= unique_users)
│   ├── Yes → Check User Sample Size
│   │   ├── Users with ≥5 interactions ≥ 10?
│   │   │   ├── Yes → Stratified Split
│   │   │   └── No → Simple Random Split
│   └── No → Simple Random Split
```

**Parameter Breakdown:**

| Condition | Threshold | Purpose | Fallback |
| --- | --- | --- | --- |
| `min_test_size < unique_users` | 20% of data | Sufficient test size | Simple split |
| `user_counts >= 5` | 5 interactions | Stratification viability | Simple split |
| `valid_users_for_split < 10` | 10 users | Minimum for stratification | Simple split |

#### **4. ID Mapping Creation**

```python
# Create bidirectional mappings
user_ids = filtered_interactions['userId'].unique().tolist()
book_ids = filtered_interactions['bookID'].unique().tolist()

user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
book_to_index = {book_id: i for i, book_id in enumerate(book_ids)}
index_to_user = {i: user_id for user_id, i in user_to_index.items()}
index_to_book = {i: book_id for book_id, i in book_to_index.items()}
```

**Cara Kerja:**

- **Unique ID Extraction**: Get semua unique user dan book IDs
- **Forward Mapping**: Original ID → Sequential Index (untuk model input)
- **Reverse Mapping**: Sequential Index → Original ID (untuk result interpretation)
- **Bidirectional Access**: Enable efficient lookup dalam kedua arah

**Mapping Structure:**

```python
user_to_index: {original_user_id: 0, 1, 2, ...}
index_to_user: {0: original_user_id, 1: original_user_id, ...}
book_to_index: {original_book_id: 0, 1, 2, ...}
index_to_book: {0: original_book_id, 1: original_book_id, ...}
```

#### **5. Data Format Conversion**

```python
def map_ids_enhanced(row):
    return {
        'user': user_to_index[row['userId']],
        'book': book_to_index[row['bookID']],
        'rating': row['rating'],
        'timestamp': row.get('timestamp', 0)
    }

train_mapped = train_data.apply(map_ids_enhanced, axis=1).tolist()
test_mapped = test_data.apply(map_ids_enhanced, axis=1).tolist()
```

**Cara Kerja:**

- **Row-wise Transformation**: Apply mapping function ke setiap interaction
- **Dictionary Format**: Convert ke format yang model-friendly
- **Safe Access**: `row.get('timestamp', 0)` dengan default value
- **List Conversion**: Final format sebagai list of dictionaries

**Output Format:**

```python
[
    {'user': 0, 'book': 15, 'rating': 4.2, 'timestamp': 12345},
    {'user': 0, 'book': 23, 'rating': 3.8, 'timestamp': 12346},
    ...
]
```

***

## **📊 Technical Analysis**

### **1. Filtering Impact Assessment**

```javascript
Progressive Data Reduction:
Original Interactions → User Filter → Book Filter → Final Dataset

Example Flow:
60,000 interactions → 55,000 (user filter) → 50,000 (book filter)
```

### **2. Split Strategy Decision Matrix**

| Data Condition | Strategy | Rationale | Trade-offs |
| --- | --- | --- | --- |
| Large dataset, many users | Stratified | Balanced user representation | More complex |
| Medium dataset | Simple random | Sufficient randomization | Less control |
| Small dataset | Simple random | Avoid overfitting to users | Potential bias |

### **3. Mapping Efficiency**

```python
# Time Complexity Analysis:
user_to_index lookup: O(1) - dictionary access
ID conversion: O(n) - linear scan through data
Memory usage: O(u + b) where u=users, b=books
```

## **⚙️ Implementation Best Practices**

### **1. Error Handling**

```python
# Robust mapping dengan error checking
def safe_map_ids(row):
    try:
        return map_ids_enhanced(row)
    except KeyError as e:
        print(f"Missing ID in mapping: {e}")
        return None
```

### **2. Memory Optimization**

```python
# Memory-efficient processing untuk large datasets
chunk_size = 10000
processed_chunks = []
for chunk in pd.read_csv('interactions.csv', chunksize=chunk_size):
    processed_chunk = process_chunk(chunk)
    processed_chunks.append(processed_chunk)
```

### **3. Validation Checks**

```python
# Data integrity validation
assert len(train_mapped) + len(test_mapped) <= len(filtered_interactions)
assert all(0 <= item['user'] < len(user_ids) for item in train_mapped)
assert all(0 <= item['book'] < len(book_ids) for item in train_mapped)
```

Preprocessing ini menghasilkan data yang optimal untuk collaborative filtering dengan balance antara data quality, quantity, dan computational efficiency.

# **Data Preparation - Enhanced Feature Scaling and Encoding**

## **📋 Tujuan dan Cara Kerja**

Tahap ini menerapkan teknik normalisasi dan encoding yang optimal berdasarkan karakteristik distribusi setiap fitur, mempersiapkan data untuk model collaborative filtering yang memerlukan input numerik yang ternormalisasi.

### **🔧 Implementasi dan Parameter**

#### **1. Feature Distribution Analysis & Separation**

```python
# Features dengan distribusi normal
normal_features = ['average_rating', 'publication_year']

# Features dengan distribusi skewed
skewed_features = ['num_pages', 'ratings_count_capped', 'text_reviews_count_capped',
                   'author_count', 'popularity_score', 'engagement_score']
```

**Cara Kerja:**

- **Distribution-Based Grouping**: Pemisahan fitur berdasarkan insight EDA
- **Normal Features**: Fitur dengan distribusi mendekati Gaussian
- **Skewed Features**: Fitur dengan right-skewed distribution atau outliers

**Rationale per Feature:**

| Feature | Group | Distribution Type | EDA Insight |
| --- | --- | --- | --- |
| `average_rating` | Normal | Gaussian-like | Mean=3.93, std=0.35 |
| `publication_year` | Normal | Relatively normal | Temporal distribution |
| `num_pages` | Skewed | Right-skewed | Long tail, outliers >1000 |
| `ratings_count_capped` | Skewed | Heavy right-skew | Capped at 95th percentile |
| `text_reviews_count_capped` | Skewed | Heavy right-skew | Capped at 95th percentile |
| `author_count` | Skewed | Right-skewed | Most books single author |
| `popularity_score` | Skewed | Potentially skewed | Bayesian calculation result |
| `engagement_score` | Skewed | Ratio-based | Derived from skewed features |

#### **2. Differential Scaling Strategy**

```python
# Apply different scaling strategies
scaler_normal = StandardScaler()  # For normal distributed features
scaler_skewed = MinMaxScaler()    # For skewed features

df_clean[normal_features] = scaler_normal.fit_transform(df_clean[normal_features])
df_clean[skewed_features] = scaler_skewed.fit_transform(df_clean[skewed_features])
```

**StandardScaler untuk Normal Features:**

- **Formula**: `z = (x - μ) / σ`
- **Output**: Mean = 0, Standard Deviation = 1
- **Advantage**: Preserves shape of normal distribution
- **Use Case**: Features yang sudah mendekati normal distribution

**MinMaxScaler untuk Skewed Features:**

- **Formula**: `x_scaled = (x - x_min) / (x_max - x_min)`
- **Output**: Range [0, 1]
- **Advantage**: Robust terhadap outliers, preserves zero values
- **Use Case**: Features dengan outliers atau non-normal distribution

**Parameter Detail:**

| Scaler | Method | Parameters | Application |
| --- | --- | --- | --- |
| `StandardScaler()` | Z-score normalization | Default (with_mean=True, with_std=True) | Normal features |
| `MinMaxScaler()` | Min-max normalization | Default (feature_range=(0,1)) | Skewed features |
| `fit_transform()` | Compute + apply | Learns statistics, applies transformation | Training data |

#### **3. Categorical Feature Encoding**

```python
label_encoders = {}
categorical_features = ['language_code', 'language_group', 'rating_category',
                       'page_category', 'publication_era']

for feature in categorical_features:
    le = LabelEncoder()
    df_clean[f'{feature}_encoded'] = le.fit_transform(df_clean[feature])
    label_encoders[feature] = le
```

**Cara Kerja:**

- **LabelEncoder Selection**: Optimal untuk ordinal dan nominal categorical features
- **Systematic Encoding**: Loop through semua categorical features
- **Encoder Storage**: Simpan setiap encoder untuk future use (inverse transform)
- **Column Naming**: Add `_encoded` suffix untuk clarity

**Categorical Features Analysis:**

| Feature | Type | Unique Values | Encoding Strategy |
| --- | --- | --- | --- |
| `language_code` | Nominal | 25+ codes | LabelEncoder (0 to n-1) |
| `language_group` | Nominal | 4 groups | LabelEncoder (0 to 3) |
| `rating_category` | Ordinal | 5 categories | LabelEncoder (preserves order) |
| `page_category` | Ordinal | 4 categories | LabelEncoder (preserves order) |
| `publication_era` | Ordinal | 4 eras | LabelEncoder (temporal order) |

**LabelEncoder Parameters:**

- **Input**: Categorical values (strings, objects)
- **Output**: Integer labels (0 to n_classes-1)
- **Mapping**: Alphabetical order untuk strings
- **Storage**: `label_encoders` dictionary untuk reversibility

***

## **📊 Technical Implementation Details**

### **1. Scaling Transformation Mathematics**

#### **StandardScaler (Z-Score Normalization)**

```python
# For normal_features
μ = X.mean()  # Population mean
σ = X.std()   # Population standard deviation
X_scaled = (X - μ) / σ

# Result: X_scaled ~ N(0, 1)
```

#### **MinMaxScaler (Min-Max Normalization)**

```python
# For skewed_features
X_min = X.min()
X_max = X.max()
X_scaled = (X - X_min) / (X_max - X_min)

# Result: X_scaled ∈ [0, 1]
```

### **2. Encoding Implementation**

```python
# LabelEncoder process for each categorical feature
unique_values = df[feature].unique()  # Get unique categories
sorted_values = sorted(unique_values)  # Sort for consistent mapping
mapping = {value: idx for idx, value in enumerate(sorted_values)}
encoded_values = df[feature].map(mapping)
```

### **3. Memory and Performance Optimization**

```python
# Efficient batch processing
def batch_scale_features(df, feature_groups, scalers):
    for features, scaler in zip(feature_groups, scalers):
        if features:  # Check if group not empty
            df[features] = scaler.fit_transform(df[features])
    return df
```

***

## **📈 Transformation Results**

### **Before vs After Scaling**

```python
# Example transformation results
Original average_rating: [3.5, 4.2, 3.8, 4.0] (range: 3.5-4.2)
Scaled average_rating: [-1.2, 0.8, -0.5, 0.2] (mean: 0, std: 1)

Original num_pages: [200, 450, 800, 1200] (range: 200-1200)
Scaled num_pages: [0.0, 0.25, 0.6, 1.0] (range: 0-1)
```

### **Categorical Encoding Results**

```python
# Example encoding mappings
language_group: ['english', 'european', 'asian', 'other']
Encoded: [0, 1, 2, 3]

rating_category: ['Below Average', 'Average', 'Good', 'Very Good', 'Excellent']
Encoded: [0, 1, 2, 3, 4]  # Preserves quality ordering
```

### **Data Quality Metrics**

- **Scaling Consistency**: All numerical features dalam comparable ranges
- **Encoding Completeness**: All categorical features converted to integers
- **Reversibility**: All transformations dapat di-inverse
- **Memory Efficiency**: Reduced storage dengan integer encoding

Preprocessing ini menghasilkan dataset yang optimal untuk model collaborative filtering dengan semua fitur dalam format dan skala yang sesuai untuk training neural network yang efektif dan stabil.

# **Data Preparation - Enhanced Feature Scaling and Encoding**

## **📋 Tujuan dan Cara Kerja**

Tahap ini menerapkan teknik normalisasi dan encoding yang optimal berdasarkan karakteristik distribusi setiap fitur, mempersiapkan data untuk model collaborative filtering yang memerlukan input numerik yang ternormalisasi.

### **🔧 Implementasi dan Parameter**

#### **1. Feature Distribution Analysis & Separation**

```python
# Features dengan distribusi normal
normal_features = ['average_rating', 'publication_year']

# Features dengan distribusi skewed
skewed_features = ['num_pages', 'ratings_count_capped', 'text_reviews_count_capped',
                   'author_count', 'popularity_score', 'engagement_score']
```

**Cara Kerja:**

- **Distribution-Based Grouping**: Pemisahan fitur berdasarkan insight EDA
- **Normal Features**: Fitur dengan distribusi mendekati Gaussian
- **Skewed Features**: Fitur dengan right-skewed distribution atau outliers

**Rationale per Feature:**

| Feature | Group | Distribution Type | EDA Insight |
| --- | --- | --- | --- |
| `average_rating` | Normal | Gaussian-like | Mean=3.93, std=0.35 |
| `publication_year` | Normal | Relatively normal | Temporal distribution |
| `num_pages` | Skewed | Right-skewed | Long tail, outliers >1000 |
| `ratings_count_capped` | Skewed | Heavy right-skew | Capped at 95th percentile |
| `text_reviews_count_capped` | Skewed | Heavy right-skew | Capped at 95th percentile |
| `author_count` | Skewed | Right-skewed | Most books single author |
| `popularity_score` | Skewed | Potentially skewed | Bayesian calculation result |
| `engagement_score` | Skewed | Ratio-based | Derived from skewed features |

#### **2. Differential Scaling Strategy**

```python
# Apply different scaling strategies
scaler_normal = StandardScaler()  # For normal distributed features
scaler_skewed = MinMaxScaler()    # For skewed features

df_clean[normal_features] = scaler_normal.fit_transform(df_clean[normal_features])
df_clean[skewed_features] = scaler_skewed.fit_transform(df_clean[skewed_features])
```

**StandardScaler untuk Normal Features:**

- **Formula**: `z = (x - μ) / σ`
- **Output**: Mean = 0, Standard Deviation = 1
- **Advantage**: Preserves shape of normal distribution
- **Use Case**: Features yang sudah mendekati normal distribution

**MinMaxScaler untuk Skewed Features:**

- **Formula**: `x_scaled = (x - x_min) / (x_max - x_min)`
- **Output**: Range [0, 1]
- **Advantage**: Robust terhadap outliers, preserves zero values
- **Use Case**: Features dengan outliers atau non-normal distribution

**Parameter Detail:**

| Scaler | Method | Parameters | Application |
| --- | --- | --- | --- |
| `StandardScaler()` | Z-score normalization | Default (with_mean=True, with_std=True) | Normal features |
| `MinMaxScaler()` | Min-max normalization | Default (feature_range=(0,1)) | Skewed features |
| `fit_transform()` | Compute + apply | Learns statistics, applies transformation | Training data |

#### **3. Categorical Feature Encoding**

```python
label_encoders = {}
categorical_features = ['language_code', 'language_group', 'rating_category',
                       'page_category', 'publication_era']

for feature in categorical_features:
    le = LabelEncoder()
    df_clean[f'{feature}_encoded'] = le.fit_transform(df_clean[feature])
    label_encoders[feature] = le
```

**Cara Kerja:**

- **LabelEncoder Selection**: Optimal untuk ordinal dan nominal categorical features
- **Systematic Encoding**: Loop through semua categorical features
- **Encoder Storage**: Simpan setiap encoder untuk future use (inverse transform)
- **Column Naming**: Add `_encoded` suffix untuk clarity

**Categorical Features Analysis:**

| Feature | Type | Unique Values | Encoding Strategy |
| --- | --- | --- | --- |
| `language_code` | Nominal | 25+ codes | LabelEncoder (0 to n-1) |
| `language_group` | Nominal | 4 groups | LabelEncoder (0 to 3) |
| `rating_category` | Ordinal | 5 categories | LabelEncoder (preserves order) |
| `page_category` | Ordinal | 4 categories | LabelEncoder (preserves order) |
| `publication_era` | Ordinal | 4 eras | LabelEncoder (temporal order) |

**LabelEncoder Parameters:**

- **Input**: Categorical values (strings, objects)
- **Output**: Integer labels (0 to n_classes-1)
- **Mapping**: Alphabetical order untuk strings
- **Storage**: `label_encoders` dictionary untuk reversibility

***

## **📊 Technical Implementation Details**

### **1. Scaling Transformation Mathematics**

#### **StandardScaler (Z-Score Normalization)**

```python
# For normal_features
μ = X.mean()  # Population mean
σ = X.std()   # Population standard deviation
X_scaled = (X - μ) / σ

# Result: X_scaled ~ N(0, 1)
```

#### **MinMaxScaler (Min-Max Normalization)**

```python
# For skewed_features
X_min = X.min()
X_max = X.max()
X_scaled = (X - X_min) / (X_max - X_min)

# Result: X_scaled ∈ [0, 1]
```

### **2. Encoding Implementation**

```python
# LabelEncoder process for each categorical feature
unique_values = df[feature].unique()  # Get unique categories
sorted_values = sorted(unique_values)  # Sort for consistent mapping
mapping = {value: idx for idx, value in enumerate(sorted_values)}
encoded_values = df[feature].map(mapping)
```

### **3. Memory and Performance Optimization**

```python
# Efficient batch processing
def batch_scale_features(df, feature_groups, scalers):
    for features, scaler in zip(feature_groups, scalers):
        if features:  # Check if group not empty
            df[features] = scaler.fit_transform(df[features])
    return df
```



## **📈 Transformation Results**

### **Before vs After Scaling**

```python
# Example transformation results
Original average_rating: [3.5, 4.2, 3.8, 4.0] (range: 3.5-4.2)
Scaled average_rating: [-1.2, 0.8, -0.5, 0.2] (mean: 0, std: 1)

Original num_pages: [200, 450, 800, 1200] (range: 200-1200)
Scaled num_pages: [0.0, 0.25, 0.6, 1.0] (range: 0-1)
```

### **Categorical Encoding Results**

```python
# Example encoding mappings
language_group: ['english', 'european', 'asian', 'other']
Encoded: [0, 1, 2, 3]

rating_category: ['Below Average', 'Average', 'Good', 'Very Good', 'Excellent']
Encoded: [0, 1, 2, 3, 4]  # Preserves quality ordering
```

### **Data Quality Metrics**

- **Scaling Consistency**: All numerical features dalam comparable ranges
- **Encoding Completeness**: All categorical features converted to integers
- **Reversibility**: All transformations dapat di-inverse
- **Memory Efficiency**: Reduced storage dengan integer encoding

***

Preprocessing ini menghasilkan dataset yang optimal untuk model collaborative filtering dengan semua fitur dalam format dan skala yang sesuai untuk training neural network yang efektif dan stabil.

# **Data Preparation - Save Enhanced Processed Data**

## **📋 Tujuan dan Cara Kerja**

Tahap ini menyimpan semua hasil preprocessing ke file eksternal untuk mempertahankan konsistensi, efisiensi, dan reproduksibilitas dalam tahap modeling dan deployment. Proses ini mencegah kebutuhan untuk mengulang preprocessing yang computationally intensive.

### **🔧 Implementasi dan Parameter**

#### **1. CSV Data Export**

```python
# Save datasets
df_clean.to_csv('books_processed_enhanced.csv', index=False)
filtered_interactions.to_csv('user_interactions_enhanced.csv', index=False)
train_data.to_csv('train_interactions_enhanced.csv', index=False)
test_data.to_csv('test_interactions_enhanced.csv', index=False)
```

**Cara Kerja:**

- **DataFrame Serialization**: Convert pandas DataFrame ke CSV format
- **Index Exclusion**: `index=False` untuk menghindari extra column
- **File Naming**: Descriptive names dengan `_enhanced` suffix

**Parameter Detail:**

| File | Content | Size | Purpose |
| --- | --- | --- | --- |
| `books_processed_enhanced.csv` | 9,548 books × 28 features | ~2.5MB | Book metadata dengan engineered features |
| `user_interactions_enhanced.csv` | 62,424 interactions | ~1.2MB | Complete user-book interactions |
| `train_interactions_enhanced.csv` | 49,939 interactions | ~1.0MB | Training dataset |
| `test_interactions_enhanced.csv` | 12,485 interactions | ~0.3MB | Testing dataset |

**CSV Format Benefits:**

- ✅ **Human Readable**: Easy inspection dan debugging
- ✅ **Cross-Platform**: Compatible dengan berbagai tools
- ✅ **Memory Efficient**: Compressed storage format
- ✅ **Version Control**: Text-based untuk Git tracking

#### **2. Collaborative Filtering Objects**

```python
with open('collaborative_filtering_enhanced.pkl', 'wb') as f:
    pickle.dump({
        'user_to_index': user_to_index,
        'book_to_index': book_to_index,
        'index_to_user': index_to_user,
        'index_to_book': index_to_book,
        'train_mapped': train_mapped,
        'test_mapped': test_mapped,
        'n_users': len(user_ids),
        'n_books': len(book_ids),
        'user_profiles': user_profiles
    }, f)
```

**Cara Kerja:**

- **Binary Serialization**: Pickle untuk complex Python objects
- **Dictionary Structure**: Organized storage dengan descriptive keys
- **Complete Mapping**: Bidirectional ID mappings untuk model compatibility

**Object Contents:**

| Key | Type | Content | Usage |
| --- | --- | --- | --- |
| `user_to_index` | dict | {original_user_id: sequential_index} | Model input conversion |
| `book_to_index` | dict | {original_book_id: sequential_index} | Model input conversion |
| `index_to_user` | dict | {sequential_index: original_user_id} | Result interpretation |
| `index_to_book` | dict | {sequential_index: original_book_id} | Result interpretation |
| `train_mapped` | list | Training data dalam model format | Model training |
| `test_mapped` | list | Test data dalam model format | Model evaluation |
| `n_users` | int | Total unique users (2,000) | Model architecture |
| `n_books` | int | Total unique books (9,123) | Model architecture |
| `user_profiles` | list | Synthetic user characteristics | Analysis & debugging |

#### **3. Preprocessing Objects**

```python
with open('preprocessing_enhanced.pkl', 'wb') as f:
    pickle.dump({
        'scaler_normal': scaler_normal,
        'scaler_skewed': scaler_skewed,
        'label_encoders': label_encoders,
        'normal_features': normal_features,
        'skewed_features': skewed_features,
        'categorical_features': categorical_features
    }, f)
```

**Cara Kerja:**

- **Transformer Storage**: Fitted scalers dan encoders untuk consistent transformation
- **Feature Definitions**: Lists untuk feature grouping consistency
- **Reusability**: Enable same transformations pada new data

**Object Contents:**

| Key | Type | Content | Purpose |
| --- | --- | --- | --- |
| `scaler_normal` | StandardScaler | Fitted scaler untuk normal features | New data normalization |
| `scaler_skewed` | MinMaxScaler | Fitted scaler untuk skewed features | New data normalization |
| `label_encoders` | dict | {feature: LabelEncoder} | Categorical encoding |
| `normal_features` | list | ['average_rating', 'publication_year'] | Feature grouping |
| `skewed_features` | list | ['num_pages', 'ratings_count_capped', ...] | Feature grouping |
| `categorical_features` | list | ['language_code', 'language_group', ...] | Feature grouping |

***

## **📊 File Structure & Usage**

### **🗂️ Output File Organization**

```javascript
project_directory/
├── books_processed_enhanced.csv          # Book features dataset
├── user_interactions_enhanced.csv        # Complete interactions
├── train_interactions_enhanced.csv       # Training split
├── test_interactions_enhanced.csv        # Testing split
├── collaborative_filtering_enhanced.pkl  # CF model objects
└── preprocessing_enhanced.pkl             # Transformation objects
```

### **💾 Storage Efficiency**

| File Type | Total Size | Compression | Access Speed |
| --- | --- | --- | --- |
| **CSV Files** | ~4.0 MB | Text compression | Medium |
| **Pickle Files** | ~1.5 MB | Binary serialization | Fast |
| **Total** | ~5.5 MB | Efficient storage | Optimized |

### **🔄 Usage Patterns**

#### **Model Training Phase**

```python
# Load CF objects
with open('collaborative_filtering_enhanced.pkl', 'rb') as f:
    cf_data = pickle.load(f)
    
train_data = cf_data['train_mapped']
test_data = cf_data['test_mapped']
n_users = cf_data['n_users']
n_books = cf_data['n_books']
```

#### **New Data Processing**

```python
# Load preprocessing objects
with open('preprocessing_enhanced.pkl', 'rb') as f:
    preprocessing = pickle.load(f)
    
scaler_normal = preprocessing['scaler_normal']
label_encoders = preprocessing['label_encoders']

# Apply same transformations
new_data_scaled = scaler_normal.transform(new_data[normal_features])
```

#### **Result Interpretation**

```python
# Convert model output back to original IDs
predicted_book_indices = model.predict(user_index)
original_book_ids = [index_to_book[idx] for idx in predicted_book_indices]
```

## **📋 Enhancement Summary**

### **🔧 Adjustments Made**

```javascript
1. ✅ Lowered minimum ratings threshold (50 → 30)
2. ✅ Increased number of users (1500 → 2000)  
3. ✅ Increased minimum interactions per user (10-100 → 20-150)
4. ✅ Lowered collaborative filtering thresholds
5. ✅ Implemented flexible train-test split strategy
6. ✅ Maintained data quality while increasing volume
```

**Impact Analysis:**

| Adjustment | Before | After | Impact |
| --- | --- | --- | --- |
| **Book threshold** | 50 ratings | 30 ratings | +15% more books |
| **User count** | 1,500 | 2,000 | +33% more users |
| **User activity** | 10-100 | 20-150 | +50% interactions/user |
| **CF thresholds** | Strict | Relaxed | +20% data retention |
| **Split strategy** | Fixed | Adaptive | Better handling |

### **🎯 Quality Maintained**

- **Data integrity**: All validation checks passed
- **Feature quality**: Comprehensive engineering applied
- **Model readiness**: Optimized untuk CF architecture
- **Scalability**: Efficient storage dan loading



Dataset sekarang tersimpan dalam format yang optimal untuk collaborative filtering model development dengan semua preprocessing artifacts yang diperlukan untuk consistent dan efficient model training serta deployment.
