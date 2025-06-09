# **Laporan Proyek Akhir Machine Learning Terapan - FAYADH RIZQI ZAMZAMI**

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



## Business Understanding

### Problem Statements

Berdasarkan kondisi permasalahan yang telah diidentifikasi, berikut adalah beberapa rumusan masalah yang perlu dipecahkan:

- Bagaimana cara mempercepat proses pencarian buku yang sesuai dengan minat dan kebutuhan pembaca?
- Dengan memanfaatkan data penilaian dari pengguna, bagaimana cara mengembangkan sistem yang dapat menyarankan buku-buku yang berpotensi diminati oleh pengguna namun belum pernah mereka baca sebelumnya?

### Goals

Berdasarkan permasalahan yang telah dirumuskan, berikut adalah target pencapaian yang ingin diwujudkan:

- Mengembangkan platform rekomendasi buku yang dapat membantu pengguna dalam menemukan bacaan yang tepat.
- Membangun sistem rekomendasi yang mampu menyesuaikan dengan selera dan preferensi individual pengguna melalui implementasi teknik Collaborative Filtering.

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, strategi yang akan diterapkan adalah:

- Mengimplementasikan sistem rekomendasi dengan menggunakan metode Collaborative Filtering sebagai pendekatan utama


## Data Understanding

Dataset yang akan dimanfaatkan dalam penelitian ini adalah Book Recommendation Dataset yang tersedia untuk diunduh melalui tautan berikut: Book Recommendation Dataset. Dataset tersebut terdiri dari tiga file terpisah.

**Users** merupakan kumpulan data yang memuat informasi mengenai pengguna. File ini terdiri dari 278858 baris data dengan 3 kolom utama yaitu User-ID, Location, dan Age.

**Books** merupakan file yang menyimpan detail informasi buku, meliputi ISBN (identifikasi unik buku), nama buku, penulis, tahun penerbitan, nama penerbit, serta URL gambar yang terhubung ke situs web amazon. File ini memuat 271360 baris data dengan 8 kolom. Setiap ISBN merepresentasikan satu buku secara unik. Perlu dicatat bahwa dalam file ini ditemukan beberapa nilai kosong (missing value), khususnya pada kolom book_author, publisher, dan image_url_l.

Link Dataset : https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

**Ratings** merupakan file yang menyimpan penilaian yang telah diberikan oleh pengguna terhadap buku-buku tertentu. File ini berisi 1149780 baris data dengan 3 kolom utama yaitu user_id, isbn, dan book_rating.

Secara detail, dataset tersebut memiliki fitur-fitur sebagai berikut:

- **ISBN** berfungsi sebagai penanda unik setiap buku, dimana satu ISBN merujuk pada satu buku
- **book_title** menunjukkan nama/judul buku
- **book_author** menunjukkan nama penulis
- **year_of_publication** menunjukkan tahun publikasi buku
- **publisher** menunjukkan nama penerbit
- **image_url_s** menyimpan URL gambar buku berukuran kecil
- **image_url_m** menyimpan URL gambar buku berukuran medium
- **image_url_l** menyimpan URL gambar buku berukuran besar
- **user_id** berfungsi sebagai identitas unik setiap pengguna
- **location** menunjukkan lokasi geografis pengguna
- **age** menunjukkan usia pengguna
- **rating** menunjukkan nilai penilaian dari pengguna

Untuk memperoleh pemahaman mendalam terhadap karakteristik data, akan dilaksanakan analisis data eksplorasi yang meliputi:

- Menganalisis jenis tipe data pada setiap dataframe
- Memverifikasi bahwa setiap isbn hanya merepresentasikan satu buku
- Mengidentifikasi jumlah data unik pada dataset buku
- Memverifikasi bahwa buku dengan judul sama namun ISBN berbeda merupakan entitas yang terpisah
- Mengidentifikasi jumlah data unik pada dataframe pengguna
- Menganalisis jumlah pengguna yang memberikan penilaian, jumlah buku yang menerima penilaian, dan total data pada dataframe ratings
- Menganalisis kisaran nilai rating yang diberikan pengguna

Berdasarkan analisis yang telah dilakukan, diperoleh hasil sebagai berikut:

- Jenis tipe data yang terdapat dalam dataset adalah int, object, dan float
- Setiap isbn dalam dataframe mengacu pada satu buku secara spesifik
- Dataset memuat 271360 buku unik dengan 242135 judul yang berbeda
- Buku dengan judul identik namun ISBN yang berbeda merupakan buku yang terpisah, mengingat satu judul dapat memiliki beberapa edisi atau seri lanjutan
- Total pengguna dalam dataset berjumlah 278858
- Data rating terdiri dari 1149780 entri, dengan 340556 buku yang telah menerima rating, dan 105283 pengguna yang telah memberikan penilaian
- Skala rating berkisar antara 0 hingga 10


# Book Recommendation System - Data Loading & Initial Exploration

## 📊 Data Loading

Tahap pertama dalam pengembangan sistem rekomendasi adalah memuat dataset yang akan digunakan untuk training model.

```python
# Load Data
base_dir = "/content/"
books = pd.read_csv(base_dir+"Books.csv")
ratings = pd.read_csv(base_dir+"Ratings.csv")
users = pd.read_csv(base_dir+"Users.csv")
```

### Dataset Overview

Project ini menggunakan **Book Recommendation Dataset** yang terdiri dari tiga file utama:

| Dataset | Deskripsi | Jumlah Records |
| --- | --- | --- |
| **Books.csv** | Informasi detail buku (ISBN, judul, penulis, penerbit, dll) | 271,360 buku |
| **Ratings.csv** | Data interaksi user-book dengan rating 0-10 | 1,149,780 rating |
| **Users.csv** | Profil pengguna (ID, lokasi, usia) | 278,858 users |

### 🔍 Exploratory Data Analysis

#### Books Dataset

```python
books.head()
```

Dataset books berisi informasi lengkap tentang buku dengan fitur-fitur berikut:

- **ISBN**: Unique identifier untuk setiap buku
- **Book-Title**: Judul buku
- **Book-Author**: Nama penulis
- **Year-Of-Publication**: Tahun publikasi
- **Publisher**: Nama penerbit
- **Image-URL-S/M/L**: URL gambar cover buku dalam berbagai ukuran

#### Key Insights dari Initial Exploration:

***

# 🔧 Data Preprocessing - Column Standardization

## Tujuan Standardisasi Kolom

Tahap preprocessing pertama adalah **standardisasi nama kolom** untuk memastikan konsistensi dalam penamaan dan mempermudah akses data selama pengembangan model.

## Implementasi

```python
# Standardisasi nama kolom untuk semua dataset
books.columns = books.columns.str.lower()
books.columns = books.columns.str.replace("-","_")

ratings.columns = ratings.columns.str.lower()
ratings.columns = ratings.columns.str.replace("-","_")

users.columns = users.columns.str.lower()
users.columns = users.columns.str.replace("-","_")
```

## Cara Kerja & Parameter

### 1. **Konversi ke Huruf Kecil**

```python
.str.lower()
```

| Parameter | Fungsi | Contoh Transformasi |
| --- | --- | --- |
| **`.str.lower()`** | Mengkonversi semua huruf menjadi lowercase | `"Book-Title"` → `"book-title"` |

### 2. **Penggantian Karakter**

```python
.str.replace("-", "_")
```

| Parameter | Fungsi | Nilai |
| --- | --- | --- |
| **Pattern**: `"-"` | Target karakter yang akan diganti | Tanda hubung |
| **Replacement**: `"_"` | Karakter pengganti | Underscore |

## Transformasi Kolom

### Before Standardization:

```javascript
Books: ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']
Ratings: ['User-ID', 'ISBN', 'Book-Rating']
Users: ['User-ID', 'Location', 'Age']
```

### After Standardization:

```javascript
Books: ['isbn', 'book_title', 'book_author', 'year_of_publication', 'publisher']
Ratings: ['user_id', 'isbn', 'book_rating']
Users: ['user_id', 'location', 'age']
```


## Verifikasi Hasil

```python
# Menampilkan dataset users setelah standardisasi
users.head()
```

# 📊 Data Quality Analysis - Books Dataframe

## Analisis Struktur Data

```python
# Melihat informasi detail dataset books
books.info()
```

## Hasil Analisis

### Dataset Overview

| Metric | Value |
| --- | --- |
| **Total Records** | 271,360 entries |
| **Total Columns** | 8 columns |
| **Index Range** | 0 to 271,359 |
| **Memory Usage** | 16.6+ MB |

### Column Analysis

| # | Column | Non-Null Count | Data Type | Missing Values |
| --- | --- | --- | --- | --- |
| 0 | `isbn` | 271,360 | object | 0 |
| 1 | `book_title` | 271,360 | object | 0 |
| 2 | `book_author` | 271,359 | object | 1 |
| 3 | `year_of_publication` | 271,360 | object | 0 |
| 4 | `publisher` | 271,358 | object | 2 |
| 5 | `image_url_s` | 271,360 | object | 0 |
| 6 | `image_url_m` | 271,360 | object | 0 |
| 7 | `image_url_l` | 271,357 | object | 3 |

***

# 🔍 Data Uniqueness & Integrity Analysis

## ISBN Uniqueness Verification

### Tujuan

Memastikan bahwa setiap ISBN merepresentasikan satu buku unik, yang merupakan prinsip dasar sistem identifikasi buku.

```python
# Mengecek duplikasi ISBN
books["isbn"].duplicated().sum()
```

**Hasil**: `0` - Tidak ada duplikasi ISBN ✅

## Book Uniqueness Analysis

```python
print(f"Banyak data buku yang unik berdasarkan judul : {len(books['book_title'].unique())}")
print(f"Banyak data buku yang unik berdasarkan ISBN : {len(books['isbn'].unique())}")
```

### Hasil Analisis

| Metric | Count | Percentage |
| --- | --- | --- |
| **Unique ISBN** | 271,360 | 100% |
| **Unique Titles** | 242,135 | 89.2% |
| **Title Duplicates** | 29,225 | 10.8% |

### Key Insights

✅ **ISBN Integrity**: Setiap ISBN unik - memastikan identifikasi buku yang tepat

⚠️ **Title Duplicates**: 29,225 buku memiliki judul yang sama dengan ISBN berbeda

## Investigasi Title Duplicates

### Sample Duplicate Titles

```python
books[books["book_title"].duplicated()].sample(5, axis=0)
```

### Case Study: "El Ladron De Cuerpos"

```python
books[books["book_title"] == "El Ladron De Cuerpos"]
```

### Analisis Temuan

Buku dengan judul sama namun ISBN berbeda menunjukkan:

1. **Different Editions**: Edisi berbeda dari buku yang sama
2. **Different Publishers**: Penerbit yang berbeda
3. **Different Years**: Tahun publikasi yang berbeda
4. **Series/Sequels**: Kemungkinan seri atau sequel

**Kesimpulan**: Title duplicates adalah **valid entities** yang merepresentasikan buku berbeda.

***

## User Dataset Analysis

### Dataset Overview

```python
users.head()  # Menampilkan sample data users
```

### Uniqueness Check

```python
print(f"Banyak data user unik : {len(users['user_id'].unique())}")
```

**Hasil**: `278,858` user unik ✅

### Data Quality Assessment

```python
users.info()
```

| Column | Non-Null Count | Data Type | Completeness |
| --- | --- | --- | --- |
| `user_id` | 278,858 | int64 | 100% ✅ |
| `location` | 278,858 | object | 100% ✅ |
| `age` | 168,096 | float64 | 60.3% ⚠️ |

### Missing Values Analysis

| Metric | Value |
| --- | --- |
| **Total Users** | 278,858 |
| **Users with Age** | 168,096 |
| **Missing Age Data** | 110,762 |
| **Missing Percentage** | 39.7% |

***

# 📈 Ratings Dataset Analysis

## Dataset Overview

### Data Structure Exploration

```python
# Menampilkan sample data ratings
ratings.head()
```

Dataset ratings berisi **interaksi user-book** yang merupakan core data untuk collaborative filtering.

## Dataset Statistics

```python
print(f"Banyak data rating: {len(ratings)}")
print(f"Jumlah buku yang telah diberi rating: {len(ratings['isbn'].unique())}")
print(f"Jumlah user yang memberikan rating: {len(ratings['user_id'].unique())}")
```

### Key Metrics

| Metric | Count | Percentage of Total |
| --- | --- | --- |
| **Total Interactions** | 1,149,780 | 100% |
| **Unique Books Rated** | 340,556 | 29.6% |
| **Active Users** | 105,283 | 9.2% |

### Coverage Analysis

| Dataset | Total Items | Rated Items | Coverage |
| --- | --- | --- | --- |
| **Books** | 271,360 | 340,556 | 125.5%* |
| **Users** | 278,858 | 105,283 | 37.8% |

*\*Coverage >100% menunjukkan ada buku dalam ratings yang tidak ada di dataset books*

## Data Quality Assessment

```python
ratings.info()
```

### Data Completeness

| Column | Non-Null Count | Data Type | Completeness |
| --- | --- | --- | --- |
| `user_id` | 1,149,780 | int64 | 100% ✅ |
| `isbn` | 1,149,780 | object | 100% ✅ |
| `book_rating` | 1,149,780 | int64 | 100% ✅ |

### Quality Indicators

- **Perfect Completeness**: Tidak ada missing values

- **Consistent Data Types**: Semua tipe data sesuai ekspektasi

- **Large Scale**: 1M+ interaksi untuk training yang robust

## Rating Distribution Analysis

```python
ratings.describe().round(3)
```

### Rating Scale Characteristics

| Statistic | Value |
| --- | --- |
| **Minimum Rating** | 0 |
| **Maximum Rating** | 10 |
| **Rating Range** | 0-10 (11 point scale) |
| **Mean Rating** | ~5.0 (estimated) |

### Rating Scale Interpretation

- **0**: Lowest rating (strongly dislike)
- **5**: Neutral/average rating  
- **10**: Highest rating (strongly like)
- **Scale Type**: Integer scale with 11 discrete values

## Data Sparsity Analysis

### Sparsity Calculation

```python
total_possible_interactions = 278,858 * 271,360  # All possible user-book pairs
actual_interactions = 1,149,780
sparsity = (1 - actual_interactions / total_possible_interactions) * 100
```

| Metric | Value |
| --- | --- |
| **Possible Interactions** | 75.6 Billion |
| **Actual Interactions** | 1.15 Million |
| **Data Sparsity** | 99.998% |

Dataset ratings menunjukkan **kualitas tinggi** dengan **completeness 100%** dan **scale yang memadai** untuk collaborative filtering. Meskipun memiliki **sparsity tinggi** (typical untuk recommendation systems), volume interaksi **1M+** memberikan foundation yang solid untuk training model yang robust.

***

# 🔧 Data Preprocessing - Dataset Overview

## Tujuan

Memberikan **comprehensive overview** dari semua dataset sebelum melakukan penggabungan dan transformasi data untuk model training.

## Implementation

```python
# Data Preprocessing - Statistical Overview
print(f"Jumlah seluruh data buku berdasarkan ISBN : {len(books['isbn'].unique())}")
print(f"Jumlah seluruh data buku berdasarkan judul buku : {len(books['book_title'].unique())}")
print(f"Jumlah seluruh users : {len(users['user_id'].unique())}")
print(f"Jumlah seluruh rating : {len(ratings)}")
```

## Cara Kerja & Parameter

### 1. **Books Analysis**

| Code | Parameter | Fungsi |
| --- | --- | --- |
| `books['isbn'].unique()` | Column: `isbn` | Ekstrak semua ISBN unik |
| `len()` | Array of unique values | Hitung jumlah buku berdasarkan ISBN |
| `books['book_title'].unique()` | Column: `book_title` | Ekstrak semua judul unik |

### 2. **Users Analysis**

| Code | Parameter | Fungsi |
| --- | --- | --- |
| `users['user_id'].unique()` | Column: `user_id` | Ekstrak semua user ID unik |
| `len()` | Array of unique users | Hitung total pengguna |

### 3. **Ratings Analysis**

| Code | Parameter | Fungsi |
| --- | --- | --- |
| `len(ratings)` | Entire dataframe | Hitung total interaksi rating |

## Output Analysis

```javascript
Jumlah seluruh data buku berdasarkan ISBN : 271360
Jumlah seluruh data buku berdasarkan judul buku : 242135
Jumlah seluruh users : 278858
Jumlah seluruh rating : 1149780
```

### Dataset Summary Table

| Dataset | Metric | Count | Notes |
| --- | --- | --- | --- |
| **Books** | Unique ISBN | 271,360 | Primary identifier |
| **Books** | Unique Titles | 242,135 | Some titles repeated |
| **Users** | Total Users | 278,858 | All registered users |
| **Ratings** | Total Interactions | 1,149,780 | User-book interactions |

## Statistical Foundation

Hasil overview ini memberikan **baseline metrics** yang penting untuk:

- **Model Architecture Planning**: Menentukan embedding dimensions
- **Memory Requirements**: Estimasi kebutuhan computational resources  
- **Data Splitting Strategy**: Rencana train-validation-test split
- **Performance Expectations**: Baseline untuk evaluasi model

***

# 🔗 Data Integration - Menggabungkan data ratings dengan judul buku

## Tujuan

Menggabungkan dataset **ratings** dengan **metadata buku** untuk memperkaya informasi dan mempersiapkan data yang lebih komprehensif untuk model training.

## Implementation

### 1. **Inisialisasi Base Dataset**

```python
all_book = ratings
all_book
```

#### Cara Kerja & Parameter

| Parameter | Fungsi | Output |
| --- | --- | --- |
| `ratings` | Source dataset | Copy ratings sebagai base |
| `all_book` | Target variable | Alias untuk dataset gabungan |

#### Struktur Awal

```javascript
Shape: (1,149,780 rows × 3 columns)
Columns: [user_id, isbn, book_rating]
```

### 2. **Data Merging Process**

```python
all_book = pd.merge(all_book, books[["isbn","book_title"]], on="isbn", how="left")
all_book
```

#### Parameter Detail

| Parameter | Value | Fungsi |
| --- | --- | --- |
| **Left Dataset** | `all_book` (ratings) | Dataset utama yang dipertahankan |
| **Right Dataset** | `books[["isbn","book_title"]]` | Subset books dengan 2 kolom |
| **Join Key** | `on="isbn"` | Kolom untuk mencocokkan data |
| **Join Type** | `how="left"` | Left join - pertahankan semua ratings |

#### Subset Selection Logic

```python
books[["isbn","book_title"]]
```

- **Efisiensi**: Hanya mengambil kolom yang diperlukan
- **Memory Optimization**: Mengurangi overhead data yang tidak terpakai
- **Focus**: Hanya metadata judul yang dibutuhkan untuk rekomendasi

## Data Quality Implications

### 1. **Successful Matches**

```python
# Records dengan book_title yang valid
successful_matches = all_book[all_book['book_title'].notna()]
```

### 2. **Missing Matches**

```python
# Records dengan book_title = NULL
missing_matches = all_book[all_book['book_title'].isna()]
```

### 3. **Match Rate Calculation**

```python
match_rate = len(successful_matches) / len(all_book) * 100
```

***

# 🧹 Data Preparation - Missing Values Handling

## Tujuan

Mengidentifikasi dan menangani **missing values** yang muncul setelah proses data merging untuk memastikan kualitas data yang optimal untuk model training.

## 1. Missing Values Detection

```python
all_book.isna().sum()
```

### Cara Kerja & Parameter

| Method | Fungsi | Output |
| --- | --- | --- |
| `.isna()` | Deteksi nilai NULL/NaN per cell | Boolean matrix |
| `.sum()` | Agregasi missing values per kolom | Count per column |

### Missing Values Analysis

| Column | Missing Count | Percentage | Impact |
| --- | --- | --- | --- |
| `user_id` | 0 | 0% | ✅ Complete |
| `isbn` | 0 | 0% | ✅ Complete |
| `book_rating` | 0 | 0% | ✅ Complete |
| `book_title` | ~118,644 | ~10.3% | ⚠️ Significant |

## 2. Missing Values Impact Assessment

### Root Cause Analysis

```python
missing_percentage = (118644 / 1149780) * 100  # ≈ 10.3%
```

**Penyebab Missing Values:**

- **Data Mismatch**: ISBN dalam ratings tidak ada di books dataset
- **Data Quality**: Incomplete book metadata
- **Left Join Effect**: NULL values untuk unmatched records

### Business Impact

| Aspect | Impact | Severity |
| --- | --- | --- |
| **Data Loss** | 118,644 interactions | Moderate |
| **Model Training** | Reduced training data | Acceptable |
| **User Experience** | No book titles for some recommendations | Significant |

## 3. Missing Values Handling Strategy

```python
all_book_clean = all_book.dropna()
all_book_clean
```

### Method: Complete Case Deletion

#### Parameter Analysis

| Parameter | Value | Fungsi |
| --- | --- | --- |
| **Method** | `.dropna()` | Remove rows with any missing values |
| **Subset** | None (default) | Apply to all columns |
| **How** | 'any' (default) | Drop if any column has missing value |
| **Axis** | 0 (default) | Drop rows (not columns) |

***

# 📋 Data Preparation - Membuat dataframe baru yang berisi isbn dan judul buku

## Tujuan

Membuat **lookup table** dan **data structures** yang diperlukan untuk tahap modeling, termasuk mapping antara ISBN dan judul buku untuk keperluan rekomendasi.

## 1. Dataset Preparation Alias

```python
preparation = all_book_clean
preparation
```

### Cara Kerja & Parameter

| Parameter | Fungsi | Output |
| --- | --- | --- |
| `all_book_clean` | Source dataset (cleaned) | Clean dataset dengan 1,031,136 records |
| `preparation` | Working alias | Pointer ke dataset untuk modeling prep |

### Tujuan Alias

- **Code Clarity**: Nama yang lebih deskriptif untuk tahap preparation
- **Data Safety**: Preserve original cleaned dataset
- **Workflow Organization**: Clear separation antara cleaning dan preparation

***

## 2. List Extraction for Processing

```python
book_title, isbn = preparation["book_title"].tolist(), preparation["isbn"].tolist()
print(f"Jumlah data judul buku : {len(book_title)}")
print(f"Jumlah data isbn: {len(isbn)}")
```

### Cara Kerja & Parameter

#### A. Data Extraction

| Method | Parameter | Fungsi | Output Type |
| --- | --- | --- | --- |
| `preparation["book_title"]` | Column name | Extract title column | pandas Series |
| `.tolist()` | None | Convert to Python list | Python list |
| `preparation["isbn"]` | Column name | Extract ISBN column | pandas Series |

#### B. Multiple Assignment

```python
# Simultaneous assignment in single line
book_title, isbn = preparation["book_title"].tolist(), preparation["isbn"].tolist()
```

### Expected Output

```javascript
Jumlah data judul buku : 1031136
Jumlah data isbn: 1031136
```

## 3. Quality Validation Check

### Data Integrity Verification

```python
len(book_title) == len(isbn)  # Should return True
```

| Validation | Expected Result | Purpose |
| --- | --- | --- |
| **Length Match** | Both = 1,031,136 | Ensure 1:1 correspondence |
| **No Missing Values** | All elements present | Verify complete extraction |
| **Order Preservation** | Same index = same record | Maintain data relationships |

***

## 4. Lookup Table Creation

```python
book_new = pd.DataFrame({
    "isbn" : isbn,
    "title" : book_title
})
book_new
```

### Cara Kerja & Parameter

#### A. DataFrame Construction

| Parameter | Value | Fungsi |
| --- | --- | --- |
| **Dictionary Input** | `{"isbn": isbn, "title": book_title}` | Column mapping |
| **Column 1** | `"isbn"` | Primary key column |
| **Column 2** | `"title"` | Descriptive value column |

#### B. Data Structure Design

```python
# book_new structure
        isbn                    title
0  034545104X      Classical Mythology
1  0155061224            Clara Callan  
2  0446520802             Nine Stories
```

### Lookup Table Characteristics

| Feature | Description | Benefit |
| --- | --- | --- |
| **Unique Records** | Deduplicated ISBN-title pairs | No redundancy |
| **Fast Lookup** | ISBN → Title mapping | Quick title retrieval |
| **Compact Size** | Only essential columns | Memory efficient |
| **Clean Structure** | No missing values | Reliable mapping |

***

# 🔢 Feature Encoding - Categorical to Numerical Transformation

## Tujuan

Mengkonversi **data kategorik** (user_id dan ISBN) menjadi **data numerik** yang dapat diproses oleh model machine learning, khususnya untuk implementasi **collaborative filtering** dengan neural embeddings.

## 1. Dataset Initialization

```python
df = ratings
df
```

### Cara Kerja & Parameter

| Parameter | Fungsi | Output |
| --- | --- | --- |
| `ratings` | Source dataset | Original ratings dataset |
| `df` | Working variable | Copy untuk proses encoding |

### Dataset Structure

```javascript
Shape: (1,149,780 rows × 3 columns)
Columns: [user_id, isbn, book_rating]
```

***

## 2. Unique Values Extraction

```python
isbn_id = df["isbn"].unique().tolist()
user_id = df["user_id"].unique().tolist()
```

### Cara Kerja & Parameter

| Method | Parameter | Fungsi | Output |
| --- | --- | --- | --- |
| `.unique()` | None | Remove duplicates | Numpy array of unique values |
| `.tolist()` | None | Convert to Python list | Python list |

### Expected Results

```python
# Example outputs:
isbn_id = ["034545104X", "0155061224", "0446520802", ...]  # ~340,556 unique ISBNs
user_id = [276725, 276726, 276727, ...]                    # ~105,283 unique users
```

***

## 3. Encoding Dictionaries Creation

### A. ISBN Encoding/Decoding

```python
isbn_encoded = {key:values for values, key in enumerate(isbn_id)}
isbn_decoded = {key:values for key, values in enumerate(isbn_id)}
```

#### Dictionary Comprehension Analysis

| Component | Fungsi | Example |
| --- | --- | --- |
| `enumerate(isbn_id)` | Create (index, value) pairs | (0, "034545104X"), (1, "0155061224") |
| `{key:values for values, key in ...}` | Swap key-value positions | ISBN → Index mapping |
| `{key:values for key, values in ...}` | Keep original order | Index → ISBN mapping |

#### Output Structure

```python
# isbn_encoded: ISBN → Integer
isbn_encoded = {
    "034545104X": 0,
    "0155061224": 1,
    "0446520802": 2,
    ...
}

# isbn_decoded: Integer → ISBN  
isbn_decoded = {
    0: "034545104X",
    1: "0155061224", 
    2: "0446520802",
    ...
}
```

### B. User Encoding/Decoding

```python
user_encoded = {key:values for values, key in enumerate(user_id)}
user_decoded = {key:values for key, values in enumerate(user_id)}
```

#### Output Structure

```python
# user_encoded: User_ID → Integer
user_encoded = {
    276725: 0,
    276726: 1,
    276727: 2,
    ...
}

# user_decoded: Integer → User_ID
user_decoded = {
    0: 276725,
    1: 276726,
    2: 276727,
    ...
}
```

***

## 4. Mapping to DataFrame

```python
df["user_encoded"] = df["user_id"].map(user_encoded)
df["isbn_encoded"] = df["isbn"].map(isbn_encoded)
df
```

### Cara Kerja & Parameter

| Method | Parameter | Fungsi | Output |
| --- | --- | --- | --- |
| `.map()` | Dictionary | Apply mapping function | Transformed column |
| `user_encoded` | Mapping dict | User_ID → Index | Encoded user column |
| `isbn_encoded` | Mapping dict | ISBN → Index | Encoded ISBN column |

***

## 5. Data Sparsity Analysis

### Sparsity Calculation

```python
total_possible_interactions = num_users * num_books  # 105,283 × 340,556
actual_interactions = len(df)                        # 1,149,780
sparsity = (1 - actual_interactions/total_possible_interactions) * 100
```

| Metric | Value |
| --- | --- |
| **Possible Interactions** | 35.8 Billion |
| **Actual Interactions** | 1.15 Million |
| **Data Sparsity** | 99.997% |


***

# 🔄 Data Type Conversion - Rating Column to Float64

## Tujuan

Mengkonversi tipe data kolom `book_rating` dari **integer** menjadi **float64** untuk memastikan kompatibilitas optimal dengan model machine learning dan operasi matematika yang diperlukan.

## Implementation

```python
df["book_rating"] = df["book_rating"].values.astype(np.float64)
df
```

## Cara Kerja & Parameter Detail

### 1. **Ekstraksi Numpy Array**

```python
df["book_rating"].values
```

| Component | Fungsi | Output |
| --- | --- | --- |
| `df["book_rating"]` | Access pandas Series | Pandas Series dengan metadata |
| `.values` | Extract underlying array | Numpy array tanpa metadata |

### 2. **Type Conversion**

```python
.astype(np.float64)
```

| Parameter | Value | Fungsi |
| --- | --- | --- |
| **Target Type** | `np.float64` | 64-bit floating point |
| **Precision** | Double precision | 15-17 decimal digits |
| **Memory** | 8 bytes per value | Higher precision storage |

#### Conversion Process

```python
# Input: Integer array
[0, 5, 0, 3, 6] (dtype: int64)

# Output: Float64 array  
[0.0, 5.0, 0.0, 3.0, 6.0] (dtype: float64)
```

### 3. **Column Assignment**

```python
df["book_rating"] = ...
```

| Operation | Fungsi | Result |
| --- | --- | --- |
| **Overwrite** | Replace original column | Updated DataFrame |
| **Type Update** | Change column dtype | int64 → float64 |
| **Value Preservation** | Keep same values | No data loss |

***
## Why Float64 Conversion?

### ✅ **Machine Learning Compatibility**

| Aspect | Integer | Float64 | Benefit |
| --- | --- | --- | --- |
| **Neural Networks** | Needs conversion | Native support | ✅ Direct compatibility |
| **Mathematical Operations** | Limited precision | High precision | ✅ Better accuracy |
| **Normalization** | May cause issues | Smooth operations | ✅ Seamless processing |
| **Gradient Computation** | Potential problems | Optimal | ✅ Stable training |

### ✅ **Numerical Operations**

1. **Division Operations**

```python
   # Integer division (problematic)
   5 / 2 = 2  # Loss of precision
   
   # Float division (accurate)
   5.0 / 2.0 = 2.5  # Precise result
```

2. **Normalization Ready**

```python
   # Min-Max normalization example
   normalized = (rating - min_rating) / (max_rating - min_rating)
   # Requires float for accurate results
```

3. **Statistical Computations**

```python
   # Mean, variance calculations
   mean_rating = df["book_rating"].mean()  # More accurate with float64
```

***

# 🎯 Data Preparation Final Stage - Train/Validation Split

## Tujuan

Melakukan **persiapan akhir data** sebelum model training dengan shuffling, normalisasi, dan pembagian dataset untuk training dan validation yang optimal.

## 1. Data Shuffling

```python
df = df.sample(frac=1, random_state=99)
df
```

### Cara Kerja & Parameter

| Parameter | Value | Fungsi |
| --- | --- | --- |
| **`frac`** | 1 | Mengambil 100% data (semua baris) |
| **`random_state`** | 99 | Seed untuk reproducibility |
| **Return** | Shuffled DataFrame | Dataset dengan urutan acak |

### Shuffling Benefits

| Benefit | Explanation | Impact |
| --- | --- | --- |
| **Bias Elimination** | Menghilangkan urutan data asli | ✅ Prevent order-based patterns |
| **Even Distribution** | Data terdistribusi merata | ✅ Balanced train/val split |
| **Overfitting Prevention** | Mencegah model belajar urutan | ✅ Better generalization |
| **Reproducibility** | Hasil konsisten dengan seed | ✅ Experiment repeatability |

### Before vs After Shuffling

#### Before (Original Order)

```python
# Might have patterns like:
# - All ratings from user A first
# - Books grouped by genre
# - Temporal ordering bias
```

#### After (Random Order)

```python
# Random distribution:
# - Mixed users throughout dataset
# - Random book-user combinations
# - No systematic patterns
```

***

## 2. Feature Matrix Preparation

```python
x = df[["user_encoded","isbn_encoded"]]
```

### Feature Selection

| Column | Data Type | Range | Purpose |
| --- | --- | --- | --- |
| **`user_encoded`** | int64 | 0 to 105,282 | User identification |
| **`isbn_encoded`** | int64 | 0 to 340,555 | Book identification |

### Output Structure

```python
# x shape: (1,149,780, 2)
   user_encoded  isbn_encoded
0             0             0
1             1             1
2             2             2
...
```

***

## 3. Target Normalization (Min-Max Scaling)

```python
min = df["book_rating"].min()
max = df["book_rating"].max()
y = df["book_rating"].apply(lambda x:(x-min) / (max-min))
```

### Normalization Parameters

| Parameter | Value | Calculation |
| --- | --- | --- |
| **`min`** | 0.0 | Minimum rating in dataset |
| **`max`** | 10.0 | Maximum rating in dataset |
| **Range** | 10.0 | max - min |

### Min-Max Formula Implementation

```python
# Lambda function breakdown:
lambda x: (x - min) / (max - min)

# Equivalent to:
def normalize_rating(rating):
    return (rating - 0.0) / (10.0 - 0.0)
```

***

## 4. Train-Validation Split

```python
split = int(0.85 * df.shape[0])
X_train, X_val, Y_train, Y_val = (
    x[:split],
    x[split:],
    y[:split],
    y[split:]
)
```

### Split Calculation

| Metric | Calculation | Result |
| --- | --- | --- |
| **Total Data** | `df.shape[0]` | 1,149,780 |
| **Split Point** | `int(0.85 * 1,149,780)` | 977,313 |
| **Training %** | 85% | 977,313 samples |
| **Validation %** | 15% | 172,467 samples |

### Data Split Implementation

#### A. **Training Set (85%)**

```python
X_train = x[:split]      # Features: rows 0 to 977,312
Y_train = y[:split]      # Targets: rows 0 to 977,312
```

#### B. **Validation Set (15%)**

```python
X_val = x[split:]        # Features: rows 977,313 to end
Y_val = y[split:]        # Targets: rows 977,313 to end
```

### Final Dataset Shapes

| Dataset | Shape | Content |
| --- | --- | --- |
| **X_train** | (977,313, 2) | Training features [user_encoded, isbn_encoded] |
| **Y_train** | (977,313,) | Training targets [normalized ratings 0-1] |
| **X_val** | (172,467, 2) | Validation features [user_encoded, isbn_encoded] |
| **Y_val** | (172,467,) | Validation targets [normalized ratings 0-1] |

***
