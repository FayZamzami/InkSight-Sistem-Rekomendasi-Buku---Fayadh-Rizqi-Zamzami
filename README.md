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

Standardisasi ini memastikan **data consistency** yang sangat penting untuk **maintainability** dan **reliability** dalam seluruh pipeline data science project.

***
