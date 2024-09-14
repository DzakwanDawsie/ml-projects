# Laporan Proyek Machine Learning - Dzakwan Dawsie

## Project Overview

Dalam era digital yang semakin berkembang, jumlah konten multimedia seperti anime terus meningkat dengan pesat. Platform streaming seperti Crunchyroll, Netflix, dan katalog anime MyAnimeList memiliki ribuan judul anime dari berbagai genre. Namun, dengan banyaknya pilihan, pengguna sering kali kesulitan untuk menemukan anime baru yang sesuai dengan preferensi mereka.

Masalah ini menciptakan kebutuhan akan sistem rekomendasi yang cerdas, yang mampu membantu pengguna dalam menemukan anime baru berdasarkan anime yang sudah mereka tonton atau sukai. Sistem rekomendasi menjadi semakin penting karena dapat meningkatkan pengalaman pengguna, mengurangi waktu pencarian, serta mempromosikan anime yang mungkin tidak diketahui sebelumnya.

Proyek ini ditujukan untuk merekomendasikan anime berdasarkan anime yang telah disukai atau ditonton oleh pengguna. Proyek ini menggunakan pendekatan Content-based Filtering untuk menganalisis fitur-fitur dari anime (seperti sinopsis, genre, jumlah member yang mengikuti, popularitas, dan score) dan merekomendasikan anime yang serupa dari katalog.

## Business Understanding
Dalam merekomendasikan anime yang sesuai dengan selera penonton sangatlah sulit, karena ada banyak faktor yang perlu diperhatikan untuk memberikan rekomendasi yang terbaik.

### Problem Statements
Penggemar anime sering kesulitan menemukan anime baru yang sesuai dengan selera mereka. Sistem rekomendasi ini akan membantu pengguna menemukan anime baru berdasarkan anime yang sudah mereka sukai.

### Goals
Memberikan rekomendasi anime yang mirip dengan preferensi pengguna.

**Rubrik/Kriteria Tambahan (Opsional)**:
### Solution statements
Menggunakan content-based filtering dengan mempertimbangkan atribut-atribut seperti genre, studio, dan sinopsis untuk memberikan rekomendasi anime yang mirip dengan preferensi pengguna.

## Data Understanding
Proyek ini menggunakan data katalog anime yang terdaftar antara tahun 1917 hingga 2020. Data ini bersumber dari kaggle dan dapat diunduh melalui link berikut: [Anime Dataset with Reviews - MyAnimeList](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews).

### Variabel-variabel pada Anime Dataset with Reviews - MyAnimeLis adalah sebagai berikut:
- uid: Merupakan unik id dari anime.
- title: Merupakan judul dari anime.
- synopsis: Merupakan sinopsis atau gambaran singkat mengenai alur dari anime.
- synopsis: Merupakan genre dari anime.
- aired: Merupakan tahun atau tanggal tayang dari anime.
- episodes: Merupakan jumlah episode dari anime.
- members: Merupakan banyaknya pengguna website MyAnimeList yang mengikuti anime tersebut.
- popularity: Merupakan urutan popularitas dari anime pada website MyAnimeList.
- ranked: Merupakan ranking dari anime pada website MyAnimeList.
- score: Merupakan skor dari anime pada website MyAnimeList, dalam skala 0-10.
- img_url: Merupakan URL gambar dari anime pada website MyAnimeList.
- link: Merupakan URL dari anime pada website MyAnimeList.

### Dari data tersebut diketahui bahwa:
- Total dari data berjumlah 19.311 baris dan 12 kolom.
- Terdapat nilai null pada kolom berikut.
  - `synopsis` sejumlah 975.
  - `episodes` sejumlah 706.
  - `ranked` sejumlah 3212.
  - `score` sejumlah 579.
  - `img_url` sejumlah 180.
- Nilai unik berdasarkan kolom `uid` hanya berjumlah 16.216. Itu artinya, terdapat 3.095 baris data yang duplikat.
- Pada kolom `aired` terdapat nilai 'Not available' sejumlah 372 baris.

### Grafik Perbandingan
![Bar Chart Null Data](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/bar-chart-null-data.png)
![Pie Chart Unique Data](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/pie-chart-unique-data.png)
![Pie Chart Not Available Data](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/pie-chart-na-data.png)

## Data Preparation
Data preparation dimulai dari melakukan cleansing data, hingga melakukan Feature Encoding, dengan tahapan sebagai berikut:

1. Hapus baris `synopsis` yang bernilai null

   Karena `synopsis` diperlukan untuk proses feature encoding, maka untuk data dengan `synopsis` bernilai nullharus dihapus.
3. Hapus kolom `episodes`

   Karena kolom `episodes` tidak diperlukan dalam proses modeling, maka kolom tersebut perlu dihapus, agar hasil filtering menjadi maksimal
5. Replace kolom `ranked` yang bernilai null dengan nilai maksimum dari `ranked`

   Kolom `ranked` perlu direplace dengan nilai maksimum dari `ranked` yaitu 18336. Dengan anggapan, anime tanpa ranking akan memasuki urutan terbawah.
7. Replace kolom `score` yang bernilai null dengan nilai rata-rata dari `score`

   Karena kolom `score` diperlukan untuk proses modeling, maka perlu direplace dengan nilai rata-rata, supaya tidak merusak hasil rekomendasi.
9. Menghapus kolom `img_url` dan `link` dan `aired` karena tidak diperlukan.
10. Mentransformasikan nilai pada kolom `genre` menjadi kolom tersendiri.
11. Hapus kolom `genre` setelah transformasi.
12. Menghapus data yang memiliki `synopsis` selain huruf alphabet.
14. Melakukan beberapa teknik preprocessing text terhadap kolom `synopsis` dan membuatkannya kolom baru, yaitu `synopsis_clean`. Seperti teknik cleaning text, casefolding text, tokenizing text, dan juga filtering text. Kemudian merubahnya kembali ke bentuk kalimat.
15. Hapus kolom `synopsis` setelah preprocessing text.
16. Melakukan feature encoding menggunakan TF-ID.
17. Menghapus `feature` jika hanya dimiliki oleh 1 baris data.
18. Menggabungkan `feature` hasil encoding dengan DataFrame `anime_df` tanpa kolom `title` dan `synopsis_clean`.

## Modeling & Result
Model ini dibangun dengan menggunakan algoritma Cosine Similarity. Yaitu, sebuah algoritma yang mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity. 

![Illustration](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:87157b70a8c08f2bb7b464f190fef8fa20210910171725.jpeg)

Metrik ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks. Sebagai contoh, dalam studi kasus ini, cosine similarity digunakan untuk mengukur kesamaan nama restoran dan nama masakan.

Cosine similarity dirumuskan sebagai berikut.

![Formula](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:784efd3d2ba47d47153b050526150ba920210910171725.jpeg)

Hasil:

![Predict Result](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/predict-result.png)

## Evaluation
Dengan cosine similarity, diketahui kesamaan antara satu anime dengan anime lainnya. Shape (18336, 18336) merupakan ukuran matriks similarity dari data yang kita miliki. Berdasarkan data yang ada, matriks di atas sebenarnya berukuran 18336 restoran x 18336 anime (masing-masing dalam sumbu X dan Y). Artinya, telah di-identifikasi tingkat kesamaan pada 18336 judul anime. Tapi tentu hal ini tidak bisa ditampilkan semuanya. Oleh karena itu, dipilih 10 anime pada baris vertikal dan 5 anime pada sumbu horizontal seperti pada contoh di bawah. 

![Evaluation](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/cosine-sim-df-new.png)

Tabel cosine similarity menunjukkan tingkat kemiripan antar anime berdasarkan fitur-fitur yang diekstrak.
Nilai dalam tabel berkisar antara 0 hingga 1, di mana:
- Nilai 1 menunjukkan bahwa dua anime tersebut sangat mirip.
- Nilai 0 menunjukkan bahwa dua anime tersebut tidak memiliki kemiripan.

Tabel ini digunakan untuk merekomendasikan anime yang mirip dengan anime yang disukai pengguna. 
Misalnya, jika pengguna menyukai anime "Gunslinger Stratos The Animation: Bunki/Futatsu no Mirai", maka sistem akan merekomendasikan anime yang memiliki nilai cosine similarity yang mendekati nilai 1, misalnya "Sakana no Kuni" dengan kemiripan 86% atau "Shinya! Tensai Bakabon" dengan kemiripan 99%

Dengan kata lain, tabel ini membantu dalam menemukan anime yang memiliki karakteristik serupa, genre yang sama, atau plot yang mirip dengan anime yang sudah ditonton.

#### Metrik Evaluasi
Metrik `precision` digunakan untuk mengukur seberapa tepat sebuah model dalam membuat prediksi positif. Dalam konteks klasifikasi biner, precision menghitung proporsi prediksi positif yang benar dari semua prediksi positif yang dihasilkan oleh model.

Rumus precision adalah:

![Rumus Precision](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/rumus-precision.png)

Pada kasus ini
![Hasil Precision](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/hasil-metric.png)
![Hitungan Precision](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/cbf-anime-recomendation/hitungan-metric.png)

TP = 9

FP = 10

Sehingga nilai precision nya adalah 0.47

#### Kesimpulan
Meskipun sulit untuk menemukan kemiripan yang benar-benar tepat, hasil dari penggunaan cosine similarity dapat memberikan rekomendasi yang akurat dengan mempertimbangkan beberapa variabel, seperti sinopsis dan genre yang menjadi bobot penilaian terbanyak. Solusi ini berdampak signifikan karena memungkinkan para penggemar anime mendapatkan rekomendasi yang lebih baik berdasarkan anime yang mereka sukai, tanpa perlu bersusah payah membandingkan banyak anime secara manual.

**---Ini adalah bagian akhir laporan---**
