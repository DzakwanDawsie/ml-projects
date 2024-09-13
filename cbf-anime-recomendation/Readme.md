![image](https://github.com/user-attachments/assets/39bdafaa-320a-405d-b6ca-68cce87ddcfb)# Laporan Proyek Machine Learning - Dzakwan Dawsie

## Domain Proyek

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
12. Melakukan beberapa teknik preprocessing text terhadap kolom `synopsis` dan membuatkannya kolom baru, yaitu `synopsis_clean`. Seperti teknik cleaning text, casefolding text, tokenizing text, dan juga filtering text. Kemudian merubahnya kembali ke bentuk kalimat.
13. Hapus kolom `synopsis` setelah preprocessing text.
14. Melakukan feature encoding menggunakan TF-ID.
15. Menggabungkan `feature` hasil encoding dengan DataFrame `anime_df` tanpa kolom `title` dan `synopsis_clean`.

## Modeling & Result
Model ini dibangun dengan menggunakan algoritma Cosine Similarity. Yaitu, sebuah algoritma yang mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity. 

![Illustration](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:87157b70a8c08f2bb7b464f190fef8fa20210910171725.jpeg)

Metrik ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks. Sebagai contoh, dalam studi kasus ini, cosine similarity digunakan untuk mengukur kesamaan nama restoran dan nama masakan.

Cosine similarity dirumuskan sebagai berikut.

![Formula](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:784efd3d2ba47d47153b050526150ba920210910171725.jpeg)

Hasil prediksi:


Dengan cosine similarity, diketahui kesamaan antara satu anime dengan anime lainnya. Shape (18336, 18336) merupakan ukuran matriks similarity dari data yang kita miliki. Berdasarkan data yang ada, matriks di atas sebenarnya berukuran 18336 restoran x 18336 anime (masing-masing dalam sumbu X dan Y). Artinya, telah di-identifikasi tingkat kesamaan pada 18336 judul anime. Tapi tentu hal ini tidak bisa ditampilkan semuanya. Oleh karena itu, dipilih 10 anime pada baris vertikal dan 5 anime pada sumbu horizontal seperti pada contoh di atas. 

## Evaluation
Dengan penggunaan metrik Mean Squared Error (MSE) pada model machine learning ini. Didapatkan hasil pelatihan sebagai berikut (yang ditampilkan dalam bentuk *line plot*):

![Training Result](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/train-result.png)

Pelatihan dimulai dengan nilai dari `loss` dan `val_loss` adalah: loss: 23867654144.0000 - val_loss: 21040510976.0000

Dan diakhiri dengan nilai dari `loss` dan `val_loss` adalah: loss: 22257076224.0000 - val_loss: 21322461184.0000

Hasil ini menunjukkan bahwa selama pelatihan, model mengalami penurunan dalam nilai `loss`, yang menunjukkan bahwa model belajar untuk memprediksi dengan lebih baik. Namun, ada sedikit peningkatan pada `val_loss` pada akhir pelatihan, yang mungkin mengindikasikan sedikit overfitting.

Selanjutnya, model diuji dengan memasukan sampel data yaitu data baris pertama pada dataset. Dan didapatkan hasil sebagai berikut:
![Predict Result](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/first-row-prediction.png)

Hasil ini menunjukkan bahwa di antara nilai prediksi dan nilai aktual, terdapat *gap* atau jarak yang diukur dalam nilai MSE.

#### Kesimpulan
Dari proyek machine learning ini didapati hasil akhir bahwa:
- Berdasarkan analisis data dan model machine learning, faktor yang paling berpengaruh dalam kenaikan harga rumah di Seattle adalah **dekatnya jarak dari rumah menuju ke pusat kota**.
- Meskipun sulit untuk menemukan prediksi harga rumah yang tepat, hasil dari model machine learning ini dapat memberikan estimasi harga yang lebih akurat dibandingkan metode tradisional. Metrik MSE digunakan untuk mengevaluasi kinerja model dan memberikan wawasan tentang akurasi prediksi model.
- Solusi ini berdampak signifikan karena memungkinkan pengambilan keputusan yang lebih baik dalam hal harga jual atau beli rumah, perencanaan investasi, dan strategi pasar. Dengan estimasi harga yang lebih akurat dari model machine learning, pengguna dapat membuat keputusan yang lebih terinformasi dan strategis.

**---Ini adalah bagian akhir laporan---**
