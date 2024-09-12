# Laporan Proyek Machine Learning - Dzakwan Dawsie

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
Data dibersihkan dengan menghapus kolom yang tidak diperlukan, dan juga baris yang memiliki nilai null. Dan juga pengkonversian beberapa kolom menjadi sebuah kolom baru.

**Rubrik/Kriteria Tambahan (Opsional)**: 

Tahapan dimulai dengan menghapus baris yang memiliki nilai null. 

![Remove Null Data](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/remove-null-data.png)

Setelah itu kolom yang tidak diperlukan seperti `id` dan `date` dihapus agar tidak merusak hasil train data.

![Drop Column](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/drop-column.png)

Setelah mengetahui fakta bahwa "Kebanyakan rumah yang berada di tengah-tengah kota, memiliki harga yang relatif mahal". Selanjutnya kolom `lat` dan `long` dikonversikan menjadi kolom `distance_to_center`, menggunakan rumus Haversine:
```
a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
c = 2 ⋅ atan2( √a, √(1−a) )
d = R ⋅ c
```
Yang dimana:
- φ adalah latitude, λ adalah longitude
- Δφ = φ2 − φ1
- Δλ = λ2 − λ1
- R adalah radius bumi (rata-rata 6.371 km)

Perhitungan dilakukan dengan bantuan library `geodesic` dari python.

Setelah itu, data dipisah menjadi data `train` dan `test` dengan menggunakan fungsi `train_test_split` dari library `sklearn`. Pemisahan dilakukan dengan perbandingan `80% data train` dan `20% data test`.

Setelah data berhasil dipisah. Kemudian data `X_train` dan `X_test` ditransformasikan menggunakan **StandarScaler**.

## Modeling
Data dilatih dengan menggunakan model Deep Learning Sequential. Yaitu, sebuah model dari Keras yang menghubungkan lapisan-lapisan (layers) secara berurutan. Ini berarti bahwa output dari satu layer menjadi input untuk layer berikutnya. Model ini sangat cocok untuk arsitektur model yang linear, di mana lapisan-lapisan saling terhubung secara berurutan.

Selain menggunakan model Deep Learning Sequential. model ini juga menggunakan metrik evaluasi Mean Squared Error (MSE). Yaitu sebuah metrik yang mengukur rata-rata dari kuadrat perbedaan antara nilai yang diprediksi oleh model dan nilai aktual (ground truth). Metrik ini memberikan informasi tentang seberapa jauh prediksi model dari nilai yang sebenarnya.

Tahapan modeling dimulai dari perancangan model machine learning, yang terdiri dari layer sebagai berikut:

- Sequential
  - Input Layer
    - Dense 1
  - Hidden Layer 
    - Dropout 1
    - Dense 2
    - Dropout 2
    - Dense 3
    - Dropout 3
    - Dense 4
  - Output Layer
    - Dense 5

Model dibangun dengan hyperparameter sebagai berikut:

- Jumlah Neuron per Layer:
  - Dense Layer 1: 128 neuron
  - Dense Layer 2: 64 neuron
  - Dense Layer 3: 32 neuron
  - Dense Layer 4: 16 neuron
  - Dense Layer 5: 1 neuron

- Activation Function:
  - ReLU (Rectified Linear Unit): Digunakan pada layer-layer Dense, kecuali layer output.

- Dropout Rate:
  - Dropout Layer 1: 0.2 (20%)
  - Dropout Layer 2: 0.2 (20%)
  - Dropout Layer 3: 0.2 (20%)
  - Dropout digunakan untuk mengurangi overfitting dengan secara acak mematikan neuron selama pelatihan.

- Optimizer: Adam (yaitu, algoritma optimisasi yang digunakan untuk memperbarui bobot model selama pelatihan. Adam mengadaptasi learning rate berdasarkan estimasi momen pertama dan kedua dari gradien).

- Loss Function: Mean Squared Error (MSE) yang digunakan untuk menghitung seberapa jauh prediksi model dari nilai aktual.

- Epochs: Jumlah Epoch: 375 epoch (Epoch mengacu pada jumlah iterasi model melalui seluruh dataset selama pelatihan).

Setelah model terbuat, model tersebut kemudian dicompile dengan menggunakan *loss function* `mean_squared_error` dan *optimizer* `adam`.

Dan setelah itu data dilatih dengan jumlah epoch sebanyak 375.

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
