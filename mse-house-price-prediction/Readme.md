# Laporan Proyek Machine Learning - Dzakwan Dawsie

## Domain Proyek

Pasar properti di Seattle, Amerika Serikat, memiliki harga yang tinggi, terutama di area yang dekat dengan pusat kota. Banyak faktor yang mempengaruhi harga rumah, seperti ukuran rumah, jumlah kamar tidur dan kamar mandi, kondisi rumah, serta lokasi strategis. Analisis harga properti menjadi penting bagi pembeli dan penjual untuk memahami nilai pasar yang tepat.

Proyek ini bertujuan untuk memprediksi harga rumah di Seattle menggunakan algoritma machine learning. Data historis rumah di wilayah tersebut dianalisis dengan mempertimbangkan berbagai fitur, seperti ukuran rumah, jumlah lantai, pemandangan, serta jarak dari pusat kota. Untuk meningkatkan akurasi, fitur-fitur tersebut dinormalisasi menggunakan StandarScaler, dan kinerja model diukur menggunakan Mean Squared Error (MSE).

Hasil awal menunjukkan bahwa salah satu faktor utama yang mempengaruhi harga rumah adalah jaraknya dari pusat kota, di mana rumah yang lebih dekat cenderung lebih mahal. Dengan model prediksi ini, diharapkan dapat memberikan wawasan yang lebih baik bagi para pelaku pasar properti di Seattle.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.
Dalam memprediksi harga rumah di Seattle, beberapa tantangan utama perlu diidentifikasi dan dijelaskan untuk memperjelas masalah yang dihadapi.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Harga rumah yang bervariasi di Seattle tergantung pada banyak faktor, namun sulit memahami faktor yang paling berpengaruh.
- Sulitnya menemukan prediksi harga yang tepat untuk rumah yang berada di Seattle.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengidentifikasi faktor kunci yang memengaruhi harga rumah berdasarkan data historis.
- Mengukur pengaruh jarak ke pusat kota secara kuantitatif dengan analisis machine learning untuk mengetahui dampaknya pada harga.


**Rubrik/Kriteria Tambahan (Opsional)**:
### Solution statements
- Membuat visualisasi penyebaran rumah menggunakan Scatter Plot untuk mengidentifikasi faktor jarak rumah ke pusat kota Seattle.
- Menggunakan model Deep Learning Sequential dengan metrik evaluasi Mean Squared Error (MSE) untuk prediksi harga rumah.

## Data Understanding
Proyek ini menggunakan data historis rumah yang terjual antara mei 2014 hingga mei 2015. Data ini bersumber dari kaggle dan dapat diunduh melalui link berikut: [kc_house_data](https://www.kaggle.com/datasets/shivachandel/kc-house-data/data).

### Variabel-variabel pada kc_house_data adalah sebagai berikut:
- id: Merupakan id dari baris data.
- date: Merupakan tanggal terjualnya rumah.
- price: Merupakan harga dari rumah yang terjual, digunakan sebagai label target yang ingin diprediksi.
- bedrooms: Merupakan jumlah kamar tidur yang terdapat di rumah.
- bathrooms: Merupakan jumlah kamar mandi di rumah, termasuk kamar mandi setengah.
- sqft_living: Merupakan luas area hunian dalam kaki persegi, yaitu ruang yang dapat digunakan untuk aktivitas sehari-hari di rumah.
- sqft_lot: Merupakan luas total lot atau tanah tempat rumah berdiri dalam kaki persegi.
- floors: Merupakan jumlah lantai di rumah.
- waterfront: Merupakan indikator apakah rumah berada di tepi laut (1 jika ya, 0 jika tidak).
- view: Merupakan skor yang menunjukkan seberapa baik pemandangan dari rumah, dalam skala 0-4.
- condition: Merupakan skor yang menggambarkan kondisi keseluruhan rumah, dalam skala 1-5.
- grade: Merupakan skor yang menunjukkan kualitas bangunan dan desain rumah, dalam skala 1-13.
- sqft_above: Merupakan luas area di atas tanah (tidak termasuk basement) dalam kaki persegi.
- sqft_basement: Merupakan luas basement dalam kaki persegi.
- yr_built: Merupakan tahun rumah dibangun.
- yr_renovated: Merupakan tahun terakhir rumah direnovasi. Jika rumah tidak pernah direnovasi, maka nilainya 0.
- zipcode: Merupakan kode pos area rumah.
- lat: Merupakan latitude dari koordinat rumah.
- long: Merupakan longitude dari koordinat rumah.
- sqft_living15: Merupakan luas area hunian dalam kaki persegi pada 15 rumah terdekat.
- sqft_lot15: Merupakan luas lot (tanah) dalam kaki persegi pada 15 rumah terdekat.

**Rubrik/Kriteria Tambahan (Opsional)**:

Selanjutnya data divisualisasikan dalam bentuk scatter plot, dengan x adalah `longitude` dan y adalah `latitude`. Dan dikelompokan berdasarkan `price`. Setelah itu dibandingkan secara *side by side* dengan peta map asli Seattle.
![Scatter Plot And Real Map Seattle](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/seattle-scatter-map.png)
Dari perbandingan antara Scatter Plot dan Map Asli Seattle ini, diketahui bahwa:

**Kebanyakan rumah yang berada di tengah-tengah kota, memiliki harga yang relatif mahal**

## Data Preparation
Data dibersihkan dengan menghapus kolom yang tidak diperlukan, dan juga baris yang memiliki nilai null. Dan juga pengkonversian beberapa kolom menjadi sebuah kolom baru.

**Rubrik/Kriteria Tambahan (Opsional)**: 

Tahapan dimulai dengan menghapus baris yang memiliki nilai null. Dan juga menghapus data yang duplikat. Setelah itu kolom yang tidak diperlukan seperti `id` dan `date` dihapus agar tidak merusak hasil train data.

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

## Modeling
Data dilatih dengan menggunakan model Deep Learning Sequential, dengan metrik evaluasinya adalah Mean Squared Error (MSE). Tahapan modeling dimulai dari perancangan model machine learning, yang terdiri dari layer sebagai berikut:
- Sequential
  - Dense (128 neuron, activation dengan relu, dan input_shape adalah fitur/kolom dari dataset)
  - Dropout (rate 0.2)
  - Dense (64 neuron, activation dengan relu)
  - Dropout (rate 0.2)
  - Dense (32 neuron, activation dengan relu)
  - Dropout (rate 0.2)
  - Dense (16 neuron, activation dengan relu)
  - Dense (1 neuron)

Kemudian, data model di-compile dengan menggunakan *loss function* `mean_squared_error` dan *optimizer* `adam`.

Dan setelah itu data dilatih dengan jumlah epoch 375.

## Evaluation
Dengan penggunaan metrik Mean Squared Error (MSE) pada model machine learning ini. Didapatkan hasil pelatihan sebagai berikut (yang ditampilkan dalam bentuk *line plot*):

![Training Result](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/train-result.png)

Pelatihan dimulai dengan nilai dari `loss` dan `val_loss` adalah: loss: 23867654144.0000 - val_loss: 21040510976.0000

Dan diakhiri dengan nilai dari `loss` dan `val_loss` adalah: loss: 22257076224.0000 - val_loss: 21322461184.0000

Selanjutnya model diuji dengan memasukan sampel data yaitu data baris pertama pada dataset. Dan didapatkan hasil sebagai berikut:
![Predict Result](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/first-row-prediction.png)

**---Ini adalah bagian akhir laporan---**
