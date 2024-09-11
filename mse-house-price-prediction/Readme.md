# Laporan Proyek Machine Learning - Dzakwan Dawsie

## Domain Proyek

Pasar properti di Seattle, Amerika Serikat, memiliki harga yang tinggi, terutama di area yang dekat dengan pusat kota. Banyak faktor yang mempengaruhi harga rumah, seperti ukuran rumah, jumlah kamar tidur dan kamar mandi, kondisi rumah, serta lokasi strategis. Analisis harga properti menjadi penting bagi pembeli dan penjual untuk memahami nilai pasar yang tepat.

Proyek ini bertujuan untuk memprediksi harga rumah di Seattle menggunakan algoritma machine learning. Data historis rumah di wilayah tersebut dianalisis dengan mempertimbangkan berbagai fitur, seperti ukuran rumah, jumlah lantai, pemandangan, serta jarak dari pusat kota. Untuk meningkatkan akurasi, fitur-fitur tersebut dinormalisasi menggunakan StandarScaler, dan kinerja model diukur menggunakan Mean Squared Error (MSE).

Hasil awal menunjukkan bahwa salah satu faktor utama yang mempengaruhi harga rumah adalah jaraknya dari pusat kota, di mana rumah yang lebih dekat cenderung lebih mahal. Dengan model prediksi ini, diharapkan dapat memberikan wawasan yang lebih baik bagi para pelaku pasar properti di Seattle.

## Business Understanding
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

Dari data tersebut diketahui terdapat 2 baris data yang null pada kolom `sqft_above`

![Null Data](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/preview-null-data.png)

Dan juga, dari data tersebut tidak ditemukan data yang duplikat

![Duplicate Data](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/duplicate-data.png)

**Rubrik/Kriteria Tambahan (Opsional)**:

Selanjutnya data divisualisasikan dalam bentuk scatter plot, dengan x adalah `longitude` dan y adalah `latitude`. Dan dikelompokan berdasarkan `price`. Setelah itu dibandingkan secara *side by side* dengan peta map asli Seattle.
![Scatter Plot And Real Map Seattle](https://raw.githubusercontent.com/DzakwanDawsie/ml-projects/main/mse-house-price-prediction/seattle-scatter-map.png)
Dari perbandingan antara Scatter Plot dan Map Asli Seattle ini, diketahui bahwa:

**Kebanyakan rumah yang berada di tengah-tengah kota, memiliki harga yang relatif mahal**

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
