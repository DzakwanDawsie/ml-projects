# Laporan Proyek Machine Learning - Dzakwan Dawsie

## Domain Proyek

Pasar properti di Seattle, Amerika Serikat, memiliki harga yang tinggi, terutama di area yang dekat dengan pusat kota. Banyak faktor yang mempengaruhi harga rumah, seperti ukuran rumah, jumlah kamar tidur dan kamar mandi, kondisi rumah, serta lokasi strategis. Analisis harga properti menjadi penting bagi pembeli dan penjual untuk memahami nilai pasar yang tepat.

Proyek ini bertujuan untuk memprediksi harga rumah di Seattle menggunakan algoritma machine learning. Data historis rumah di wilayah tersebut dianalisis dengan mempertimbangkan berbagai fitur, seperti ukuran rumah, jumlah lantai, pemandangan, serta jarak dari pusat kota. Untuk meningkatkan akurasi, fitur-fitur tersebut dinormalisasi menggunakan StandarScaler, dan kinerja model diukur menggunakan Mean Squared Error (MSE).

Hasil awal menunjukkan bahwa salah satu faktor utama yang mempengaruhi harga rumah adalah jaraknya dari pusat kota, di mana rumah yang lebih dekat cenderung lebih mahal. Dengan model prediksi ini, diharapkan dapat memberikan wawasan yang lebih baik bagi para pelaku pasar properti di Seattle.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.
Dalam memprediksi harga rumah di Seattle, beberapa tantangan utama perlu diidentifikasi dan dijelaskan untuk memperjelas masalah yang dihadapi.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Harga rumah yang bervariasi di Seattle tergantung pada banyak faktor, namun sulit memahami faktor yang paling berpengaruh
- Sulitnya menemukan prediksi harga yang tepat untuk rumah yang berada di Seattle

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengidentifikasi faktor kunci yang memengaruhi harga rumah berdasarkan data historis.
- Mengukur pengaruh jarak ke pusat kota secara kuantitatif dengan analisis machine learning untuk mengetahui dampaknya pada harga.

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

Selanjutnya data divisualisasikan dalam bentuk scatter plot, dengan x adalah `longitude` dan y adalah `latitude`. Dan dikelompokan berdasarkan `price`. Setelah itu dibandingkan secara *side by side* dengan peta map asli Seattle
**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
