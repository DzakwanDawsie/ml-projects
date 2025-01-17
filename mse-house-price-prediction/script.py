# -*- coding: utf-8 -*-
"""DCD_MLT_Proyek 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uyNdzkW2otwKC9Q2AsQSHAKuWTyCHcVf

# Proyek Prediksi Harga Rumah
- **Nama:** Dzakwan Dawsie
- **Email:** d.dawsie136@gmail.com
- **ID Dicoding:** zack01

## Import Semua Packages/Library yang Digunakan
"""

!pip install requests

!pip install tensorflowjs

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import regularizers

import tensorflowjs as tfjs

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Another libraries
from geopy.distance import geodesic
import pandas as pd
import seaborn as sns
import zipfile,os, shutil, time
import numpy as np
import glob
import requests
from PIL import Image
from google.colab import files
from io import BytesIO

"""## Data Preparation

### Data Loading

Mengupload dataset csv melalui form dibawah ini
"""

uploaded = files.upload()
filenames = list(uploaded.keys())
filename = filenames[0]

print(f"Uploaded file: {filename}")

"""Membuka dataset menggunakan pandas dan menjadikannya DataFrame"""

house_df = pd.read_csv(filename)

"""Cek jumlah baris dan kolom pada DataFrame

*jika csv sesuai, maka jumlah baris dan kolomnya adalah 21.613 baris x 21 kolom*
"""

jumlah_ulasan, jumlah_kolom = house_df.shape

print(f"Jumlah baris: {jumlah_ulasan}")
print(f"Jumlah kolom: {jumlah_kolom}")

"""Memastikan isian data dengan menampilkan data 5 baris teratas"""

house_df.head()

"""### Data Preprocessing

#### Data Cleansing

Mengecek apakah ada data yang null
"""

house_df.isnull().sum()

"""Dari pengecekan diatas, diketahui bahwa terdapat 2 baris data yang null pada kolon `sqft_above`

Maka, selanjutnya baris *null* tersebut akan dibersihkan
"""

clean_house_df = house_df.dropna()
clean_house_df.isnull().sum()

"""Memastikan data bersih dengan menghapus data yang duplikat"""

jumlah_ulasan_setelah_hapus_duplikat, jumlah_kolom_setelah_hapus_duplikat = clean_house_df.shape

print(f"Jumlah baris sebelum menghapus duplikat: {jumlah_ulasan_setelah_hapus_duplikat}")

clean_house_df = clean_house_df.drop_duplicates()

jumlah_ulasan_setelah_hapus_duplikat, jumlah_kolom_setelah_hapus_duplikat = clean_house_df.shape

print(f"Jumlah baris setelah menghapus duplikat: {jumlah_ulasan_setelah_hapus_duplikat}")

"""Menghapus kolom yang tidak diperlukan dalam proses training.

Yaitu kolom `id` dan `date`
"""

clean_house_df = clean_house_df.drop('id', axis='columns')
clean_house_df = clean_house_df.drop('date', axis='columns')

"""#### Data Understanding

Menampilkan persebaran data
"""

plt.figure(figsize=(10, 8))
sns.scatterplot(data=clean_house_df, x=clean_house_df['long'], y=clean_house_df['lat'])

"""Pada scatter plot diatas, diketahui bahwa persebaran data membentuk suatu daerah.

Yaitu pusat kota Seattle, Washington DC, Amerika Serikat

![picture](https://drive.google.com/uc?id=1bt_PJeTJqjqe499ul5RkdLdNpkP4lvoL&authuser=0)

Mengelompokan scatter plot dengan menjadikan harga rumah sebagai hue
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(data=clean_house_df, x=clean_house_df['long'], y=clean_house_df['lat'], hue=clean_house_df['price'], ax=ax1)

url = "https://drive.google.com/uc?id=1bt_PJeTJqjqe499ul5RkdLdNpkP4lvoL&authuser=0"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
ax2.imshow(img)
ax2.axis('off')

"""Berdasarkan scatter plot diatas. Diketahui bahwa:

> Kebanyakan rumah yang berada di tengah-tengah kota, memiliki harga yang relatif mahal

Itu artinya, jarak rumah antara pusat kota menjadi faktor mahalnya sebuah rumah

Konversi long & lat menjadi `distance` ke pusat kota
"""

center_lat = 47.6062
center_lon = -122.3321

def calculate_distance(lat, lon):
  house_coords = (lat, lon)
  center_coords = (center_lat, center_lon)
  distance = geodesic(house_coords, center_coords).km
  return distance

clean_house_df['distance_to_center'] = clean_house_df.apply(lambda row: calculate_distance(row['lat'], row['long']), axis=1)
clean_house_df = clean_house_df.drop(['long', 'lat'], axis=1)

clean_house_df.info()

"""#### Data Splitting

Membuat kolom `price` sebagai label dan memisahkannya dari kolom lainnya
"""

X = clean_house_df.drop('price', axis='columns').values
y = clean_house_df['price'].values

"""Memecah data menjadi `80% train data` dan `20% test data`"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Mengaplikasikan Standar Scaler"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""## Modelling

Membuat model deep learning menggunakan algoritma Sequential
"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

"""Melatih model"""

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=375)

losses = pd.DataFrame(model.history.history)
plt.plot(losses)

"""### Evaluation

Membuat data test menggunakan data row pertama dari data rumah
"""

test_house = clean_house_df.drop('price', axis='columns').iloc[0]
test_house = scaler.transform(test_house.values.reshape(-1, 17))
test_house

"""Prediksi harga dan bandingkan dengan harga asli, dan juga nilai mse"""

predictions = model.predict(test_house)

predicted_price = predictions[0][0]
actual_price = clean_house_df.iloc[0].price
mse = mean_squared_error([actual_price], [predicted_price])

print(f"Predicted Price: {predicted_price}")
print(f"Actual Price: {actual_price}")
print(f"MSE: {mse}")