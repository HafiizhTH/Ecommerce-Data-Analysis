# melihat versi dari setiap library

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

"""# Proyek Analisis Data: E-Commerce
- Nama: Hafiizh Taufiqul Hakim
- Email: 2012500720@student.budiluhur.ac.id
- Id Dicoding: hafizhtaufiqul1002

"""

"""## Menentukan Pertanyaan Bisnis

- Pertanyaan 1: Bagaimana kita dapat mengukur tingkat kepuasan pelanggan dalam bentuk persentase?
- Pertanyaan 2: Bagaimana distribusi status pesanan pelanggan, seperti pesanan yang sedang diproses, pesanan dalam pengiriman, pesanan yang telah selesai, hingga pesanan yang mengalami pembatalan?
- Pertanyaan 3: Produk apa saja yang memiliki penjualan terbanyak dan rating terbaik?
- Pertanyaan 4: Kota mana saja yang memiliki penjualan terbanyak?

"""


"""## Menyiapkan semua library yang dibutuhkan

Library yang saya gunakan sebagai berikut:
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Sklearn
- Streamlit

"""


"""## Data Wrangling

### Gathering Data
"""

data_customer = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\customers_dataset.csv')
data_geolocation = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\geolocation_dataset.csv')
data_order_items = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\order_items_dataset.csv')
data_order_payments = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\order_payments_dataset.csv')
data_order_reviews = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\order_reviews_dataset.csv')
data_orders = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\orders_dataset.csv')
data_product_category_name = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\product_category_name_translation.csv')
data_products = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\products_dataset.csv')
data_sellers = pd.read_csv(r'C:\Users\Siswantoro\Documents\Hafiizh\Dicoding\Tugas\ProyekAkhir_AnalisisData\dataset\sellers_dataset.csv')

data_products.sample(5)
data_product_category_name.sample(5)
data_order_items.sample(5)
data_orders.sample(5)
data_order_reviews.sample(5)
data_sellers.sample(5)

data_merge = pd.merge(
    left=data_products,
    right=data_product_category_name,
    how="inner",
    left_on="product_category_name",
    right_on="product_category_name"
)

data_merge = pd.merge(
    left=data_merge,
    right=data_order_items,
    how="inner",
    left_on="product_id",
    right_on="product_id"
)

data_merge = pd.merge(
    left=data_merge,
    right=data_orders,
    how="inner",
    left_on="order_id",
    right_on="order_id"
)

data_merge = pd.merge(
    left=data_merge,
    right=data_order_reviews,
    how="inner",
    left_on="order_id",
    right_on="order_id"
)

data_merge = pd.merge(
    left=data_merge,
    right=data_sellers,
    how="inner",
    left_on="seller_id",
    right_on="seller_id"
)

st.write(data_merge.sample(5))

"""**Observasi:**  
Saya menggabungkan 6 dataset untuk dapat dianalisa lebih lanjut.

dataset yang saya gunakan sebagai berikut:
- data_products
- data_product_category_name
- data_order_items
- data_orders
- data_order_reviews
- data_sellers

"""

"""### Assessing Data

##### Missing Value
"""

st.write(data_merge.isna().sum())

"""**Observasi:**  
Terdapat missing value pada dataset, sehingga perlu dilakukan cleanning data

"""

"""##### Duplicated"""

st.write(data_merge.duplicated().sum())

"""**Observasi:**  
Tidak terdapat Duplicated pada dataset, sehingga tidak perlu dilakukan cleaning data

"""

"""##### Outlier"""

num_outlier = data_merge.select_dtypes(include=['int64', 'float64']).columns.tolist()

plt.subplots(figsize=(20, 10))
for i in range(0,len(num_outlier)):
    plt.subplot(1, len(num_outlier), i+1)
    sns.boxplot(y=data_merge[num_outlier[i]], color='blue')
    plt.tight_layout()

"""**Observasi:**  
Terdapat banyak data yang mengalami outliers, sehingga perlu dilakukan cleaning data

"""

"""### Cleaning Data

##### Missing Value
"""

data_merge.isna().sum()

# Handle Missing Value untuk Tipe data object

cat_missing_value = ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'review_comment_title', 'review_comment_message']

for col in cat_missing_value:
    data_merge[col].fillna(data_merge[col].mode()[0], inplace=True)

# Handle Missing Value untuk Tipe data numeric

num_missing_value = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']

for col in num_missing_value:
    data_merge[col].fillna(data_merge[col].mean(), inplace=True)

st.write(data_merge.isna().sum())

"""**Observasi:**  
- Menggunakan `fillna()` untuk mengganti nilai yang missing value dengan nilai tertentu.
- Menggunakan `mode()` untuk nilai yang bertipe data kategorik
- Menggunakan `mean()` untuk nilai yang bertipe data numerik

"""

"""##### Outlier"""

num_outlier = data_merge.select_dtypes(include=['int64', 'float64']).columns.tolist()

plt.figure(figsize=(16, 8))
for i in range(2,len(num_outlier)):
    plt.subplot(1, len(num_outlier), i+1)
    sns.boxplot(y=data_merge[num_outlier[i]], color='blue')
    plt.tight_layout()

# Handling outliers dengan IQR
print(f'Jumlah baris sebelum memfilter outlier: {len(data_merge[num_outlier])}')

for col in data_merge[num_outlier]:
    Q1 = data_merge[col].quantile(0.25)
    Q3 = data_merge[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - (IQR * 1.5)
    upper_limit = Q3 + (IQR * 1.5)

    filtered_ouliers = ((data_merge[col] >= lower_limit) & (data_merge[col] <= upper_limit))

data_clean = data_merge[filtered_ouliers]

print('Jumlah baris setelah memfilter outlier', len(data_clean))

"""**Observasi:**  
- Menggunakan metode IQR untuk melakukan handle outlier
- Jumlah baris sebelum memfilter outlier terdapat 110750, kemudian jumlah baris berkurang menjadi 93580. karena metode IQR menghapus seluruh baris yang mengalamin outlier.

"""


"""## Exploratory Data Analysis (EDA)"""

data_clean.info()

cat = data_clean.select_dtypes(include=['object']).columns.tolist()
num = data_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()

"""##### Mengeksplorasi Parameter Statistik"""

# Metode describe() untuk data yang bertipe kategorik

data_clean[cat].describe()

# Menggunakan metode describe() untuk data yang bertipe numerik

data_clean[num].describe()

# Menggunakan metode corr() untuk memeriksa korelasi antar data numerik

data_clean[num].corr()

"""**Observasi:**  
- Menggunakan `describe()` dan `corr()` untuk melihat korelasi dan distribusi data numerik

"""

"""##### Mengelompokan Data"""

# product dengan penjualan terbanyak dan memiliki rating terbaik

best_product = data_clean.groupby(by='product_category_name_english').agg({'order_id': 'count',
                                                                          'review_score': 'mean'}).reset_index()

best_product = best_product.sort_values(by=['order_id', 'review_score'], ascending=[False, False]).head(10).reset_index()

best_product

# Kota dengan penjualan terbanyak

best_seller = data_clean.groupby(by='seller_city').agg({'order_id': 'count'}).reset_index()
best_seller = best_seller.sort_values(by='order_id', ascending=False).head(10).reset_index()

best_seller

"""**Observasi:**  
Membuat pengelompokan data sebagai berikut:
- Produk dengan penjualan terbanyak dan rating tertinggi menggunakan kolom ***product_category_name_english***, ***order_id***, dan ***review_score***
- Kota dengan penjualan terbanyak menggunakan kolom ***seller_city***, dan ***order_id***

"""


"""## Visualization & Explanatory Analysis"""

data_clean.info()

"""### Univariate Analysis

##### Pertanyaan 1:
Bagaimana kita dapat mengukur tingkat kepuasan pelanggan dalam bentuk persentase?

"""

review_score = data_clean['review_score'].value_counts()

review_score

review_score = data_clean['review_score'].value_counts()

labels = ['Sangat Puas', 'Puas', 'Kurang Puas', 'Kecewa', 'Sangat Kecewa']
colors = ['#1640D6', '#379237', '#E9B824', '#FF6C22', '#D80032']
sizes = review_score.values

fig, ax = plt.subplots(figsize=(20, 10))
ax.pie(x=sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 12})
ax.set_title('Distribusi Umpan Balik Pelanggan', fontsize=20)

st.pyplot(fig)

"""**Observasi:**  
Visualisasi ini memberikan gambaran yang jelas tentang tingkat kepuasan pelanggan. Mayoritas pelanggan menyatakan **Sangat Puas** dengan persentase sebanyak 56.2%. kemudian dilanjut dengan pelanggan yang menyatakan **Puas** dengan persentase sebanyak 19%. Meskipun ada sebagian besar respon positif, kita juga melihat bahwa sekitar 12.8% pelanggan merasa **Kurang Puas**, sementara pelanggan yang merasa **Kecewa** dan **Sangat Kecewa** jika dijumlahkan sebanyak 11%, walaupun mayoritas pelanggan merasa puas, masih perlu perhatian terhadap aspek-aspek yang membuat sebagian pelanggan merasa kurang puas atau kecewa.

"""

"""##### Pertanyaan 2:
Bagaimana distribusi status pesanan pelanggan, seperti pesanan yang sedang diproses, pesanan dalam pengiriman, pesanan yang telah selesai, hingga pesanan yang mengalami pembatalan?

"""

data_clean['order_status'].value_counts()

fig, ax = plt.subplots(figsize=(20, 10))
bars = sns.countplot(data=data_clean, x='order_status', orient='h', color='#3970F1', ax=ax)
for p in bars.patches:
    bars.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=20, color='black', xytext=(0, 5), textcoords='offset points')

ax.set_title('Distribusi Status Pesanan Pelanggan', fontsize=20)
ax.set_xlabel('Status Pesanan', fontsize=20)
ax.set_ylabel('Jumlah Pelanggan', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

st.pyplot(fig)

"""**Observasi:**  
Terdapat status proses pesanan sebagai berikut:
- Jumlah pesanan yang **delivered** sebanyak 91598
- Jumlah pesanan yang **shipped** sebanyak 953
- Jumlah pesanan yang **canceled** sebanyak 448
- Jumlah pesanan yang **unavailable** sebanyak 5
- Jumlah pesanan yang **processing** sebanyak 280
- Jumlah pesanan yang **invoiced** sebanyak 293
- Jumlah pesanan yang **approved** sebanyak 3

Namun, fokus perlu kita perhatikan kepada pelanggan yang melakukan **canceled** dengan jumlah sebanyak 448 pesanan dibatalkan. sehingga perlu strategi khusus untuk menurunkan pembatalan pesanan tersebut.

"""


"""### Multivariate Analysis

##### Pertanyaan 3:
Produk apa saja yang memiliki penjualan terbanyak dan rating terbaik?

"""

fig, ax = plt.subplots(figsize=(20, 10))
bars = ax.barh(best_product['product_category_name_english'],
                best_product['review_score'],
                color='#3970F1'
               )
for bar, order_count in zip(bars, best_product['order_id']):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(order_count), ha='center', va='center', fontsize=20)

ax.set_title('10 Produk dengan Penjualan Terbanyak dan Rating Terbaik', fontsize=20)
ax.set_xlabel('Jumlah Rating', fontsize=20)
ax.set_ylabel('Kategori Produk', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

ax.invert_yaxis()
st.pyplot(fig)


"""**Observasi:**  
Terdapat Sumbu Y yang menunjukkan kolom ***product_category_name_english*** dan sumbu X yang menunjukkan kolom ***review_score*** yang menggambarkan Rating/Umpan Balik dari pelanggan.

Berdasarkan data visualisasi terlihat bahwa produk dengan rating tertinggi terdapat pada produk **healty_beuty** dan **auto** dengan rating 4 sekian. tetapi untuk produk yang memiliki penjualan terbanyak ialah produk **bed_bath_table** dengan penjualan sebanyak 10286 dan memiliki rating 3.9.

"""

"""
##### Pertanyaan 4:
Kota mana saja yang memiliki penjualan terbanyak?

"""

fig, ax = plt.subplots(figsize=(20, 10))
ax.barh(best_seller['seller_city'],
         best_seller['order_id'],
         color='#3970F1'
         )

ax.set_title('10 Kota dengan Penjualan Terbanyak', fontsize=20)
ax.set_xlabel('Jumlah Order', fontsize=20)
ax.set_ylabel('Kota', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

ax.invert_yaxis()
st.pyplot(fig)


"""**Observasi:**  
Terdapat Sumbu Y yang menunjukkan kolom ***seller_city*** dan sumbu X yang menunjukkan kolom ***order_id*** yang menggambarkan jumlah penjualan.

Berdasarkan data visualisasi terlihat bahwa kota dengan penjualan terbanyak terdapat pada kota **sao paulo** dengan penjualan sebanyak 25000 sekian. Jika kita lihat bahwa distribusi penjualan mengalami perbedaan yang signifikan sehingga perlu strategi khusus untuk meningkatkan penjualan untuk kota-kota yang lain.

"""


"""## Conclusion

Saya akan memberikan beberapa rekomendasi atau solusi yang dapat digunakan oleh stakeholder

**Solusi Pertanyaan 1:**
- Stakeholder dapat meningkatkan kualitas produk atau layanan. seperti Melakukan riset pasar untuk mengetahui kebutuhan dan keinginan pelanggan, serta melakukan inovasi produk atau layanan secara berkala.
- stakeholder dapat meningkatkan komunikasi dengan pelanggan. seperti Meningkatkan kemudahan dalam menyampaikan keluhan kepada pelanggan, atau meningkatkan responsivitas terhadap keluhan pelanggan.

**Solusi Pertanyaan 2:**
- Mengumpulkan umpan balik dari pelanggan yang membatalkan pesanan. Mungkin dengan menyelenggarakan survei singkat atau menghubungi mereka langsung. Ini dapat memberikan insight langsung dan membantu meningkatkan proses pelayanan.
- Pastikan proses pengiriman berjalan lancar dan sesuai dengan harapan pelanggan. Informasi pelacakan yang akurat dan pengiriman tepat waktu dapat mengurangi kemungkinan pembatalan.

**Solusi Pertanyaan 3:**
- Meskipun produk kategori "health_beauty" dan "auto" memiliki rating tertinggi, perhatikan untuk tetap mempertahankan dan meningkatkan kualitas produk dalam kategori tersebut. Ini dapat menciptakan kepuasan pelanggan yang lebih besar dan membangun reputasi positif.
- Mengidentifikasi faktor-faktor yang menyebabkan rating menurun dan perbaiki masalah tersebut. Hal ini dapat membantu meningkatkan kepuasan pelanggan dan meningkatkan rating produk.
- Tinjau kembali strategi persediaan dan permintaan untuk memastikan bahwa produk dengan rating tinggi memiliki ketersediaan yang memadai. Jika ada kekurangan stok, ini dapat menyebabkan penurunan kepuasan pelanggan.

**Solusi Pertanyaan 4:**
- Lakukan analisis pasar untuk masing-masing kota dengan penjualan rendah. Pahami karakteristik demografis, preferensi pelanggan, dan tren pasar di setiap kota untuk merancang strategi yang sesuai.
- Sesuaikan penawaran produk dengan kebutuhan pelanggan. Produk atau layanan yang populer di Sao Paulo mungkin tidak sepopuler di kota lain, jadi perlu adaptasi.

"""

"""## Teknik Analisis Lanjutan

##### Metode Clustering

"""

feature = ['price', 'review_score']
X = data_clean[feature].values

data_clean[feature].describe()

X_std = StandardScaler().fit_transform(X)
data_model = pd.DataFrame(data = X_std, columns = feature)

data_model.head()

inertia = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_model.values)
    inertia.append(kmeans.inertia_)

kmeans


fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(x=range(2, 11), y=inertia, color='#3970F1', linewidth = 4, ax=ax)
sns.scatterplot(x=range(2, 11), y=inertia, s=300, color='black',  linestyle='--', ax=ax)

st.pyplot(fig)


st.caption('Copyright © Project Akhir - Hafiizh Taufiqul Hakim')

