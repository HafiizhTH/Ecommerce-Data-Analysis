# melihat versi dari setiap library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Dashboard Analisis Data E-commerce")

st.markdown(""" 
### Pertanyaan Bisnis
- Pertanyaan 1: Bagaimana kita dapat mengukur tingkat kepuasan pelanggan dalam bentuk persentase?
- Pertanyaan 2: Bagaimana distribusi status pesanan pelanggan, seperti pesanan yang sedang diproses, pesanan dalam pengiriman, pesanan yang telah selesai, hingga pesanan yang mengalami pembatalan?
- Pertanyaan 3: Produk apa saja yang memiliki penjualan terbanyak dan rating terbaik?
- Pertanyaan 4: Kota mana saja yang memiliki penjualan terbanyak?
""")

# Load Data
main_data = pd.read_csv(r'main_data.csv')

##### Mengeksplorasi Parameter Statistik
cat = main_data.select_dtypes(include=['object']).columns.tolist()
num = main_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

main_data[cat].describe()
main_data[num].describe()
main_data[num].corr()

##### Mengelompokan Data
# product dengan penjualan terbanyak dan memiliki rating terbaik
best_product = main_data.groupby(by='product_category_name_english').agg({'order_id': 'count',
                                                                          'review_score': 'mean'}).reset_index()

best_product = best_product.sort_values(by=['order_id', 'review_score'], ascending=[False, False]).head(10).reset_index()

# Kota dengan penjualan terbanyak
best_seller = main_data.groupby(by='seller_city').agg({'order_id': 'count'}).reset_index()
best_seller = best_seller.sort_values(by='order_id', ascending=False).head(10).reset_index()


## Visualization & Explanatory Analysis
##### Pertanyaan 1:
# Bagaimana kita dapat mengukur tingkat kepuasan pelanggan dalam bentuk persentase?

st.subheader(""" Distribusi Umpan Balik Pelanggan """)

review_score = main_data['review_score'].value_counts()

labels = ['Sangat Puas', 'Puas', 'Kurang Puas', 'Kecewa', 'Sangat Kecewa']
colors = ['#1640D6', '#379237', '#E9B824', '#FF6C22', '#D80032']
sizes = review_score.values

fig, ax = plt.subplots(figsize=(20, 15))
ax.pie(x=sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 12})

st.pyplot(fig)

##### Pertanyaan 2:
# Bagaimana distribusi status pesanan pelanggan, seperti pesanan yang sedang diproses, pesanan dalam pengiriman, pesanan yang telah selesai, hingga pesanan yang mengalami pembatalan?

st.subheader(""" Distribusi Status Pesanan Pelanggan """)

main_data['order_status'].value_counts()

fig, ax = plt.subplots(figsize=(20, 16))
bars = sns.countplot(data=main_data, x='order_status', orient='h', color='#3970F1', ax=ax)
for p in bars.patches:
    bars.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=20, color='black', xytext=(0, 5), textcoords='offset points')

ax.set_xlabel('Status Pesanan', fontsize=20)
ax.set_ylabel('Jumlah Pelanggan', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

st.pyplot(fig)

##### Pertanyaan 3:
# Produk apa saja yang memiliki penjualan terbanyak dan rating terbaik?

st.subheader(""" 10 Produk dengan Penjualan Terbanyak dan Rating Terbaik """)

fig, ax = plt.subplots(figsize=(20, 16))
bars = ax.barh(best_product['product_category_name_english'],
                best_product['review_score'],
                color='#3970F1'
               )
for bar, order_count in zip(bars, best_product['order_id']):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(order_count), ha='center', va='center', fontsize=20)

ax.set_xlabel('Jumlah Rating', fontsize=20)
ax.set_ylabel('Kategori Produk', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

ax.invert_yaxis()
st.pyplot(fig)

##### Pertanyaan 4:
# Kota mana saja yang memiliki penjualan terbanyak?

st.subheader(""" 10 Kota dengan Penjualan Terbanyak """)

fig, ax = plt.subplots(figsize=(20, 16))
ax.barh(best_seller['seller_city'],
         best_seller['order_id'],
         color='#3970F1'
         )

ax.set_xlabel('Jumlah Order', fontsize=20)
ax.set_ylabel('Kota', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

ax.invert_yaxis()
st.pyplot(fig)


st.markdown("""
## Rekomendasi Bisnis

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

""")

st.caption('Copyright © Project Akhir - Hafiizh Taufiqul Hakim')

