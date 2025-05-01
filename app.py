import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import streamlit as st
from datetime import datetime
import io

# Fungsi preprocessing
def preprocess_data(df):
    # Copy dataframe
    df_clean = df.copy()
    
    # 1. Ubah InvoiceDate ke datetime
    if 'InvoiceDate' in df_clean.columns:
        try:
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        except:
            st.error("Format kolom 'InvoiceDate' tidak valid. Pastikan format tanggal benar.")
            return None
    
    # 2. Handle missing value
    missing_cols = df_clean.isnull().sum()
    st.write("Missing Values:", missing_cols[missing_cols > 0])
    df_clean = df_clean.dropna(subset=['Description', 'InvoiceNo', 'Quantity', 'InvoiceDate'])
    
    # 3. Handle duplikat
    duplicates = df_clean.duplicated().sum()
    st.write(f"Duplicated Rows: {duplicates}")
    df_clean = df_clean.drop_duplicates()
    
    # 4. Filter transaksi valid
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean.get('UnitPrice', 1) > 0)]
    
    # 5. Handle transaksi cancelled (InvoiceNo mulai dengan 'C')
    if 'InvoiceNo' in df_clean.columns:
        df_clean = df_clean[~df_clean['InvoiceNo'].str.startswith('C', na=False)]
    
    # 6. Tambah kolom TotalPrice
    if 'UnitPrice' in df_clean.columns:
        df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # 7. Tambah kolom Total_Item per transaksi
    df_clean['Total_Item'] = df_clean.groupby('InvoiceNo')['Description'].transform('nunique')
    
    st.write("Data Shape setelah Preprocessing:", df_clean.shape)
    return df_clean

# Fungsi Market Basket Analysis
def market_basket_analysis(df, min_support=0.02, min_confidence=0.6):
    # Fokus ke negara dengan transaksi terbanyak (jika ada kolom Country)
    if 'Country' in df.columns:
        top_country = df['Country'].value_counts().index[0]
        df_mba = df[df['Country'] == top_country]
    else:
        df_mba = df
    
    # Buat basket
    basket = df_mba.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Apriori
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    
    # Association Rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] > 1.2]
    
    return frequent_itemsets, rules

# Fungsi Time Series Analysis
def time_series_analysis(df):
    # Agregasi harian
    daily_trx = df.groupby(df['InvoiceDate'].dt.date)['InvoiceNo'].nunique().reset_index()
    daily_trx.columns = ['Date', 'Transaction_Count']
    daily_trx['Date'] = pd.to_datetime(daily_trx['Date'])
    
    # Cek stationarity
    adf_result = adfuller(daily_trx['Transaction_Count'])
    if adf_result[1] > 0.05:
        daily_trx['Transaction_Count_Diff'] = daily_trx['Transaction_Count'].diff().dropna()
    
    # Split train-test
    train_data = daily_trx['Transaction_Count'][:-30]
    test_data = daily_trx['Transaction_Count'][-30:]
    
    # ARIMA
    arima_model = ARIMA(train_data, order=(1,1,1)).fit()
    arima_pred = arima_model.forecast(steps=30)
    
    # Evaluasi
    mae = mean_absolute_error(test_data, arima_pred)
    
    return daily_trx, arima_pred, test_data, mae

# Streamlit App
st.title("Retail Analytics Automation")
st.write("Upload file Excel/CSV untuk analisis rekomendasi produk dan prediksi tren transaksi.")

# Upload file
uploaded_file = st.file_uploader("Pilih file Excel/CSV", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Baca file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("Data Awal:", df.head())
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        st.stop()
    
    # Preprocessing
    st.header("Preprocessing")
    df_clean = preprocess_data(df)
    if df_clean is None:
        st.stop()
    
    # Simpan data bersih
    output = io.BytesIO()
    df_clean.to_csv(output, index=False)
    st.download_button(
        label="Download Data Bersih",
        data=output.getvalue(),
        file_name="retail_cleaned.csv",
        mime="text/csv"
    )
    
    # Market Basket Analysis
    st.header("Market Basket Analysis")
    frequent_itemsets, rules = market_basket_analysis(df_clean)
    
    st.write("Top Frequent Itemsets:")
    st.dataframe(frequent_itemsets.head(10))
    
    st.write("Top Association Rules:")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    
    # Visualisasi: Network Graph
    st.subheader("Network of Top Product Associations")
    G = nx.DiGraph()
    for idx, rule in rules.head(5).iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                G.add_edge(ant, cons, weight=rule['lift'])
    fig, ax = plt.subplots(figsize=(10,8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): f"Lift: {d['weight']:.2f}" for u,v,d in G.edges(data=True)})
    st.pyplot(fig)
    
    # Visualisasi: Barplot Top Rules
    st.subheader("Top 5 Rules by Lift")
    top_rules = rules.nlargest(5, 'lift')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='lift', y=top_rules['consequents'].apply(lambda x: ', '.join(x)), hue='confidence', data=top_rules, ax=ax)
    ax.set_title('Top 5 Association Rules by Lift')
    st.pyplot(fig)
    
    # Rekomendasi produk spesifik
    st.subheader("Rekomendasi Produk")
    focus_item = st.selectbox("Pilih produk untuk rekomendasi", df_clean['Description'].unique())
    rekomendasi = rules[rules['antecedents'].apply(lambda x: focus_item in x)]
    if not rekomendasi.empty:
        st.write(f"Rekomendasi jika beli {focus_item}:")
        st.dataframe(rekomendasi[['consequents', 'confidence', 'lift']])
    else:
        st.write(f"Tidak ada rekomendasi untuk {focus_item}.")
    
    # Time Series Analysis
    st.header("Time Series Analysis")
    daily_trx, arima_pred, test_data, mae = time_series_analysis(df_clean)
    
    st.write(f"MAE Prediksi: {mae:.2f}")
    
    # Visualisasi: Prediksi vs Aktual
    st.subheader("Prediksi vs Aktual Jumlah Transaksi Harian")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(daily_trx['Date'][-30:], test_data, label='Actual', marker='o')
    ax.plot(daily_trx['Date'][-30:], arima_pred, label='Predicted', marker='x')
    ax.set_title('Prediksi vs Aktual Jumlah Transaksi Harian')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Jumlah Transaksi')
    ax.legend()
    ax.grid()
    st.pyplot(fig)
    
    # Visualisasi: Tren Bulanan
    st.subheader("Tren Jumlah Transaksi per Bulan")
    monthly_trx = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['InvoiceNo'].nunique().reset_index()
    monthly_trx['InvoiceDate'] = monthly_trx['InvoiceDate'].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(monthly_trx['InvoiceDate'], monthly_trx['InvoiceNo'], marker='o', color='blue')
    ax.set_title('Tren Jumlah Transaksi per Bulan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Jumlah Transaksi')
    ax.grid()
    st.pyplot(fig)
    
    # Rekomendasi Bisnis
    st.header("Rekomendasi Bisnis")
    st.write("- **Bundling Produk**: Gunakan association rules untuk membuat promo bundling (misal, jika beli produk A, tawarkan produk B).")
    st.write("- **Perencanaan Stok**: Siapkan stok ekstra di bulan dengan prediksi transaksi tinggi (misal, Q4 berdasarkan tren).")
    st.write("- **Promo Musiman**: Luncurkan kampanye di bulan dengan transaksi rendah untuk tingkatkan penjualan.")