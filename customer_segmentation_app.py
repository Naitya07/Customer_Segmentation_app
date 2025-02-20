import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Customer Segmentation & Recommendation System")

# File uploaders for both datasets
uploaded_customers = st.file_uploader("Upload Mall Customers Dataset", type=["csv"], key="customers")
uploaded_transactions = st.file_uploader("Upload Online Retail Dataset", type=["csv"], key="transactions")

if uploaded_customers is not None:
    df_customers = pd.read_csv(uploaded_customers)
    st.write("### Mall Customers Data Preview")
    st.write(df_customers.head())
    
    # Selecting relevant features
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    if not all(col in df_customers.columns for col in features):
        st.error("Required columns not found in dataset.")
    else:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_customers[features])
        
        # K-Means Clustering
        k = st.slider("Select number of clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df_customers['Cluster'] = kmeans.fit_predict(df_scaled)
        
        st.write("### Clustered Data")
        st.write(df_customers.head())
        
        # Visualization
        st.write("### Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_customers[features[0]], y=df_customers[features[1]], hue=df_customers['Cluster'], palette='viridis', ax=ax)
        st.pyplot(fig)

if uploaded_transactions is not None:
    df_transactions = pd.read_csv(uploaded_transactions, encoding='ISO-8859-1')
    st.write("### Online Retail Data Preview")
    st.write(df_transactions.head())
    
    if 'Description' in df_transactions.columns and 'InvoiceNo' in df_transactions.columns and 'CustomerID' in df_transactions.columns:
        # Data preprocessing
        df_transactions.dropna(subset=['CustomerID', 'Description'], inplace=True)
        df_transactions['CustomerID'] = df_transactions['CustomerID'].astype(str)
        
        # Basket for Apriori
        basket = df_transactions.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Apply Apriori algorithm
        min_support = st.slider("Select Minimum Support", 0.01, 0.2, 0.05)
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        
        # Association Rules
        min_lift = st.slider("Select Minimum Lift", 1.0, 5.0, 1.5)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        
        st.write("### Frequent Itemsets")
        st.write(frequent_itemsets.head())
        
        st.write("### Association Rules")
        st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        
        # Customer-specific recommendations
        customer_ids = df_transactions['CustomerID'].unique().tolist()
        selected_customer = st.selectbox("Select a Customer ID", customer_ids)
        
        if selected_customer:
            customer_basket = set(df_transactions[df_transactions['CustomerID'] == selected_customer]['Description'].unique())
            
            # Convert frozen sets to lists for proper comparison
            rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
            
            matched_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(customer_basket))]
            
            if not matched_rules.empty:
                st.write("### Recommended Products for Customer")
                recommended_products = set(matched_rules['consequents'].explode())
                st.write(list(recommended_products))
            else:
                st.write("No recommendations found for this customer. Try lowering the minimum lift or support.")
    else:
        st.error("Required columns 'InvoiceNo', 'Description', and 'CustomerID' not found in dataset.")
