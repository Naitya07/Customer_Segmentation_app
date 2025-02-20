import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Customer Segmentation & Recommendation System")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Selecting numeric columns for clustering
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns found for clustering.")
    else:
        features = st.multiselect("Select features for clustering", numeric_columns, default=numeric_columns[:2])
        
        if len(features) < 2:
            st.error("Please select at least two features for clustering.")
        else:
            # Data preprocessing
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[features])
            
            # K-Means Clustering
            k = st.slider("Select number of clusters (K)", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_data)
            
            st.write("### Clustered Data")
            st.write(df.head())
            
            # Visualization
            st.write("### Cluster Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Cluster'], palette='viridis', ax=ax)
            st.pyplot(fig)
    
    # Recommendation System Section
    st.write("## Product Recommendation System")
    if 'Description' in df.columns and 'InvoiceNo' in df.columns:
        # Prepare data for association rule mining
        basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
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
    else:
        st.error("Required columns 'InvoiceNo' and 'Description' not found in dataset.")
