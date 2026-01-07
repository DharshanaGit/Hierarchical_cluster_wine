import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files
scaler = joblib.load('scaler_model.pkl')
centroids = joblib.load('centroids.pkl')

st.set_page_config(page_title="Customer Cluster Finder")
st.title("🏦 Find Bank Customer Cluster")
st.write("This model helps to find which cluster the bank customer belong to using Hierarchical Clustering")
# Manual Input fields
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Customer Age", value=45)
    dep = st.number_input("Dependent Count", value=2)
    mob = st.number_input("Months on Book", value=36)
    rel = st.number_input("Total Relationship Count", value=3)
    inact = st.number_input("Months Inactive (12mo)", value=1)
    cont = st.number_input("Contacts Count (12mo)", value=3)
    limit = st.number_input("Credit Limit", value=5000.0)

with col2:
    rev_bal = st.number_input("Total Revolving Bal", value=1000)
    open_buy = st.number_input("Average Open To Buy", value=4000.0)
    amt_chng = st.number_input("Total Amount Chng (Q4/Q1)", value=0.7)
    trans_amt = st.number_input("Total Transaction Amount", value=2000)
    trans_ct = st.number_input("Total Transaction Count", value=50)
    ct_chng = st.number_input("Total Ct Change (Q4/Q1)", value=0.6)
    util = st.number_input("Avg Utilization Ratio", value=0.2)

if st.button("Find My Cluster"):
    # 1. Prepare Data
    data = [age, dep, mob, rel, inact, cont, limit, rev_bal, open_buy, amt_chng, trans_amt, trans_ct, ct_chng, util]
    input_array = np.array(data).reshape(1, -1)
    
    # 2. Scale the Input
    scaled_input = scaler.transform(input_array)
    
    # 3. Find the closest cluster (Euclidean Distance)
    # We find the distance between the user input and all 4 centroids
    distances = np.linalg.norm(centroids - scaled_input, axis=1)
    closest_cluster = np.argmin(distances) + 1 # +1 because clusters start at 1
    
    # 4. Show Result
    st.markdown(f"### 🎯 Result: This customer belongs to **Cluster {closest_cluster}**")
    
    # Simple interpretation (Optional)
    st.write("---")
    st.write(f"Distance to all clusters: {distances}")