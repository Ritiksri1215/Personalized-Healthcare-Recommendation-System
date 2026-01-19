import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load YOUR existing pickle files (already in Colab!)
@st.cache_data
def load_model():
    new_df = pd.DataFrame(pickle.load(open('medicine_dict.pkl', 'rb')))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return new_df, similarity

new_df, similarity = load_model()

st.markdown("""
<style>
.main-header {font-size: 3rem; color: #2E86AB; text-align: center;}
.med-card {background-color: #F8F9FA; padding: 1.5rem; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💊 Medicine Recommendation System</h1>', unsafe_allow_html=True)

# Medicine Search Tab
selected_medicine = st.selectbox(
    "Choose medicine:",
    new_df['Drug_Name'].tolist()
)

if st.button("Recommend Similar Medicines", type="primary"):
    medicine_index = new_df[new_df['Drug_Name'] == selected_medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.success(f"**Found:** {selected_medicine}")
    for i, medicine in enumerate(medicines_list):
        rec_name = new_df.iloc[medicine[0]]['Drug_Name']
        st.write(f"{i+1}. **{rec_name}**")
    st.markdown('</div>', unsafe_allow_html=True)
