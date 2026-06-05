import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the math functions from our engine file
import svd_engine as svd

# App Setup
st.set_page_config(layout="wide", page_title="SVD Image Compressor")
st.title("🖼️ End-to-End SVD Image Compressor")

# Step 1: Dynamic File Upload replaces hardcoded 'your_image.jpg'
uploaded_file = st.file_uploader("C:\\Users\\komal\\Downloads\download.jpg", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Safely handle format conversion to ensure 3 color channels
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img) / 255.0
    
    height, width, _ = img_array.shape
    max_possible_rank = min(height, width)
    
    # Separate channels
    R, G, B = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Step 2: Dynamic User Input replaces fixed ranks list
    st.sidebar.header("Parameters")
    r = st.sidebar.slider("Select Target Rank Component", 1, int(max_possible_rank), int(max_possible_rank * 0.1))
    
    # Step 3: Run execution pipeline dynamically based on slider
    with st.spinner("Recomputing SVD Matrices..."):
        R_compressed = svd.compress_channel(R, r)
        G_compressed = svd.compress_channel(G, r)
        B_compressed = svd.compress_channel(B, r)
        
        compressed_img = np.clip(np.stack([R_compressed, G_compressed, B_compressed], axis=2), 0.0, 1.0)
        
        # Calculate stats
        original_elements = R.shape[0] * R.shape[1]
        compressed_elements = r * (R.shape[0] + R.shape[1] + 1)
        storage_percentage = (compressed_elements / original_elements) * 100
        
        energy_curve = svd.get_energy_data(R)
        energy_retained = energy_curve[r - 1] * 100

    # Step 4: Display Output on Screen
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_array, use_container_width=True, caption=f"Rank: {max_possible_rank}")
    with col2:
        st.subheader(f"Compressed Image")
        st.image(compressed_img, use_container_width=True, caption=f"Rank: {r}")
        
    # Metrics display
    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Storage Size (per channel)", f"{storage_percentage:.1f}%")
    m2.metric("Mathematical Variance Retained", f"{energy_retained:.1f}%")