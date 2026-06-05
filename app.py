import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
from streamlit_image_comparison import image_comparison

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVD Image Compressor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; font-family: 'Space Mono', monospace; color: #4f8ef7; }
    .metric-card .label { font-size: 0.78rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .section-header {
        font-family: 'Space Mono', monospace; font-size: 1rem; color: #4f8ef7;
        text-transform: uppercase; letter-spacing: 2px;
        border-bottom: 1px solid #2e3450; padding-bottom: 8px; margin: 24px 0 16px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f8ef7, #7c5cfc);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-family: 'Space Mono', monospace; font-size: 0.85rem; width: 100%;
    }
    .comparison-label { text-align: center; font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .tip-box { background: #1a2235; border-left: 3px solid #4f8ef7; border-radius: 6px; padding: 12px 16px; font-size: 0.85rem; color: #a0aec0; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Core Cached SVD Engines (The Fix for Slowness) ───────────────────────────

@st.cache_data(show_spinner="Performing Full SVD Matrix Decomposition...")
def get_image_svd(img_array, mode="RGB"):
    """
    Computes SVD exactly ONCE when the image is uploaded or mode changes.
    Saves the full decomposition in memory for fast slicing.
    """
    if mode == "Grayscale":
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        U, S, Vt = np.linalg.svd(gray.astype(float), full_matrices=False)
        return [U], [S], [Vt]
    
    # RGB Mode - separate cache tracking per channel
    U_list, S_list, Vt_list = [], [], []
    for i in range(3):
        U, S, Vt = np.linalg.svd(img_array[:, :, i].astype(float), full_matrices=False)
        U_list.append(U)
        S_list.append(S)
        Vt_list.append(Vt)
    return U_list, S_list, Vt_list


def reconstruct_channel(U, Sigma, Vt, r):
    r = max(1, min(r, len(Sigma)))
    return np.dot(U[:, :r], np.dot(np.diag(Sigma[:r]), Vt[:r, :]))


def fast_reconstruct_image(U_list, S_list, Vt_list, rank, mode="RGB"):
    """Instantly slices cached matrices down to rank 'r' without recomputing SVD"""
    if mode == "Grayscale":
        result = np.clip(reconstruct_channel(U_list[0], S_list[0], Vt_list[0], rank), 0, 255)
        return np.stack([result]*3, axis=2).astype(np.uint8)
    
    channels = []
    for i in range(3):
        res = reconstruct_channel(U_list[i], S_list[i], Vt_list[i], rank)
        channels.append(np.clip(res, 0, 255))
    return np.stack(channels, axis=2).astype(np.uint8)


def energy_at_rank(sigma, r):
    r = max(1, min(r, len(sigma)))
    return np.sum(sigma[:r]**2) / np.sum(sigma**2) * 100

def auto_rank(sigma, target=90):
    total = np.sum(sigma**2)
    hits = np.where(np.cumsum(sigma**2) / total >= target / 100)[0]
    return int(hits[0]) + 1 if len(hits) > 0 else len(sigma)

def file_size_kb(img_array, fmt="JPEG"):
    buf = io.BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buf, format=fmt)
    return buf.tell() / 1024

def storage_pct(m, n, r):
    return r * (m + n + 1) / (m * n) * 100


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    mode = st.radio("Image Mode", ["RGB", "Grayscale"], horizontal=True)
    st.markdown("---")
    st.markdown("### Rank Selection")
    auto_optimize = st.button("✨ Auto Optimize (90% energy)")
    target_energy = st.slider("Target energy %", 50, 99, 90)
    
    # Sensible default step bound until file dictates otherwise
    rank = st.slider("Manual Rank", min_value=1, max_value=200, value=50)
    st.markdown("---")
    st.markdown("### Batch Compression")
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg","jpeg","png","bmp","webp","tiff"],
        accept_multiple_files=True, key="batch"
    )
    st.markdown('---')
    st.markdown('<div class="tip-box">💡 Lower rank = smaller file, less detail.<br>Higher rank = larger file, more detail.</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("# 🖼️ SVD Image Compressor")
st.markdown("Compress images using **Singular Value Decomposition** — rank-1 matrix decomposition in action.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp","tiff"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    m, n = img_array.shape[:2]
    max_rank = min(m, n)

    # 1. Compute Full SVD ONCE via cache
    U_list, S_list, Vt_list = get_image_svd(img_array, mode)

    if auto_optimize:
        rank = auto_rank(S_list[0], target_energy)
        st.sidebar.success(f"Auto rank set to **{rank}** for {target_energy}% energy")

    rank = min(rank, max_rank)
    
    # 2. Reconstruct dynamically and instantly
    compressed_array = fast_reconstruct_image(U_list, S_list, Vt_list, rank, mode)

    # ── Metrics ───────────────────────────────────────────────────────────────
    orig_kb   = file_size_kb(img_array)
    comp_kb   = file_size_kb(compressed_array)
    stor_pct  = storage_pct(m, n, rank)
    eng_pct   = energy_at_rank(S_list[0], rank)
    saved_pct = max(0, (1 - comp_kb / orig_kb) * 100)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, label in zip(
        [c1,c2,c3,c4,c5],
        [f"{orig_kb:.0f} KB", f"{comp_kb:.0f} KB", f"{saved_pct:.1f}%", f"{stor_pct:.1f}%", f"{eng_pct:.1f}%"],
        ["Original Size","Compressed Size","Space Saved","Storage Used","Energy Preserved"]
    ):
        col.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Image comparison (Upgraded to interactive Split Slider) ───────────────
    st.markdown('<div class="section-header">Interactive Canvas Comparison</div>', unsafe_allow_html=True)
    
    # Convert arrays back to Pillow handles for the canvas widget
    orig_pil = Image.fromarray(img_array if mode == "RGB" else np.stack([np.mean(img_array, axis=2)]*3, axis=2).astype(np.uint8))
    comp_pil = Image.fromarray(compressed_array)
    
    image_comparison(
        img1=orig_pil,
        img2=comp_pil,
        label1="Original Image",
        label2=f"SVD Truncated Rank {rank}",
        show_labels=True,
        make_responsive=True
    )

    buf = io.BytesIO()
    Image.fromarray(compressed_array).save(buf, format="JPEG")
    st.download_button("⬇️ Download Compressed Image", buf.getvalue(), file_name=f"compressed_rank{rank}.jpg", mime="image/jpeg")

    st.markdown("---")

    # ── Residual Error Map (Advanced Analysis Addition) ──────────────────────
    st.markdown('<div class="section-header">Residual Error Map (Lost Structural Information)</div>', unsafe_allow_html=True)
    col_err_txt, col_err_img = st.columns([1, 2])
    with col_err_txt:
        st.write("")
        st.write("")
        st.markdown("""
        **What are you looking at?**
        
        This matrix represents the absolute differential mathematical error: 
        $$\\text{Error} = |A_{\\text{original}} - A_{\\text{compressed}}|$$
        
        Bright white lines highlight high-frequency geometric components (such as sharp boundaries or fine textures) that were truncated during the low-rank approximation framework.
        """)
    with col_err_img:
        # Calculate structural loss map
        error_map = np.abs(img_array.astype(float) - compressed_array.astype(float))
        error_map = np.mean(error_map, axis=2) # Collapse channels to emphasize structural visibility
        if np.max(error_map) > 0:
            error_map = (error_map / np.max(error_map) * 255).astype(np.uint8)
        else:
            error_map = error_map.astype(np.uint8)
        st.image(error_map, caption="Normalized Pixel-Wise Reconstruction Error", use_column_width=True)

    st.markdown("---")

    # ── Rank comparison table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Rank Comparison Table</div>', unsafe_allow_html=True)
    rows = []
    for r in [5, 10, 25, 50, 75, 100]:
        if r > max_rank: continue
        # Quickly re-simulate steps using fast slice matrices
        arr = fast_reconstruct_image(U_list, S_list, Vt_list, r, mode)
        rows.append({
            "Rank": r,
            "Storage Used": f"{storage_pct(m,n,r):.1f}%",
            "Energy Preserved": f"{energy_at_rank(S_list[0],r):.1f}%",
            "Compressed Size": f"{file_size_kb(arr):.0f} KB",
            "Space Saved": f"{max(0,(1-file_size_kb(arr)/orig_kb)*100):.1f}%"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Analysis Charts</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Cumulative Energy Preservation**")
        fig, ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        colors = ['#e05c7a','#4caf8e','#4f8ef7']
        labels = ['R','G','B'] if mode=="RGB" else ['Gray']
        for sig, color, lbl in zip(S_list, colors, labels):
            xs = range(1, min(200,len(sig))+1)
            ax.plot(xs, [energy_at_rank(sig,r) for r in xs], color=color, linewidth=2, label=lbl)
        ax.axvline(x=rank, color='white', linestyle='--', linewidth=1, alpha=0.6, label=f'Rank {rank}')
        ax.axhline(y=90, color='#f5c842', linestyle=':', linewidth=1, alpha=0.6, label='90% energy')
        ax.set_xlabel("Rank", color='#8892a4', fontsize=9); ax.set_ylabel("Energy %", color='#8892a4', fontsize=9)
        ax.tick_params(colors='#8892a4', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3450')
        ax.legend(fontsize=8, facecolor='#1e2130', labelcolor='white', framealpha=0.5)
        st.pyplot(fig); plt.close(fig)

    with ch2:
        st.markdown("**Singular Values Distribution**")
        fig, ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        top_n = min(80, len(S_list[0]))
        for sig, color, lbl in zip(S_list, ['#e05c7a','#4caf8e','#4f8ef7'], ['R','G','B'] if mode=="RGB" else ['Gray']):
            ax.bar(range(1,top_n+1), sig[:top_n], color=color, alpha=0.6, label=lbl, width=1.0)
        ax.axvline(x=rank, color='white', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Rank {rank}')
        ax.set_xlabel("Singular Value Index", color='#8892a4', fontsize=9); ax.set_ylabel("Magnitude", color='#8892a4', fontsize=9)
        ax.tick_params(colors='#8892a4', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3450')
        ax.legend(fontsize=8, facecolor='#1e2130', labelcolor='white', framealpha=0.5)
        st.pyplot(fig); plt.close(fig)

    # Channel-wise curves (RGB only)
    if mode == "RGB":
        st.markdown("---")
        st.markdown('<div class="section-header">Channel-wise Energy at Current Rank</div>', unsafe_allow_html=True)
        for col, sig, name, color in zip(
            st.columns(3), S_list,
            ["Red Channel","Green Channel","Blue Channel"],
            ["#e05c7a","#4caf8e","#4f8ef7"]
        ):
            e = energy_at_rank(sig, rank)
            fig, ax = plt.subplots(figsize=(3.5,2.5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            xs = range(1, min(150,len(sig))+1)
            ax.plot(xs, [energy_at_rank(sig,r) for r in xs], color=color, linewidth=2)
            ax.axvline(x=rank, color='white', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_title(f"{name}\n{e:.1f}% energy", color='white', fontsize=9)
            ax.tick_params(colors='#8892a4', labelsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor('#2e3450')
            col.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # ── SVD vs JPEG ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">SVD vs JPEG Comparison</div>', unsafe_allow_html=True)
    cmp_rows = []
    for q in [10, 30, 60, 90]:
        buf_j = io.BytesIO()
        Image.fromarray(img_array).save(buf_j, format="JPEG", quality=q)
        j_kb = buf_j.tell() / 1024
        cmp_rows.append({"Method": f"JPEG quality={q}", "File Size (KB)": f"{j_kb:.0f} KB", "Space Saved": f"{max(0,(1-j_kb/orig_kb)*100):.1f}%"})
    cmp_rows.append({"Method": f"SVD rank={rank}", "File Size (KB)": f"{comp_kb:.0f} KB", "Space Saved": f"{saved_pct:.1f}%"})
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)


# ── Batch ─────────────────────────────────────────────────────────────────────
if batch_files:
    st.markdown("---")
    st.markdown('<div class="section-header">Batch Compression</div>', unsafe_allow_html=True)
    batch_rank = st.slider("Rank for batch images", 1, 200, 50, key="batch_rank")
    if st.button("🗜️ Compress All & Download ZIP"):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            progress = st.progress(0)
            for idx, f in enumerate(batch_files):
                barr = np.array(Image.open(f).convert("RGB"))
                m_b, n_b = barr.shape[:2]
                
                # We calculate SVD locally for batch files directly
                U_b, S_b, Vt_b = [], [], []
                for i in range(3):
                    u, s, vt = np.linalg.svd(barr[:,:,i].astype(float), full_matrices=False)
                    U_b.append(u)
                    S_b.append(s)
                    Vt_b.append(vt)
                
                br = min(batch_rank, min(m_b, n_b))
                comp_b = fast_reconstruct_image(U_b, S_b, Vt_b, br, "RGB")
                
                img_buf = io.BytesIO()
                Image.fromarray(comp_b).save(img_buf, format="JPEG")
                zf.writestr(f"compressed_{f.name}", img_buf.getvalue())
                progress.progress((idx+1)/len(batch_files))
        st.download_button("⬇ " "Download ZIP", zip_buf.getvalue(), file_name="svd_compressed.zip", mime="application/zip")
        st.success(f"✅ {len(batch_files)} images compressed!")

elif not uploaded_file:
    st.markdown("---")
    st.info("👆 Upload an image above to get started, or upload multiple images in the sidebar for batch compression.")