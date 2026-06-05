# 🖼️ SVD Image Compressor
  🌐 Live Demo: https://svd-image-compreappr-n5gmdosgmvagzfgqw3x7u8.streamlit.app/
An interactive web application that compresses images using
**Singular Value Decomposition (SVD)** — bridging pure linear
algebra with real-world computer vision.

> Built with Python · NumPy · Streamlit · Matplotlib · Pillow

---

## 🚀 Features

- **Channel-wise SVD Compression** — Compresses RGB images
  independently per channel for accurate color reconstruction
- **Interactive Rank Slider** — Adjust compression level in
  real time from rank 1 up to the image's maximum rank
- **Live Metrics** — Shows storage retention % and
  mathematical energy (variance) preserved at each rank
- **Energy Curve** — Cumulative singular value plot shows
  how much structure each rank captures
- **Download Button** — Save the compressed image directly
  from the app

---

## 🧮 Mathematical Foundation

### 1. Image as a Matrix

A grayscale image is represented as a matrix $A$ of size
$m \times n$. A color (RGB) image is handled as three separate matrices —
one per channel (R, G, B).

### 2. Singular Value Decomposition

Using the "economy" SVD variant (`full_matrices=False`), any matrix $A$ of rank $k$ is factored into three constituent matrices:

$$A = U \Sigma V^T$$

Where:

| Matrix | Size | Meaning |
|--------|------|---------|
| $U$ | $m \times k$ | Left singular vectors (orthogonal columns spanning the column space) |
| $\Sigma$ | $k \times k$ | Diagonal matrix containing the singular values |
| $V^T$ | $k \times n$ | Right singular vectors (orthogonal rows spanning the row space) |

The singular values on the diagonal of $\Sigma$ are strictly sorted in descending order:

$$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k > 0$$

Each $\sigma_i$ acts as a weight representing the **importance** of its corresponding rank-1 structural component.

### 3. Rank-1 Decomposition

The matrix multiplication can be expanded and rewritten as a finite linear combination of rank-1 outer products:

$$A = \sum_{i=1}^{k} \sigma_i \, u_i (v_i)^T$$

Where $u_i$ is the $i$-th column vector of $U$, and $v_i$ is the $i$-th column vector of $V$. 
The **first term** ($\sigma_1 u_1 v_1^T$) captures the absolute highest geometric variance and dominant structure of the image. Each subsequent term adds progressively finer spatial details, high-frequency elements, and noise.

### 4. Low-Rank Approximation (Compression)

To compress the image, we truncate the sum at a target rank $r$ where $r \ll k$:

$$A_r \approx \sum_{i=1}^{r} \sigma_i \, u_i (v_i)^T$$

In matrix terms, this is equivalent to slicing and keeping only the top $r$ columns of $U$, the top $r \times r$ submatrix of $\Sigma$, and the top $r$ rows of $V^T$:

$$A_r = U_r \, \Sigma_r \, V_r^T$$

### 5. Storage Efficiency

Storing the uncompressed matrix channel requires $m \times n$ scalar values. The truncated rank-$r$ approximation requires only:

$$\text{Elements stored per channel} = r(m + n + 1)$$

The storage retention efficiency percentage is computed as:

$$\text{Storage \%} = \frac{r(m + n + 1)}{m \times n} \times 100\%$$

For a standard $512 \times 512$ image compressed down to a target rank of $r = 50$, the application only needs to store roughly **19.6%** of the original structural elements.

### 6. Energy Preservation

The total structural "energy" or variance contained within a matrix is mathematically defined by its Frobenius norm, which equals the sum of its squared singular values. The proportion of total geometric energy preserved at an arbitrary rank $r$ is calculated via:

$$\text{Energy Ratio} = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{k} \sigma_i^2} \times 100\%$$

> **Key Insight:** Due to the exponential decay of singular values in natural images, keeping just 10% of the dominant components typically preserves over 90% of the total image energy.

### 7. Eckart–Young–Mirsky Theorem

The core mathematical justification for this project relies on the **Eckart–Young–Mirsky Theorem**, which proves that the truncated SVD matrix $A_r$ provides the *globally optimal* low-rank approximation of $A$ under both the Frobenius norm and the spectral norm:

$$\| A - A_r \|_F = \sqrt{\sum_{i=r+1}^{k} \sigma_i^2}$$

No other matrix of rank $r$ can achieve a closer mathematical approximation to the original image.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core execution language |
| NumPy | High-performance matrix factorization via `np.linalg.svd` |
| Streamlit | Modern reactive web UI and layout framework |
| Matplotlib | Custom styling and tracking of the cumulative energy curve |
| Pillow (PIL) | Safe spatial rendering, channel splitting, and image export |

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone [https://github.com/Komal-phogat/SVD-Image-Compressor.git](https://github.com/Komal-phogat/SVD-Image-Compressor.git)
cd SVD-Image-Compressor
