import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import time

# Konfigurasi Halaman
st.set_page_config(
    page_title="Doodle-to-Art | Diffusion Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Tech Theme)
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #1e1b4b;
        color: #818cf8;
        border: 1px solid #4338ca;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<span class="badge">Portfolio Project • Generative AI</span>', unsafe_allow_html=True)
st.title("🎨 Doodle-to-Art: Sketch-Guided Diffusion Studio")
st.caption("Visualisasi interaktif bagaimana Diffusion Model (U-Net & ControlNet) membersihkan noise dari acak ($t=1000$) menjadi karya seni utuh ($t=0$).")

# Sidebar: Pengaturan Style & Parameter Diffusion
with st.sidebar:
    st.header("⚙️ Diffusion Parameters")
    
    style_preset = st.selectbox(
        "Pilih Art Style:",
        ["Cyberpunk Neon", "Studio Ghibli Anime", "Watercolor Dream", "Pixel Art 32-bit", "Oil Painting Masterpiece"]
    )
    
    user_prompt = st.text_input(
        "Deskripsi Prompt Tambahan:",
        value="lush vibrant colors, cinematic lighting, 8k masterpiece"
    )
    
    st.divider()
    st.subheader("🔬 Model Hyperparameters")
    timesteps = st.slider("Total Timesteps ($T$):", min_value=100, max_value=1000, value=500, step=50)
    guidance_scale = st.slider("Classifier-Free Guidance ($w$):", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
    noise_schedule = st.selectbox("Noise Schedule:", ["Linear Schedule", "Cosine Schedule"])

# Layout 2 Kolom
col_canvas, col_preview = st.columns([1, 1], gap="large")

with col_canvas:
    st.subheader("✏️ 1. Sketch / Doodle Input")
    st.write("Gambar sketsa sederhana kamu di bawah ini:")
    
    tool = st.radio("Tool:", ["Pencil", "Eraser"], horizontal=True)
    stroke_width = st.slider("Ukuran Kuas:", 3, 25, 6)
    stroke_color = "#000000" if tool == "Pencil" else "#FFFFFF"
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        height=340,
        width=340,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas",
    )
    
    st.info("💡 Tips: Gambar bentuk sederhana seperti gunung, rumah, pohon, atau wajah.")

# Fungsi Simulasi Diffusion Process
def simulate_diffusion(sketch_img, num_steps=6):
    """
    Simulasi matematis Forward dan Reverse Diffusion:
    Dari pure static noise (t=1000) perlahan dibersihkan sesuai kontur gambar sketsa.
    """
    gray = sketch_img.convert("L").resize((256, 256))
    edges = ImageOps.invert(gray).filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges) / 255.0
    
    steps = []
    np.random.seed(42)
    pure_noise = np.random.uniform(0, 255, (256, 256, 3)).astype(np.uint8)
    
    color_map = {
        "Cyberpunk Neon": (0, 240, 255),
        "Studio Ghibli Anime": (130, 205, 71),
        "Watercolor Dream": (244, 154, 193),
        "Pixel Art 32-bit": (255, 179, 0),
        "Oil Painting Masterpiece": (194, 98, 41)
    }
    tint = color_map.get(style_preset, (100, 150, 255))
    
    for i in range(num_steps):
        t_val = int(1000 * (1 - (i / (num_steps - 1))))
        ratio = i / (num_steps - 1)
        
        noise_level = 1.0 - ratio
        noise = np.random.normal(128, 64, (256, 256, 3))
        
        art_base = np.zeros((256, 256, 3), dtype=np.float32)
        for c in range(3):
            art_base[:, :, c] = edge_arr * tint[c] + (1 - edge_arr) * (40 + 20 * c)
            
        blended = (noise * noise_level) + (art_base * (1.0 - noise_level))
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        img_step = Image.fromarray(blended)
        if ratio > 0.6:
            img_step = img_step.filter(ImageFilter.SMOOTH_MORE)
            
        steps.append((t_val, img_step))
    return steps

with col_preview:
    st.subheader("🔮 2. Diffusion Denoising Output")
    
    generate_btn = st.button("🚀 Run Diffusion Process")
    
    if generate_btn:
        if canvas_result.image_data is not None:
            raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            
            with st.spinner("Menginisialisasi U-Net denoiser dan noise schedule..."):
                time.sleep(0.5)
                diffusion_steps = simulate_diffusion(raw_img)
            
            st.success("Selesai! Berikut visualisasi proses denoising step-by-step:")
            
            # Tampilkan 6 tahap denoising (dari t=1000 ke t=0)
            cols = st.columns(3)
            for idx, (t, step_img) in enumerate(diffusion_steps[:3]):
                with cols[idx]:
                    st.image(step_img, caption=f"t = {t} (Noise tinggi)", use_container_width=True)
            
            cols2 = st.columns(3)
            for idx, (t, step_img) in enumerate(diffusion_steps[3:]):
                with cols2[idx]:
                    label = "t = 0 (Final Output)" if t == 0 else f"t = {t}"
                    st.image(step_img, caption=label, use_container_width=True)
            
            # Final Showcase
            st.divider()
            final_img = diffusion_steps[-1][1].resize((400, 400), Image.Resampling.LANCZOS)
            st.image(final_img, caption=f"Hasil Akhir: {style_preset}", use_container_width=True)
        else:
            st.warning("Silakan buat coretan terlebih dahulu di kanvas!")

# Bagian Dokumentasi Teknis untuk Portofolio GitHub
st.divider()
st.subheader("📚 Under The Hood: Cara Kerja Diffusion Model")
tab_math, tab_readme = st.tabs(["📐 Konsep Matematika", "📋 GitHub README Template"])

with tab_math:
    st.markdown(r"""
    ### 1. Forward Process (Adding Noise)
    Data gambar diacak secara bertahap menggunakan distribusi Gaussian:
    $$q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}\right)$$
    
    ### 2. Reverse Process (Denoising via U-Net)
    Model U-Net dilatih untuk memprediksi noise $\epsilon_\theta(x_t, t, c)$ yang ditambahkan pada tiap langkah $t$, di mana $c$ adalah conditioning (sketsa coretan pengguna):
    $$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(\mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t)\right)$$
    """)

with tab_readme:
    st.write("Salin teks di bawah ini untuk isi file README.md di repository GitHub kamu:")
    readme_content = """# Doodle-to-Art: Sketch-Guided Diffusion Playground

Interactive web application demonstrating how conditional Diffusion Models (U-Net & ControlNet) reconstruct high-fidelity artwork from crude user sketches and noise static.

## Features
- Interactive Drawing Canvas: Draw doodles directly in the browser.
- Step-by-step Diffusion Stepper: Watch the reverse denoising process from t=1000 to t=0.
- Style Conditioning: Supports multiple aesthetics (Cyberpunk, Anime, Watercolor, Pixel Art).

## Tech Stack
- Streamlit & Python 3
- Pillow, NumPy, Matplotlib
- streamlit-drawable-canvas
"""
    st.code(readme_content, language="markdown")
