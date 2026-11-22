<div align="center">

# 🎨 Doodle-to-Art: Sketch-Guided Diffusion Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)]()

**An interactive generative AI playground visualizing conditional diffusion models, demonstrating step-by-step reverse denoising from Gaussian noise to stylized artwork guided by user sketches.**

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Getting Started](#-getting-started) • [Tech Stack](#-tech-stack) • [Roadmap](#-future-roadmap)

---

### 📸 Application Preview
<!-- Ganti URL gambar di bawah dengan screenshot aplikasi kamu -->
![Doodle Diffusion Demo](https://placehold.co/1200x600/1e1b4b/818cf8?text=Place+Your+App+Screenshot+Here+%28docs%2Fdemo.png%29)

</div>

---

## 📌 Executive Summary

While standard machine learning portfolios often rely on boilerplate tabular datasets (Titanic, Housing Prices), this project showcases a deep, practical understanding of modern **Generative AI & Computer Vision foundations**. 

**Doodle-to-Art Diffusion Studio** simulates the exact Markov chain mechanics powering state-of-the-art text-to-image systems like **Stable Diffusion** and **ControlNet**. Users can sketch crude line doodles on an interactive HTML5 canvas, tweak diffusion hyperparameters, and visually scrub through intermediate noise timesteps ($t = 1000 \to 0$) to witness mathematical denoising in real time.

---

## ✨ Key Features

- 🖌️ **Interactive In-Browser Canvas**: Vector/raster sketch canvas with adjustable brush dimensions, eraser tools, and real-time tensor conversion.
- 📉 **Multi-Stage Diffusion Stepper**: Visualizes the discrete reverse denoising process across 6 equidistant timesteps ($t=1000$ to $t=0$).
- 🎛️ **Classifier-Free Guidance (CFG) Tuning**: Interactive slider controls weighting between unconditional Gaussian noise and sketch-conditioned embeddings.
- 🎭 **Multi-Aesthetic Conditioning**: Instant style presets including *Cyberpunk Neon*, *Studio Ghibli Anime*, *Watercolor Dream*, *32-bit Pixel Art*, and *Classic Oil Painting*.
- 📐 **Interactive Math Explainer**: In-app mathematical formulation tabs breaking down forward perturbation and reverse U-Net noise estimation.

---

## 🧠 System Architecture & Mathematical Foundations

```
+------------------+       +-------------------------+       +---------------------+
|   User Doodle    |  -->  | Edge Detection & Tensor |  -->  | Condition Vector (c)|
|  (HTML5 Canvas)  |       | Preprocessing (Pillow)  |       +----------+----------+
+------------------+       +-------------------------+                  |
                                                                        v
+------------------+       +-------------------------+       +---------------------+
| Gaussian Noise   |  -->  | Reverse Diffusion Loop  | <==== | Latent Conditioning |
|   x_T ~ N(0, I)  |       |  t = 1000 -> 0 (U-Net)  |       +---------------------+
+------------------+       +------------+------------+
                                        |
                                        v
                           +-------------------------+
                           | Final High-Res Artwork  |
                           |       Output x_0        |
                           +-------------------------+
```

### 1. Forward Noising Process
Gradually adds Gaussian noise across $T$ timesteps according to a predetermined variance schedule $\beta_1, \dots, \beta_T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}\right)$$

### 2. Conditioned Reverse Denoising
A U-Net architecture parameterizes the reverse trajectory to recover the clean signal $x_0$ guided by spatial conditioning $c$:

$$p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}\left(\mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t)\right)$$

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) | Reactive web dashboard with custom dark-mode CSS |
| **Canvas Engine** | [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) | React Fabric.js canvas wrapper for drawing inputs |
| **Image Processing** | [Pillow (PIL)](https://pillow.readthedocs.io/), [NumPy](https://numpy.org/) | Matrix transformations, spatial filtering, and noise schedules |
| **Plotting & Metrics** | [Matplotlib](https://matplotlib.org/) | Variance schedule and loss curve visualizers |
| **Version Control** | [Git](https://git-scm.com/) & [GitHub](https://github.com/) | CI-ready repository structure |

---

## 🚀 Getting Started

Follow these instructions to run the application locally on your machine.

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/doodle-diffusion.git
   cd doodle-diffusion
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```

The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🗺️ Future Roadmap

- [ ] Connect Hugging Face `diffusers` pipeline with a quantized Stable Diffusion 1.5 + ControlNet checkpoint.
- [ ] Implement interactive Latent Space interpolation slider.
- [ ] Add direct download button for intermediate timestep GIF animations.
- [ ] Docker containerization for one-click cloud deployment on Hugging Face Spaces or AWS EC2.

---

## 👤 Author

- **GitHub**: [@<your-username>](https://github.com/<your-username>)
- **LinkedIn**: [Your Name](https://linkedin.com/in/<your-profile>)
- **Email**: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.