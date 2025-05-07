<!-- Badges -->

![Python 3.10.9 (amd64)](https://img.shields.io/badge/Python-3.10.9%20amd64-blue.svg)
![Streamlit 1.45.0](https://img.shields.io/badge/Streamlit-1.45.0-orange.svg)
![TensorFlow 2.12.0](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)
![OpenCV-Headless 4.11.0.86](https://img.shields.io/badge/OpenCV--Headless-4.11.0.86-brightgreen.svg)
![scikit-image 0.25.2](https://img.shields.io/badge/scikit--image-0.25.2-lightgrey.svg)
![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)

---

# BrainTumorAssist

*A streamlined AI-driven platform for real-time MRI-based brain tumor detection and segmentation, built with Streamlit.*

[🔗 Live Demo](https://braintumorassist.streamlit.app/) | [⭐️ Star on GitHub](https://github.com/madashivakarthikgoud/BrainTumorAssist)

---

## 🚀 Features

* **Fast MRI Upload & Analysis**: Drag & drop support for `.jpg`, `.jpeg`, and `.png` files
* **Deep Learning Models**:

  * **CNN Classifier** for tumor probability estimation
  * **U-Net Segmenter** for precise tumor mask overlay
* **Interactive Metrics**: Tumor area (px & mm²), coverage percentage, processing time
* **Session History**: View, clear, and export the last 10 analyses as ZIP reports
* **Modern Dark UI**: Custom CSS for a sleek, responsive interface

---

## 📦 Tech Stack

| Layer                | Technology                    |
| -------------------- | ----------------------------- |
| **Frontend**         | Streamlit 1.45.0              |
| **Backend**          | Python 3.10.9                 |
| **Deep Learning**    | TensorFlow 2.12.0, Keras      |
| **Image Processing** | OpenCV-Headless, scikit-image |
| **Utilities**        | NumPy, Pillow, UUID, base64   |

---

## 📂 Project Structure

```plaintext
BrainTumorAssist/
├─ CNN/
│  ├─ model.py
│  └─ Model/
│     └─ weights.hdf5
├─ ImageSegmentation/
│  ├─ model_bt.py
│  └─ Model/
│     └─ weights.hdf5
├─ imgForTest/          # Sample input/output images
├─ app.py               # Streamlit application entrypoint
└─ requirements.txt     # Project dependencies
```

---

## 📥 Installation & Run Locally

> **Prerequisite:** Python 3.10.9 (amd64)

```bash
# Clone the repository
git clone https://github.com/madashivakarthikgoud/BrainTumorAssist.git
cd BrainTumorAssist

# Create & activate a virtual environment
python3.10 -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🤝 Contributing

Fork the project, add your features or fixes, and submit a pull request. We appreciate ⭐️ stars and welcome all improvements!

---

## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

> **Made with ❤️ by Shiva Karthik**
