
# Vein Hunter: Real-Time Subcutaneous Vein Extraction

[](https://www.python.org/downloads/)
[](https://pytorch.org/get-started/locally/)
[](https://opensource.org/licenses/MIT)

### Overview
Securing a successful first-attempt IV insertion is a widespread clinical challenge. While commercial vein-finding devices solve this using expensive Near-Infrared (NIR) or hyperspectral hardware, **Vein Hunter** is designed to democratize the process. This pipeline extracts complex vascular networks directly from visible-light images captured by ubiquitous optical sensors (such as standard webcams or smartphones).

By combining lightweight classical computer vision preprocessing with an attention-guided neural core (CBAM U-Net), this project successfully bypasses computationally heavy multiscale filters (like Frangi). The result is high-accuracy, ultra-low latency vein detection capable of serving as the Initial Spatial Acquisition System for **autonomous robotic venipuncture**.

-----

## Key Features

  * **Attention-Guided Architecture:** Utilizes Convolutional Block Attention Modules (**CBAM**) to dynamically suppress epidermal noise (hair, wrinkles, glare) while highlighting subtle venous gradients.
  * **Real-Time Performance:** Purely neural inference path optimized for GPU, achieving high frame rates on live video streams.
  * **Mobile Integration:** Supports wireless live-streaming via iPhone/Android using IP Camera protocols.
  * **Dual-Phase Training:** Leverages transfer learning from high-contrast retinal datasets (DRIVE/FIVES) fine-tuned on a custom dataset of 116 human arm images.
  * **Robotic-Ready:** Outputs precise 2D (X, Y) spatial coordinates for initial acquisition systems in medical robotics.

-----

## Tech Stack

  * **Deep Learning:** PyTorch, Torchvision
  * **Computer Vision:** OpenCV, Albumentations
  * **Visualization:** Matplotlib, PIL
  * **Hardware Integration:** IP Camera (Mobile-to-PC Stream)
  * **Documentation:** LaTeX (NeurIPS Formatting)

-----

## Architecture

The model uses a **U-Net** backbone enhanced with **CBAM** (Channel and Spatial Attention) blocks in the decoder path. This allow the network to perform "semantic rejection" of surface artifacts that typically confuse classical filters like the Frangi vesselness filter.

### Architecture diagram
![Architecture diagram](results/flowchart.png)
-----

## Getting Started

### 1 Installation

```bash
git clone https://github.com/pragatirokade/vein-hunter.git
cd vein-hunter
```

### 2. Install Dependencies
```bash
pip install torch torchvision opencv-python numpy pillow matplotlib albumentations
```

### 3 Live Stream Setup (Mobile)

1.  Install **IP Camera Lite** on your iPhone/Android.
2.  Start the server and note the IPv4 address (e.g., `http://172.x.x.x:8081`).
3.  Update the `url` variable in `live_vein_hunter.py` with your credentials:
    ```python
    url = "http://admin:admin@your_ip_address:8081/video"
    ```
---

## Usage

### Real-Time Video Inference
```bash
python live_vein_hunter.py
```

### Static Image Analysis
```bash
python test_image.py --input path/to/arm_image.jpg
```

---

## Results

### Original vs. Mask vs. Overlay
![Original vs. Mask vs. Overlay](results/mask.jpeg)

### Real-time vein detection
![Real-time vein detection](results/realtime.jpeg)

-----

## Academic Report

The full methodology, ablation studies, and comparative results are detailed in our technical report: **"Vein Hunter: A Hybrid Computer Vision and Deep Learning Pipeline for Subcutaneous Vein Extraction"** (Formatted for NeurIPS).

-----

## Contributors

  * **Pragati Rokade** (B23CM1055) 
  * **Radhika Agarwal** (B23ES1027)
  * **Mahi Upadhyay** (B23ES1022)

-----

## License

Distributed under the MIT License. See `LICENSE` for more information.

-----

