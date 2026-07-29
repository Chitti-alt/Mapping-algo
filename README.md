# DeepGeo-Mapper: On-Loop Aerial Mosaic Stitcher

An end-to-end, single-file aerial mosaic stitching pipeline built for real-time UAV mapping missions on **NVIDIA Jetson** and **ROS 2** environments. 

By merging deep learning-based feature extraction (`SuperPoint`) and matching (`LightGlue`) with classic OpenCV detail-blending algorithms, `mapper.py` dynamically builds a unified, geocanonically-aligned orthomosaic from incoming aerial frames.

---

## Key Features

* **Deep Learning Registration:** Replaces brittle traditional feature extractors with **SuperPoint** keypoints and **LightGlue** matching for robust registration over challenging terrain and low-texture agricultural/rural areas.
* **Real-Time Directory Watcher:** Continually polls an input directory (`/4thave`) for newly dropped image frames, processing them sequentially in real time.
* **Robust Seam Blending & Deghosting:** Features multi-mode blending (`seam`, `multiband`, `feather`, `overwrite`). Integrates `cv2.detail_DpSeamFinder` to prevent ghosting across overlapping moving objects.
* **Illumination Compensation:** Employs dynamic luminance mean/standard-deviation matching (ICM) and CLAHE normalisation to balance out solar glare, cloud cover shadows, and colour-cast terrain.
* **Heading De-rotation:** Automatically aligns frames to a canonical north-up heading by pulling camera yaw from EXIF metadata or an optional `poses.csv` sidecar file.
* **Autonomous Idle-Timeout:** Features a configurable idle-timeout (`300s` default) that automatically saves the final composite map and shuts down once the mapping sortie ends.

---

## Supported Environment & Dependencies

This codebase is validated against the following specific dependency matrix:

| Package | Expected Version | Note |
| :--- | :--- | :--- |
| **Python** | `3.10+` | Recommended for ROS 2 Humble / Jetson Pack environments |
| **NumPy** | `1.26.4` | Version guard enforced at runtime |
| **OpenCV (`cv2`)** | `4.13.0` | Provides GraphCut, DP Seam Finders, and Voronoi fallback[cite: 1] |
| **PyTorch (`torch`)**| `2.8.0` | GPU acceleration (`cuda`) supported & recommended[cite: 1] |
| **LightGlue** | *Latest* | Provides `SuperPoint` and `LightGlue` models[cite: 1] |
| **Pillow (`PIL`)**| *Latest* | EXIF GPS/metadata parsing[cite: 1] |

### Installation

```bash
# Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/DeepGeo-Mapper.git
cd DeepGeo-Mapper

# Install Python dependencies
pip install numpy==1.26.4 opencv-python==4.13.0 torch==2.8.0 pillow lightglue
