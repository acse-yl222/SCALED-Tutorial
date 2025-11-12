## SCALED Tutorial (WP4)

This repository hosts the hands-on material for the WP4 segment of the **SCALED Tutorial**, where we practice building scalable, learning-enabled digital twins for physics-based systems. The codebase is intentionally lightweight: each lesson exposes a single Python entry point that you can expand with your own models, data loading logic, and evaluation pipelines.

### Learning Path

| Lesson | Folder | Focus |
| --- | --- | --- |
| L1 | `l1_regression_based_surrogate_model` |  |
| L2 | `l2_diffusion_based_surrogate_model` |  |
| L3 | `l3_compress_and_inference_model_trainning` |  |
| L4 | `l4_domain_decomposition_method` |  |

### Getting Started

1. **Clone** the repository and move into the project root.
   ```bash
   git clone https://github.com/acse-yl222/SCALED-Tutorial.git
   cd SCALED-Tutorial
   ```
2. **Create a Python environment** (3.10+ recommended).
   ```bash
   conda create -n scaled python=3.10
   pip install torch torchvision torchaudio
   pip install -r requirement.txt
   pip install -e .
   ```