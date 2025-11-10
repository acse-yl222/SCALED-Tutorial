## SCALED Tutorial (WP4)

This repository hosts the hands-on material for the WP4 segment of the **SCALED Tutorial**, where we practice building scalable, learning-enabled digital twins for physics-based systems. The codebase is intentionally lightweight: each lesson exposes a single Python entry point that you can expand with your own models, data loading logic, and evaluation pipelines.

### Learning Path

| Lesson | Folder | Focus |
| --- | --- | --- |
| L1 | `compression_physics_information/l1.py` | Compress high-dimensional simulation fields into compact latent representations while preserving key physics constraints. |
| L2 | `diffusion_based_surrogate_model/l2.py` | Train a diffusion-based surrogate to emulate expensive simulators and quantify uncertainty. |
| L3 | `domain_decomposition_method/l3.py` | Couple local surrogates through a domain-decomposition strategy to scale to large geometries. |
| L4 | `inference_using_latent_diffusion_model/l4.py` | Perform inverse inference and design-space exploration using latent diffusion models. |

Each module is self-contained, but the lessons build on one another: outputs from L1 feed the surrogate in L2, domain decomposition techniques in L3 assume a trained surrogate, and L4 reuses the latent space learned earlier for downstream inference.

### Repository Layout

```
SCALED-Tutorial
├── README.md                          # You are here
├── compression_physics_information
│   └── l1.py                          # Lesson 1 entry script
├── diffusion_based_surrogate_model
│   └── l2.py                          # Lesson 2 entry script
├── domain_decomposition_method
│   └── l3.py                          # Lesson 3 entry script
└── inference_using_latent_diffusion_model
    └── l4.py                          # Lesson 4 entry script
```

### Getting Started

1. **Clone** the repository and move into the project root.
   ```bash
   git clone <your-fork-url>
   cd SCALED-Tutorial
   ```
2. **Create a Python environment** (3.10+ recommended).
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   ```
3. **Install dependencies** that you need for the lesson you are implementing (e.g., `torch`, `numpy`, `pytorch-lightning`, `diffusers`). You can maintain a `requirements.txt` as the project evolves:
   ```bash
   pip install -r requirements.txt  # Optional, if you create one
   ```

### Working Through the Lessons

- **Implement the scaffold** inside each `lX.py`. Organize code into reusable functions or modules if the script grows large.
- **Track experiments** by logging checkpoints, losses, and metrics (TensorBoard, Weights & Biases, or simple CSV logs all work well).
- **Persist data paths** via environment variables or a lightweight config file so the scripts stay portable.
- **Evaluate continuously**: add quick smoke tests that import each lesson module and validate shapes/dimensions before launching long trainings.

### Suggested Development Workflow

1. Prototype core logic inside `lX.py`.
2. Extract reusable utilities into helper modules as duplication appears.
3. Add docstrings/comments that clarify physics assumptions, boundary conditions, and latent representations.
4. Save trained artifacts (latents, surrogate weights, domain partitions, diffusion checkpoints) under a clearly named `outputs/` directory; keep large files out of version control or leverage Git LFS.
5. Document findings in this README (per lesson) once you complete an experiment to keep the tutorial reproducible.

### Contributing / Extending

- Fork the repository, work on a feature branch, and open a pull request describing the motivation and results.
- Prefer deterministic seeds when possible to make reviews easier.
- If you benchmark against a baseline simulator, summarize the setup and metrics so others can replicate the comparison.

### Support & Further Reading

- Diffusion models for scientific ML (e.g., *Diffusion Models Beat GANs on Image Synthesis* for fundamentals, *Latent Diffusion Models* for efficiency considerations).
- Domain decomposition references such as *Smith et al., Domain Decomposition: Parallel Algorithms and Their Convergence*.
- For physics-latent compression, variational autoencoders (VAE) and physics-informed neural networks (PINN) literature provide useful architectural ideas.

If you spot issues or have suggestions for new lessons, feel free to open an issue or start a discussion thread. Happy experimenting!
