# GRE²‑MDCL (PyTorch Geometric) — README

> **Graph Representation Embedding Enhanced via Multidimensional Contrastive Learning**  
> A single‑file, reproducible PyTorch/PyG implementation with local & global augmentations, a triple‑branch GNN, and multidimensional contrastive objectives. Works out‑of‑the‑box on Cora / Citeseer / PubMed with linear‑probe evaluation.

---

## 🚀 Highlights

- **Local Augmentation (G̃₁):** CVAE (LAGNN‑style) generates neighborhood‑consistent vectors conditioned on the center node; mixed back with raw features.
- **Global Augmentation (G̃₂):** SVD‑based low‑rank reconstruction. By default it factorizes the **feature matrix F**; optionally factorize **adjacency A**.
- **Encoder:** Multi‑head **GAT**. One **online** branch (with predictor) + **two EMA targets** (encoder+projector only).
- **Losses:**  
  - **Cross‑Network (L_cn):** online ↔ both targets (InfoNCE).  
  - **Cross‑View (L_cv):** inter‑view + intra‑view (shared negative pool).  
  - **Neighbor/Head Contrast (L_head):** multi‑positive contrast using graph neighbors, across attention heads and views.
- **Evaluation:** Freeze embeddings and train a **Logistic Regression** classifier on the standard Planetoid splits.

---

## 📦 Repository Layout

```
.
├── gre2_mdcl.py   # Self-supervised training + linear probe (single script)
└── README.md
```

> Datasets are auto‑downloaded by PyG into `./data/`.

---

## 🧰 Requirements

- Python 3.8–3.11
- PyTorch ≥ 2.0
- PyTorch Geometric (+ torch‑scatter / torch‑sparse / torch‑cluster / torch‑spline‑conv)
- scikit‑learn

Install example (CUDA 12.x; switch to CPU wheels if needed):
```bash
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install scikit-learn
```

> If PyG wheels fail to resolve, install versions matched to your PyTorch/CUDA per the PyG docs.

---

## ▶️ Quick Start

Run on **Cora**:
```bash
python gre2_mdcl.py --dataset cora --epochs 400 --hidden 256 --proj 128 --heads 4   --tau 1.0 --alpha 0.5 --beta 1.0 --gamma 1.0 --svd-k 64 --feature-drop 0.2 --attn-drop 0.4
```

Other datasets:
```bash
# Citeseer
python gre2_mdcl.py --dataset citeseer --epochs 400 --svd-k 64

# PubMed (higher feature dim; try larger k)
python gre2_mdcl.py --dataset pubmed --epochs 400 --svd-k 128
```

The script prints **Test Accuracy** of a logistic‑regression probe on frozen embeddings.

---

## ⚙️ CLI Options

| Flag | Description | Default |
|---|---|---|
| `--dataset {cora,citeseer,pubmed}` | Dataset name | `cora` |
| `--seed` | Random seed | `42` |
| `--epochs` | Self‑supervised epochs | `400` |
| `--hidden` | Hidden dim in GAT | `256` |
| `--embed` | Encoder output dim (pre‑projection) | `256` |
| `--proj` | Projection dim for contrastive space | `128` |
| `--heads` | Attention heads | `4` |
| `--layers` | GAT layers | `2` |
| `--tau` | Temperature for contrastive logits | `1.0` |
| `--alpha` | Weight of L_cn (and mixing two targets inside L_cn) | `0.5` |
| `--beta` | Weight of L_cv | `1.0` |
| `--gamma` | Weight of L_head | `1.0` |
| `--svd-k` | Truncation rank for SVD | `64` |
| `--svd-mode {features,adjacency}` | What to factorize for the global view | `features` |
| `--feature-drop` | Feature dropout when forming views | `0.2` |
| `--attn-drop` | Attention dropout in GAT | `0.4` |
| `--lr` | Adam learning rate | `1e-3` |

**Tips:** PubMed benefits from larger `--svd-k` (e.g., `128–256`). Reduce `--hidden`/`--heads`/`--svd-k` on low VRAM.

---

## 🧠 Method Overview (Implementation Notes)

- **Local view (CVAE):** Pretrain a conditional VAE on neighbor pairs `(v,u)` sampled from edges; then **freeze** it. At inference, generate a neighborhood‑consistent vector per node and mix with raw features.
- **Global view (SVD):** Low‑rank reconstruction `X_k` from SVD of `X` (default). Can switch to adjacency under `--svd-mode adjacency`.
- **Triple branches:** Online branch includes a **predictor** (BYOL‑style); two target branches are **EMA** copies (momenta `0.996` and `0.992`).
- **Objectives:**  
  - `L_cn`: align online predictions with two target projections (same‑index positives).  
  - `L_cv`: inter‑view alignment + intra‑view with a combined bank of both views.  
  - `L_head`: supervised contrast with **multi‑positives** defined by graph neighbors across heads/views.
- **Evaluation:** Normalize projected embeddings `z`, train scikit‑learn LogisticRegression on (train+val), report test.

---

## 🔬 Repro Tips

- **Stability:** 400 epochs typically suffice. If `L_cv` dominates while `L_head` is small, try increasing `--gamma` or lowering `--feature-drop`.
- **Temperature `tau`:** `0.5–1.0` usually stable; sparser graphs prefer slightly larger `tau`.
- **Randomness:** A global seed is set, but GAT and data splits can cause slight variance. Consider multiple runs and report mean±std.
- **Speed:** CVAE is pretrained once and then frozen. Cora/Citeseer train fast; PubMed is heavier.

---

## 🧯 Troubleshooting

- **PyG install issues:** Ensure PyTorch and CUDA versions match; choose compatible PyG wheels.  
- **Out‑of‑memory:** Lower `--hidden`, `--heads`, or `--svd-k`. Prefer `--svd-mode features` for memory savings.  
- **Low accuracy:** Increase `--epochs` or `--svd-k`; tune `--alpha/--beta/--gamma/--tau`. Check the three loss terms for balance.  
- **Slow on CPU:** Use a GPU if possible; otherwise reduce dimensions/epochs.

---

## 📄 Citation

If this implementation helps your research or product, please cite the original paper and reference this repo/implementation.

> *GRE²‑MDCL: Graph Representation Embedding Enhanced via Multidimensional Contrastive Learning* (see accompanying PDF).

---

## 🔧 Roadmap (Optional)

- [ ] Additional augmentations (feature masking, edge dropping)  
- [ ] Generalized positive sets (k‑hop / community)  
- [ ] Batch evaluation scripts & TensorBoard visualizations  
- [ ] Checkpointing & resume

---

## 🤝 Contributing

PRs/issues are welcome—especially for ablations, better hyper‑parameters, visualizations, and code cleanups.

---

## 📬 Support

Got questions? Share your command line, logs, and environment details and we’ll help debug.
