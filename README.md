# Integrable Elasticity via Neural Demand Surfaces

A neural-network framework for estimating own- and cross-price elasticities of demand from scanner data, grounded in derivative-coherent demand modeling. The model learns a smooth context-dependent log-demand surface and obtains elasticities as exact derivatives with respect to log-prices. Evaluated on the Dominick's Finer Foods beer dataset against directed pairwise OLS, Ridge, and a demand-first MLP.

---

## Key idea

Classical demand estimation often fits separate regressions for product pairs, yielding elasticity estimates that can be noisy, unstable, and difficult to reconcile with a single demand representation. This project takes a demand-first route:

1. **Demand surface.** The model learns a context-dependent log-demand map

   $\hat{\mathbf{y}} = g_\theta(\mathbf{u}, \mathbf{x}),$

   where $\mathbf{u}$ denotes log-prices and $\mathbf{x}$ includes store, time, promotion, product, and competitive context.

2. **Elasticities by exact differentiation.** Own- and cross-price elasticities are obtained as the Jacobian of the fitted log-demand surface:

   $\hat E_{ij} = \frac{\partial g_{\theta,i}(\mathbf{u}, \mathbf{x})}{\partial u_j}.$

   This ties demand prediction and elasticity estimation to the same differentiable representation.

3. **Integrability / derivative coherence.** For each demand component, the elasticity row is the gradient of a single log-demand surface. This guarantees row-wise integrability and path-independent demand reconstruction, rather than treating elasticities as arbitrary local outputs.

4. **Directional cross-price effects.** Cross-price elasticities are learned as directional effects: the response of product $i$'s demand to product $j$'s price need not equal the reverse response. The model therefore does not impose $E_{ij}=E_{ji}$, Slutsky symmetry, or Hicksian symmetry.

5. **Flexible price response.** Nonlinear own- and cross-price effects are represented with product-specific cubic spline bases whose derivatives are available in closed form. This enables analytic elasticities, curvature regularization, and scalable training without dense automatic-differentiation Jacobians.

6. **Context-dependent parameters.** A shared product encoder maps product-level tokens—store, time, promotions, lags, product metadata, and competitive features—into the coefficients of the structured demand surface, allowing elasticities to vary across market conditions.

7. **Sparse cross-product interaction graph.** A sparse neighbor selector identifies relevant directed competitors per product using attention and metadata such as category, brand, style, and pack-size similarity. Softmax weights are masked by per-observation **availability**, so stocked-out neighbors receive zero weight. After training, the graph can be **frozen** (one pass over the training split, then a sparse $O(B \cdot n \cdot k)$ path).

---

## Architecture

```text
┌──────────────────────┐
│ Batch (store, week)  │
│ MultiProductDataset  │
│ wide-format panel    │
│ prices, obs_mask,    │
│ availability         │
└──────────┬───────────┘
           │
┌──────────┴──────────────────┐
▼                             ▼
┌────────────────────────┐    ┌──────────────────────────┐
│ ProductTokenBuilder    │    │ build_price_basis        │
│                        │    │ truncated_cubic or       │
│ store emb + Fourier +  │    │ natural_cubic            │
│ promo + per-product    │    │ Bx, dBx, ddBx            │
│ lags + competitive     │    │                          │
└────────────┬───────────┘    └────────────┬─────────────┘
             │ tokens (B,n,d)              │ spline outputs
             └────────────┬────────────────┘
                          ▼
                ┌────────────────────────────┐
                │ IntegrableDemandHead       │
                │                            │
                │ SharedProductEncoder → h   │
                │ SparseNeighborSelector →   │
                │   pairs, attn_weights      │
                │   (availability-masked)    │
                │ DemandParameterHead(h) →   │
                │   b, β, w (own)            │
                │   β_cross, w_cross, u      │
                │ DemandCalculator →         │
                │   ŷ, ε̂, E                 │
                └────────────────────────────┘
```

**`ICDN`** (Integrable Context-Dependent Demand Network) orchestrates the full forward pass: context token building, spline evaluation, sparse neighbor selection, and the integrable demand head.

Training is **two-phase**. Phase 0 freezes spline and bilinear heads (log-linear demand, negative-$\beta$ prior). Phase 1 unfreezes them and adds smoothness and elasticity-bound penalties. The compound loss is

$\mathcal{L} = \mathcal{L}_{\mathrm{Huber}} + \lambda_{\mathrm{smooth}}\,\mathcal{L}_{\mathrm{smooth}} + \lambda_{\mathrm{elast}}\,\mathcal{L}_{\mathrm{elast}},$

with soft bounds $E_{ii}\in[-5,0]$ and $E_{ij}\in[-1,1]$. Penalty, score, and export entries are gated by `elasticity_entry_mask` (observed demand and price on $i$; observed price and availability on $j$).

---

## Evaluation framework

All ICDN, MLP, and architecture-alternative runs use a **nested temporal** protocol: no future leakage (validation is always later than training).

| Dimension | Method | Data source |
|---|---|---|
| **Tuning** | Nested temporal Optuna (5 outer × 3 inner folds) | `results/nested/` |
| **Generalization** | Expanding-window outer folds × paired seeds | `data/nn_kfold_metrics_raw_nested.csv`, `data/nn_kfold_elasticities_raw_nested.csv`, `data/benchmark_*_kfold_*` |
| **Elasticity stability** | Block bootstrap (13-week blocks) with CI comparison | `data/nn_bootstrap_elasticities_raw_nested.csv`, `data/benchmark_*_bootstrap_*` |
| **Calibration** | Bootstrap CI coverage over k-fold point estimates | Cross-referencing bootstrap CIs with k-fold elasticities |
| **Architecture** | Fixed-config / equal-budget / confirmatory 2×2 | `results/architecture_alternatives/` |

The reported ICDN panel is **4 UPCs**. The stress test scales the same architecture synthetically up to $n=200$.

---

## Project structure

```text
nn-elasticity/
├── data/                    # Processed CSVs and evaluation outputs
├── results/                 # Nested Optuna DBs, checkpoints, ablation, stress, architecture runs
├── notebooks/
│   ├── preprocess-data.ipynb                       # Raw Dominick's data → dominick_features.csv
│   ├── creation-dataset.ipynb                      # Feature filtering → elasticity_dataset.csv
│   ├── hparam-search.ipynb                         # Nested temporal Optuna (ICDN)
│   ├── nn_final_evaluation.ipynb                   # Outer-fold + holdout eval + bootstrap
│   ├── benchmark-linear.ipynb                      # Pairwise OLS and Ridge (k-fold + bootstrap)
│   ├── benchmark-mlp.ipynb                         # Demand MLP, nested protocol
│   ├── ablation-study.ipynb                        # Nested holdout leave-one-out ablations
│   ├── architecture-alternatives.ipynb             # Basis × attention 2×2 (3 levels)
│   ├── analysis-result-architecture-alternative.ipynb
│   ├── stress-test.ipynb                           # Latency / memory vs (n, k)
│   └── analysis-results.ipynb                      # Head-to-head: ICDN vs OLS / Ridge / MLP
└── src/
    ├── dominick/                      # Dominick's data loading and processing
    │   ├── dataloader.py
    │   ├── dataprocess.py
    │   ├── datasaver.py
    │   ├── multiproduct_builder.py
    │   ├── multiproduct/              # Panel selection and wide-format pivot
    │   └── processors/                # Feature engineering
    │       ├── elasticity_features.py
    │       ├── financial_ratios.py
    │       ├── liter_metrics.py
    │       ├── text_normalizer.py
    │       └── unit_converter.py
    ├── multiproduct/                  # PyTorch dataset and context token builder
    │   ├── dataset.py                 # MultiProductDataset (includes availability)
    │   └── context.py                 # ProductTokenBuilder
    ├── nn/
    │   ├── models/
    │   │   ├── icdn.py
    │   │   └── integrable_demand_head.py
    │   ├── heads/
    │   │   ├── demand_calculator.py
    │   │   ├── elasticity_calculator.py
    │   │   ├── parameter_head.py
    │   │   └── neighbor_selector.py   # scaled_dot / additive; availability softmax
    │   ├── spline/
    │   │   ├── cubic_spline_basis.py
    │   │   ├── multi_cubic_spline_basis.py      # truncated-power (default)
    │   │   ├── multi_natural_cubic_spline_basis.py
    │   │   └── basis_factory.py                 # build_price_basis(...)
    │   ├── context/
    │   │   └── context_mlp.py
    │   ├── loss/
    │   │   ├── elasticity_loss.py
    │   │   ├── elasticity_mask.py
    │   │   └── components/
    │   ├── data/
    │   └── time_features/
    ├── benchmarks/
    │   ├── config.py
    │   ├── pairs.py
    │   ├── pairwise_ols.py
    │   ├── ridge.py
    │   ├── demand_mlp.py
    │   └── summarizer.py
    ├── eda/
    └── utils/
        └── splits.py                  # TemporalSplitter, BlockBootstrapSampler
```

---

## Pipeline

```text
Dominick's raw CSVs (upcber.csv, wber.csv)
        │
        ▼
preprocess-data.ipynb
        │
        ▼
dominick_features.csv
        │
        ▼
creation-dataset.ipynb
        │
        ▼
elasticity_dataset.csv  (store × UPC × week panel)
        │
        ├──────────────────┬──────────────────┬──────────────────┐
        ▼                  ▼                  ▼                  ▼
hparam-search.ipynb   benchmark-linear   benchmark-mlp    architecture-alternatives
nn_final_evaluation   (OLS + Ridge)      (nested MLP)     analysis-result-architecture-…
ablation-study
        │                  │                  │                  │
        ▼                  ▼                  ▼                  ▼
data/nn_*_nested.csv   data/benchmark_*.csv   data/benchmark_mlp_*_nested.csv
results/nested/        results/nested/ablation/   results/architecture_alternatives/
        │
        └──────────┬───────────────────┘
                   ▼
        analysis-results.ipynb
        (generalization, stability, calibration)
```

`stress-test.ipynb` is independent of the panel: it instantiates ICDN modules with best-trial hyperparameters and random weights, then times inference and a training step over a grid of $(n,k)$. Output: `results/stress_test_icdn.csv`.

---

## Data

The project uses the Dominick's Finer Foods dataset from the Kilts Center at Chicago Booth. Place the raw UPC and weekly store files (`upcber.csv`, `wber.csv`) in `data/` and run the preprocessing notebooks in order.

---

## Benchmarks

Three baselines share the same directed-pair construction and control set where applicable:

1. **Pairwise OLS** (`PairwiseElasticityPipeline`) — separate log-log regression per directed product pair within each store: $\log q_i = \alpha + \beta_i \log p_i + \gamma_{ij} \log p_j + X\delta + \varepsilon$. Inference uses HC1 robust standard errors and block bootstrap. Configuration: `BenchmarkConfig`. Notebook: `benchmark-linear.ipynb`.

2. **Pairwise Ridge** (`RegularizedElasticityPipeline`) — same grouping and formula as OLS; only the estimator changes (RidgeCV with internal alpha selection). Isolates the effect of coefficient shrinkage. Configuration: `RidgeConfig`. Notebook: `benchmark-linear.ipynb`.

3. **Demand MLP** (`DemandMLPPipeline`) — single global demand-first MLP trained on the dyadic dataset; elasticities via autodiff. Deliberately excludes ICDN's splines, sparse attention, bilinear cross potential, and elasticity penalties. Uses the same nested temporal protocol as ICDN. Configuration: `MLPConfig`. Notebook: `benchmark-mlp.ipynb`.

---

## Ablation study

`notebooks/ablation-study.ipynb` isolates ICDN component contribution with leave-one-out variants, each retuned on the nested holdout split (20 trials × 3 inner folds):

| Family | Variants |
|---|---|
| Module ablation | `full`, `no_smooth`, `no_elast`, `no_attention`, `no_cross`, `no_splines` |
| Constraint sensitivity | `free_sign`, `unconstrained`, `wide_bounds` |

Outputs land in `results/nested/ablation/`.

---

## Architecture alternatives

`notebooks/architecture-alternatives.ipynb` compares a $2\times 2$ of **price basis × attention score**, with analysis in `analysis-result-architecture-alternative.ipynb`:

| ID | Basis | Score |
|---|---|---|
| `TP-DOT` | truncated-power cubic (default ICDN) | scaled dot-product |
| `NC-DOT` | natural cubic | scaled dot-product |
| `TP-ADD` | truncated-power cubic | additive ($v^\top\tanh(W_q h_i+W_k h_j)$) |
| `NC-ADD` | natural cubic | additive |

Three levels, same training budget as `hparam-search.ipynb`:

1. **Fixed-config substitution** — identical hyperparameters, reference outer fold, paired seeds.
2. **Equal-budget screening** — shared hyperparameter grid; select by `robust_r2 = mean_r2 − 0.25 × std_r2` (predictive only).
3. **Confirmatory outer folds** — each model's own best config, 5 expanding folds × 5 paired seeds.

Artifacts: `results/architecture_alternatives/` (metrics, elasticities, attention edges, runtime/memory, figures).

---

## Stress test

`notebooks/stress-test.ipynb` measures ICDN latency and GPU memory across product-count / neighbor-count combinations $(n, k)$ using best-trial hyperparameters and random weights. The grid covers $n\in\{5,10,25,50,100,200\}$ and $k\in\{1,2,4,8,16,32\}$ plus the dense case $k=n-1$. Each point reports a frozen-graph inference breakdown **and** a full training step (online graph selection, AMP, AdamW). Results: `results/stress_test_icdn.csv`.

---

## Setup

```bash
git clone https://github.com/carlosherediapimienta/nn-elasticity.git
cd nn-elasticity
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Authors

**Researchers**: Carlos Heredia, PhD & Daniel Roncel

**Affiliation**: IAMMResearch
