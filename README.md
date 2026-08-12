# Integrable Elasticity via Neural Demand Surfaces

A neural-network framework for estimating own- and cross-price elasticities of demand from scanner data, grounded in derivative-coherent demand modeling. The model learns a smooth context-dependent log-demand surface and obtains elasticities as exact derivatives with respect to log-prices. Evaluated on the Dominick's Finer Foods beer dataset against a directed pairwise log-log OLS benchmark.

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

7. **Sparse cross-product interaction graph.** A sparse neighbor selector identifies relevant directed competitors per product using attention and metadata such as category, brand, style, and pack-size similarity. This keeps the cross-price component scalable while preserving heterogeneous substitution and complementarity patterns.

---

## Architecture

```text
┌──────────────────────┐
│ Batch (store, week)  │
│ MultiProductDataset  │
│ wide-format panel    │
└──────────┬───────────┘
           │
┌──────────┴──────────────────┐
▼                             ▼
┌────────────────────────┐    ┌──────────────────────┐
│ ProductTokenBuilder    │    │ MultiCubicSplineBasis │
│                        │    │                      │
│ store emb + Fourier +  │    │ Bx, dBx, ddBx        │
│ promo + per-product    │    │                      │
│ lags + competitive     │    │                      │
└────────────┬───────────┘    └──────────┬───────────┘
             │ tokens (B,n,d)            │ spline outputs
             └────────────┬──────────────┘
                          ▼
                ┌────────────────────────────┐
                │ IntegrableDemandHead       │
                │                            │
                │ SharedProductEncoder → h   │
                │ SparseNeighborSelector →   │
                │   pairs, attn_weights      │
                │ DemandParameterHead(h) →   │
                │   b, β, w (own)            │
                │   α, u (cross potential)   │
                │ DemandCalculator →         │
                │   ŷ, ε̂, E                 │
                └────────────────────────────┘
```

**`ICDN`** (Integrable Context-Dependent Demand Network) orchestrates the full forward pass: context token building, spline evaluation, sparse neighbor selection, and the integrable demand head.

---

## Evaluation framework

| Dimension | Method | Data source |
|---|---|---|
| **Generalization** | Temporal k-fold CV (expanding window) | `nn_kfold_metrics_raw.csv`, `nn_kfold_elasticities_raw.csv`, `benchmark_kfold_raw.csv` |
| **Elasticity stability** | Block bootstrap with CI comparison | `nn_bootstrap_elasticities_raw.csv`, `benchmark_bootstrap_raw.csv`, `benchmark_elasticities_bootstrap_summary.csv` |
| **Calibration** | Bootstrap CI coverage over k-fold point estimates | Cross-referencing bootstrap CIs with k-fold elasticities |

The temporal splitter ensures no future leakage: validation folds are always chronologically after training data.

---

## Project structure

```text
nn-elasticity/
├── data/                    # Processed CSVs and evaluation outputs
├── results/                 # Optuna DB, best hyperparameters, checkpoints, ablation + stress outputs
├── notebooks/
│   ├── preprocess-data.ipynb          # Raw Dominick's data → dominick_features.csv
│   ├── creation-dataset.ipynb         # Feature filtering → elasticity_dataset.csv
│   ├── hparam-search.ipynb            # Optuna hyperparameter optimization
│   ├── nn_final_evaluation.ipynb      # Full evaluation of best trial (k-fold + bootstrap)
│   ├── benchmark.ipynb                # OLS / Ridge / MLP benchmarks (k-fold + bootstrap)
│   ├── ablation-study.ipynb           # Leave-one-out ICDN component ablation
│   ├── stress-test.ipynb              # ICDN forward-pass latency/memory vs (n, k)
│   └── analysis-results.ipynb         # Head-to-head: ICDN vs OLS / Ridge / MLP
└── src/
    ├── dominick/                      # Dominick's data loading and processing
    │   ├── dataloader.py              # Raw CSV loader
    │   ├── dataprocess.py             # Processing pipeline
    │   ├── datasaver.py               # Saving utilities
    │   ├── multiproduct_builder.py    # Orchestrates multiproduct pivot
    │   ├── multiproduct/              # Panel selection and wide-format pivot
    │   │   ├── filter_complete.py
    │   │   ├── panel_selector.py
    │   │   └── pivot.py
    │   └── processors/                # Feature engineering
    │       ├── elasticity_features.py
    │       ├── financial_totals.py
    │       ├── liter_metrics.py
    │       ├── text_normalizer.py
    │       └── unit_converter.py
    ├── multiproduct/                  # PyTorch dataset and context token builder
    │   ├── dataset.py                 # MultiProductDataset (wide-format panel)
    │   └── context.py                 # ProductTokenBuilder
    ├── nn/
    │   ├── models/
    │   │   ├── icdn.py                # ICDN: top-level nn.Module
    │   │   └── integrable_demand_head.py  # IntegrableDemandHead
    │   ├── heads/
    │   │   ├── demand_calculator.py       # DemandCalculator
    │   │   ├── elasticity_calculator.py   # ElasticityCalculator
    │   │   ├── parameter_head.py          # DemandParameterHead
    │   │   └── neighbor_selector.py       # SparseNeighborSelector
    │   ├── spline/
    │   │   ├── cubic_spline_basis.py      # Per-product spline basis
    │   │   └── multi_cubic_spline_basis.py # Vectorized multi-product spline
    │   ├── context/
    │   │   └── context_mlp.py             # SharedProductEncoder
    │   ├── loss/
    │   │   ├── elasticity_loss.py         # ElasticityLoss (Huber + smoothness + positivity + cross)
    │   │   └── components/
    │   │       ├── fit_loss.py
    │   │       ├── smoothness_penalty.py
    │   │       ├── positivity_penalty.py
    │   │       └── curvature_calculator.py
    │   ├── data/
    │   │   ├── dataset/
    │   │   │   └── dataloader_factory.py  # DataLoaderFactory
    │   │   ├── preprocessing/
    │   │   │   └── column_encoder.py      # ColumnEncoder
    │   │   └── spline/
    │   │       ├── knot_generator.py
    │   │       ├── spline_builder.py      # SplineBuilder
    │   │       └── statistics_calculator.py
    │   └── time_features/
    │       └── fourier_time_features.py   # Fourier seasonal features
    ├── benchmarks/                    # Baseline demand / elasticity pipelines
    │   ├── config.py                  # BenchmarkConfig, RidgeConfig, MLPConfig
    │   ├── pairs.py                   # Long → directed pair dataset
    │   ├── pairwise_ols.py            # Pairwise OLS with robust SEs
    │   ├── ridge.py                   # Pairwise RidgeCV (same grouping as OLS)
    │   ├── demand_mlp.py              # Global demand-first MLP + autodiff elasticities
    │   └── summarizer.py              # Bootstrap aggregation
    ├── eda/                           # Exploratory data analysis
    │   ├── eda.py
    │   └── functions/
    │       ├── competitors/           # Competitive feature builders
    │       ├── grain/                 # Panel balance, coverage, gap imputation
    │       ├── missing_data/          # NaN analysis
    │       ├── outliers/              # Outlier detection
    │       ├── price_variation/       # Log-price/demand, collinearity, baseline OLS
    │       └── time_series/           # Trends, autocorrelation, temporal features
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
        ├──────────────────────────────┐
        ▼                              ▼
hparam-search.ipynb           benchmark.ipynb
nn_final_evaluation.ipynb     (pairwise OLS)
(ICDN training + eval)
        │                              │
        ▼                              ▼
nn_kfold_*.csv              benchmark_*.csv
nn_bootstrap_*.csv          benchmark_bootstrap*.csv
        │                              │
        └──────────┬───────────────────┘
                   ▼
        analysis-results.ipynb
        (generalization, stability, calibration)
```

---

## Data

The project uses the Dominick's Finer Foods dataset from the Kilts Center at Chicago Booth. Place the raw UPC and weekly store files (`upcber.csv`, `wber.csv`) in `data/` and run the preprocessing notebooks in order.

---

## Benchmark

Three baselines share the same directed-pair construction and control set where applicable:

1. Pairwise OLS (PairwiseElasticityPipeline) — separate log-log regression per directed product pair within each store: $\log q_i = \alpha + \beta_i \log p_i + \gamma_{ij} \log p_j + X\delta + \varepsilon$. Inference uses HC1 robust standard errors and block bootstrap. Configuration: BenchmarkConfig.

2. Pairwise Ridge (RegularizedElasticityPipeline) — same grouping and formula as OLS; only the estimator changes (RidgeCV with internal alpha selection). Isolates the effect of coefficient shrinkage. Configuration: RidgeConfig.

3. Demand MLP (DemandMLPPipeline) — single global demand-first MLP trained on the dyadic dataset; elasticities via autodiff. Deliberately excludes ICDN's splines, sparse attention, bilinear cross potential, and elasticity penalties. Configuration: MLPConfig.

---

## Ablation study

notebooks/ablation-study.ipynb isolates ICDN component contribution with leave-one-out variants (e.g. full, no_smooth, no_elast, no_attention, no_cross, no_splines, plus constraint / sign variants). Each variant runs a small Optuna search; outputs land in results/ablation_*.

---

## Stress test

notebooks/stress-test.ipynb measures ICDN forward-pass latency and memory across product-count / neighbor-count combinations $(n, k)$ using best-trial hyperparameters and random weights. Results: results/stress_test_icdn.csv.

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
