# Integrable Elasticity via Neural Demand Potentials

A neural network framework for estimating own- and cross-price elasticities of demand from scanner data, grounded in microeconomic integrability theory. The model derives demand functions as gradients of a scalar potential, guaranteeing symmetric cross-price effects by construction. Evaluated on Dominick's Finer Foods retail dataset against a pairwise log-log OLS benchmark.

---

## Key idea

Classical demand estimation fits separate regressions per product pair, yielding elasticity matrices that are neither symmetric nor derived from a coherent utility-maximization problem. This project takes a different route:

1. **Demand potential.** A scalar function $\Phi(\mathbf{x}, \mathbf{c})$ of log-prices $\mathbf{x}$ and context $\mathbf{c}$ (store, time, promotions) is parameterized by a neural network. Predicted log-demand for product $i$ is obtained as:

   $$
   \hat{y}_i = \frac{\partial \Phi}{\partial x_i}
   $$

2. **Integrability by construction.** Because demand is the gradient of a single potential, the Jacobian $\partial \hat{y}_i / \partial x_j$ is the Hessian of $\Phi$ — symmetric by construction. Cross-price elasticities satisfy $E_{ij} = E_{ji}$ without post-hoc symmetrization.

3. **Flexible price response.** Non-linearity in prices is captured by cubic spline bases with analytically available derivatives (up to third order) and antiderivatives, enabling closed-form elasticities, curvature penalties, and cross-product potential terms.

4. **Context-dependent parameters.** An MLP maps store embeddings, Fourier time features, and product-level covariates to the spline coefficients and cross-product weights, so the demand surface adapts to heterogeneous market conditions.

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
│ MultiProductContext    │    │ MultiCubicSplineBasis│
│ Embeddings             │    │                      │
│                        │    │ Bx, dBx, ddBx,      │
│ store emb + Fourier +  │    │ dddBx, IBx          │
│ promo + product feats  │    │                     │
└────────────┬───────────┘    └──────────┬───────────┘
             │ context c                  │ spline outputs
             └────────────┬───────────────┘
                          ▼
                ┌────────────────────────┐
                │ IntegrableDemandHead   │
                │                        │
                │ ContextMLP(c) → h      │
                │ ParameterHead(h) →     │
                │   b, β, w (own)        │
                │   u (cross potential)  │
                │ DemandCalculator →     │
                │   ŷ, ε̂, E             │
                └────────────────────────┘
```

**`ICDN`** (Integrable Context-Dependent Demand Network) orchestrates the full forward pass: context embeddings, spline evaluation, and the integrable demand head.

## Loss function

$$
\mathcal{L}
=
\underbrace{\text{Huber}(\hat{y}, y)}_{\text{fit}}
+
\lambda_s \underbrace{\mathbb{E}\left[\left(\frac{\partial^2 \hat{y}_i}{\partial x_i^2}\right)^2\right]}_{\text{smoothness}}
+
\lambda_p \underbrace{\mathbb{E}\left[\max(0,\hat{\varepsilon}_i)\right]}_{\text{positivity penalty}}
$$

The smoothness term regularizes curvature of the demand surface in price space. The positivity penalty encodes the economic prior that own-price elasticities should be negative (downward-sloping demand).

## Evaluation framework

| Dimension | Method | Data source |
|---|---|---|
| **Generalization** | Temporal k-fold CV (expanding window) | `nn_kfold_metrics_raw.csv`, `benchmark_kfold_raw.csv` |
| **Elasticity stability** | Block bootstrap with CI comparison | `nn_bootstrap_own_summary.csv`, `benchmark_elasticities_bootstrap_summary.csv` |
| **Calibration** | Bootstrap CI coverage over k-fold point estimates | Cross-referencing bootstrap CIs with k-fold elasticities |

The temporal splitter ensures no future leakage: validation folds are always chronologically after training data.

## Project structure

nn-elasticity/
├── data/                    # Processed CSVs and evaluation outputs
├── results/                 # Optuna DB, best hyperparameters, checkpoints
├── notebooks/
│   ├── preprocess-data.ipynb        # Raw Dominick's data → dominick_features.csv
│   ├── creation-dataset.ipynb       # Feature filtering → elasticity_dataset.csv
│   ├── particular-eda-upc-store.ipynb # Focused EDA per store × UPC
│   ├── hparam-search.ipynb          # Optuna hyperparameter optimization
│   ├── nn_final_evaluation.ipynb    # Full evaluation of best trial (k-fold + bootstrap)
│   ├── benchmark.ipynb              # Pairwise OLS benchmark (k-fold + bootstrap)
│   └── analysis-results.ipynb       # Head-to-head comparison: NN vs benchmark
├── src/
│   ├── dominick/            # Data loading, processing, multiproduct pivot
│   ├── processors/          # Unit conversion, financial ratios, features
│   └── multiproduct/        # Panel selection, filtering, wide-format pivot
├── multiproduct/            # PyTorch dataset and context embeddings
├── nn/
│   ├── models/              # ICDN, IntegrableDemandHead
│   ├── heads/               # DemandCalculator, ElasticityCalculator, ParameterHead
│   ├── spline/              # Cubic spline basis with derivatives and antiderivatives
│   ├── context/             # ContextMLP
│   ├── loss/                # ElasticityLoss and components
│   ├── diagnostics/         # Integrability closure and Slutsky symmetry checks
│   ├── data/                # ColumnEncoder, SplineBuilder, DataLoaderFactory
│   └── time_features/       # Fourier seasonal features
├── benchmark/               # Pairwise log-log OLS pipeline
│   ├── pairs.py             # Long → directed pair dataset
│   ├── pairwise_ols.py      # OLS fitting with robust SEs
│   ├── symmetrizer.py       # Cross-elasticity symmetrization
│   └── summarizer.py        # Bootstrap aggregation
├── eda/                     # Exploratory data analysis modules
└── utils/                   # TemporalSplitter, BlockBootstrapSampler

## Pipeline

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
elasticity_dataset.csv (store × UPC × week panel)
        │
        ├──────────────────────────────┐
        ▼                              ▼
hparam-search.ipynb               benchmark.ipynb
nn_final_evaluation.ipynb         (pairwise OLS)
(ICDN training + eval)
        │                              │
        ▼                              ▼
nn_kfold_*.csv                   benchmark_*.csv
nn_bootstrap_*.csv               benchmark_bootstrap*.csv
        │                              │
        └──────────┬───────────────────┘
                   ▼
         analysis-results.ipynb
(generalization, stability, calibration)

## Data

The project uses the Dominick's Finer Foods dataset from the Kilts Center at Chicago Booth. Place the raw UPC and weekly store files (`upcber.csv`, `wber.csv`) in `data/` and run the preprocessing notebooks in order.

## Benchmark

The OLS benchmark fits a separate log-log regression for each directed product pair within each store:

$$
\log q_i = \alpha + \beta_i \log p_i + \gamma_{ij} \log p_j + X\delta + \varepsilon
$$

where $X$ includes promotion indicators, seasonal harmonics, lags, rolling means, and trend controls. Cross-elasticities are symmetrized by averaging both directions of each canonical pair. Inference uses HC1 robust standard errors and block bootstrap.

## Setup

git clone https://github.com/carlosherediapimienta/nn-elasticity.git
cd nn-elasticity
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## Authors: 
**Researchers**: Carlos Heredia, PhD & Daniel Roncel
**Affiliation**: IAMMResearch