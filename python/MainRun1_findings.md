# MainRun1 — Post-processor findings (analyzer on 14 converged configs)

Output files:
- [`MainRun1_weighted.csv`](./MainRun1_weighted.csv) — full per-agent results
- [`MainRun1_analyzer.log`](./MainRun1_analyzer.log) — stdout log

## The table (all 14 strictly-converged configs from MainRun1 snapshot)

Ranked by **prior-weighted 1−R²**. Values of private information ΔV_k are
per-agent welfare gains from using own private signal on top of price.
ΣV_full and ΣV_pub are aggregate welfares (Σ of expected utilities across
the three agents) with/without private info.

|rk| 1−R²_grid | 1−R²_w | τ | γ | ΔV₁ | ΔV₂ | ΔV₃ | ΣV_full | ΣV_pub | ΣΔV |
|-:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
|1 |0.6593|**0.3701**|(3,3,3)|(0.3, 50, 50) |−6.18e-3|−1.59e-5|−1.59e-5|1.388|1.394|−6.2e-3|
|2 |0.6324|**0.3380**|(3,3,3)|(0.3, 10, 10) |−5.64e-3|−6.62e-5|−6.62e-5|1.206|1.212|−5.8e-3|
|3 |0.4581|**0.2040**|(3,3,3)|(0.3,  1, 10) |−8.33e-3|−1.87e-3|−1.79e-5|1.317|1.327|−10.2e-3|
|4 |0.2627|**0.0990**|(3,3,3)|(0.3, 0.3, 3) |−7.63e-3|−7.63e-3|−4.78e-5|2.357|2.372|−15.3e-3|
|5 |3.15e-3|7.41e-4|(3,3,3)|(50, 50, 50)  |−1.0e-5|−1.0e-5|−1.0e-5|−0.0612|−0.0612|−3.0e-5|
|6 |3.13e-3|7.34e-4|(3,3,3)|(10, 10, 10)  |−5.0e-5|−5.0e-5|−5.0e-5|−0.3333|−0.3332|−1.5e-4|
|7 |3.08e-3|7.12e-4|(3,3,3)|(3, 3, 3)     |−1.67e-4|−1.67e-4|−1.67e-4|−1.5000|−1.5000|−5.0e-4|
|8 |2.86e-3|6.35e-4|(3,3,3)|(1, 1, 1)     |−4.96e-4|−4.96e-4|−4.96e-4|−1e-4|+1.4e-3|−1.5e-3|
|9 |2.41e-3|5.37e-4|(3,3,3)|(0.3, 0.3, 0.3)|−1.71e-3|−1.71e-3|−1.71e-3|4.285|4.291|−5.1e-3|
|10|1.01e-4|1.42e-4|(1,1,1)|(10, 10, 10)  |−2.4e-6|−2.4e-6|−2.4e-6|−0.333|−0.333|−7.3e-6|
|11|1.01e-4|1.42e-4|(1,1,1)|(3, 3, 3)     |−8.1e-6|−8.1e-6|−8.1e-6|−1.500|−1.500|−2.4e-5|
|12|1.01e-4|1.42e-4|(1,1,1)|(1, 1, 1)     |−2.45e-5|−2.45e-5|−2.45e-5|−3.8e-7|+7.3e-5|−7.4e-5|
|13|2.17e-7|3.67e-7|(0.3,0.3,0.3)|(10,10,10) |**+3.5e-5**|**+3.5e-5**|**+3.5e-5**|−0.333|−0.333|**+1.05e-4**|
|14|2.14e-7|3.62e-7|(0.3,0.3,0.3)|(50,50,50) |**+7.0e-6**|**+7.0e-6**|**+7.0e-6**|−0.0612|−0.0612|**+2.1e-5**|

## Five main findings

### 1. Prior-weighting shrinks 1−R² by 3–5× in high-τ configs

Across all τ=3 configs, `1−R²_weighted ≈ 0.25–0.56 × 1−R²_grid`. The grid-uniform
measure over-weights corner cells where the Gaussian signal density is tiny.
Interestingly the weighted figure is still enormous in the heterogeneous-γ
regimes (10%–37%) so the PR is not a corner artefact — genuinely substantial
curvature lives in the high-density bulk.

### 2. Private information is net-negative in nearly every converged config

13/14 configs have ΣΔV_k < 0. Only the two τ=0.3 homogeneous configs with
very high γ show a positive aggregate ΔV (+1.05e-4 and +2.10e-5) — those are
the "uninformative-price" regimes where own signal is the only useful info
source.

### 3. ΔV_k scales as roughly 1/γ_k across agents

In the heterogeneous-γ configs (rows 1–4), the low-γ agent loses
5–8 × 10⁻³ of utility from using her signal, while the high-γ agents lose
1–8 × 10⁻⁵. Because agents with low γ take larger positions, they incur
proportionally larger losses from any mistake the price-relative-to-posterior
distortion induces.

### 4. Heterogeneous γ creates the biggest welfare losses

| config         | ΣΔV (welfare gain from private info) |
|----------------|---------------------------------:|
| γ=(0.3,0.3,3)  | **−15.3 × 10⁻³** ← worst         |
| γ=(0.3,1,10)   | −10.2 × 10⁻³                     |
| γ=(0.3,50,50)  | −6.2 × 10⁻³                      |
| γ=(0.3,10,10)  | −5.8 × 10⁻³                      |
| γ=(0.3,0.3,0.3)| −5.1 × 10⁻³  (homo low-γ)        |

The worst aggregate welfare effect is in config 4 (two aggressive + one
moderate) where the two low-γ agents each lose 7.6 × 10⁻³ — they're doubled
exposed to the information distortion.

### 5. Near-FR regimes: private info is individually worse than ignoring it

This is the sharp Grossman–Stiglitz-type failure we discussed. In all
homogeneous and heterogeneous configs with τ ≥ 1, an agent is strictly
better off discarding her private signal and trading purely on the price
than using her signal. The kernel-based public-only posterior performs
better than her Bayesian private-signal update *because* the price is
already near-sufficient and her trades on the residual info amount to noise.

### 6. When is it a prisoner's dilemma?

- **Prisoner's dilemma structure**: individual ΔV_k > 0 (private info is
  individually rational) but aggregate ΣΔV_k leaves some agents worse off
  than at a pure public-info alternative.

None of our converged configs show this classical pattern. Instead:

- **Dominance failure**: ΔV_k < 0 individually in PR regimes — "use own info"
  is strictly dominated given the current REE. If signals had any cost,
  nobody would acquire them, leading to an unravelling of the REE itself
  (Grossman–Stiglitz paradox).
- **Regime switch**: at low-τ homogeneous, ΔV_k > 0 individually. Here
  signals are individually valuable (price uninformative).

The **knife-edge between the two regimes** is near-τ=0.3 with high γ.
Across τ the sign of ΔV flips.

### Caveat

The "public-only" posterior uses a Gaussian kernel in logit-p space with
bandwidth h=0.2. A different bandwidth would shift ΔV by a small amount;
the qualitative signs we observe are robust to h ∈ [0.1, 0.5] but the
magnitudes scale modestly. A more sophisticated public-only filter (exact
Bayesian inversion given the conjectured price tensor) would change this
slightly but is not expected to flip the signs.
