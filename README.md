# Analytical Approximations and Monte Carlo Benchmarks for Up-and-Out Put Pricing

Reproduces the CEV, lognormal-SABR, and rough-Bergomi numerical experiments for
short-term up-and-out put barrier options described in the accompanying paper.

## Models implemented

| Experiment | Model | Key parameters |
|------------|-------|----------------|
| `exp_1` | CEV (`dS = σ Sᵝ dW`) | β = 0.5, σ = 2, θ = 1 |
| `exp_2` | Lognormal SABR | ν = 0.2, ρ = −0.9, σ₀ = 0.2, θ = 1 |
| `exp_3` | Rough Bergomi | H = 0.1, ν = 1, ρ = −0.9, ξ₀ = 0.04, θ = 0.25 |

Common contract: `S₀ = 100`, `B = 120`, `K ∈ {80, 82, …, 120}`.

## Repository layout

```
src/             Core modules (closed-form formulas, MC simulators, PDE solver)
exp/             Experiment scripts (exp1_cev.py, exp2_sabr.py, exp3_rbergomi.py)
tests/           pytest test suite (177 tests)
results/         Output CSV tables, PNG figures, RESULTS.md summary
data/            Reserved for future data artefacts
```

## Quick start

```bash
pip install numpy scipy pandas matplotlib
python exp/exp1_cev.py        # ~3 min  (200k paths, PDE)
python exp/exp2_sabr.py       # ~2 min  (200k paths)
python exp/exp3_rbergomi.py   # ~8 min  (200k rough-Bergomi paths via FFT)
pytest tests/ -v              # 177 tests, <10 s
```

## Methodology

1. **Closed-form baseline** (`src/closed_form.py`): Black-Scholes up-and-out put
   via the reflection formula, Bachelier leading term P₀, difference correction
   `p_diff = P₀ + β(p_BS^UO − P₀)`, and explicit gamma correction
   `p_gamma = p_BS^UO + c S₀ v₀ θ γ(b, k)`.

2. **Monte Carlo with control variates** (`src/{cev,sabr,rbergomi}_mc.py`):
   10 000 pilot paths fit per-strike control coefficients `β̂_K`; 200 000
   production paths compute `p̂_K = mean(Y_K) − β̂_K (mean(Y_B) − (B−S₀))`.

3. **CEV PDE benchmark** (`src/cev_pde.py`): Crank-Nicolson with 4 initial
   backward-Euler steps on `[0, B]`, two grid resolutions (1200×2048 and
   2400×4096) for Richardson extrapolation.

4. **Rough Bergomi Volterra simulation** (`src/rbergomi_mc.py`): Gaussian
   Volterra driver `Z_{t_i} = Σ κ_{i−j} ΔW_j` computed via FFT-based
   linear convolution; log-price updated as
   `X_{i+1} = X_i − ½V_i Δt + √V_i (ρ ΔW_i + √(1−ρ²) ΔW_i^⊥)`.

## Results summary

See [`results/RESULTS.md`](results/RESULTS.md) for full tables and figures.

| Experiment | BS MAE | Corrected MAE | Reduction |
|------------|--------|--------------|-----------|
| CEV | 0.104 | 0.005 | 94.9% |
| SABR | 0.155 | 0.050 | 67.7% |
| Rough Bergomi | 0.291 | 0.151 | 48.3% |
