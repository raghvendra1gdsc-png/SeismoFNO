# SeismoFNO: Physics-Grounded Neural Operators for Seismic Structural Response Prediction
## A full walkthrough of the project — motivation, methods, experiments, and open problems

*Raghvendra Singh Gahlot · B.Tech (Civil Engineering) · September 2026*

---

## Why I started this

I got interested in this after running a handful of nonlinear time-history analyses in OpenSeesPy for a coursework assignment and noticing how long a single run took to converge, even for a small five-story shear frame. Regional risk assessment, city-scale loss estimation, and design-optimisation workflows all need this kind of analysis run thousands of times over. That gap between what rigorous simulation costs and what applications need felt like the right place to ask whether a learned surrogate could help.

What I didn't expect going in was how much the standard recipe — Fourier Neural Operators on a fixed grid — breaks the moment you try to apply it to buildings that don't all have the same number of storeys. That failure, and the two fixes that followed it, is most of what this project is about. I've tried to be as honest about what still doesn't work as about what does; I think that's the more useful document to hand someone evaluating this kind of work.

---

## 1. Foundations: getting the ground truth right first

Before any surrogate modelling, the first stretch of this project was just building a trustworthy simulation pipeline and confirming the physics was sound end-to-end. Concretely, that meant:

- Validating single-degree-of-freedom elastoplastic dynamics and Bouc-Wen nonlinear hysteretic loops against known OpenSeesPy benchmarks, so I had confidence in the ground-truth generator before scaling it up to multi-degree-of-freedom buildings.
- Tracking continuous hysteretic energy dissipation ($E_h(t) = \int f_\mathrm{int}\,du$) as a physically meaningful quantity to check against, not just displacement error.
- Curating earthquake records from the PEER NGA-West2 strong-motion database and, early on, deciding that random train/test splits weren't good enough for this problem — a model can look like it generalises while actually just interpolating within a structural family it's already seen. I switched to splitting by structural archetype and by earthquake identity instead, so "unseen" means something physical.

This groundwork mattered more than I expected. Almost every surprising result later in the project traced back to something about the representation or the physics — which is a useful thing to notice about this class of problem: most of the interesting scientific content lives in how you set the problem up, not in how many layers you add.

---

## 2. Problem formulation

A multi-storey shear building subjected to horizontal ground acceleration $a_g(t)$ is governed by the coupled equation of motion:

$$M\ddot{u}(t) + C\dot{u}(t) + f_\mathrm{int}(u(t),\dot{u}(t)) = -M\iota\, a_g(t)$$

where $M, C \in \mathbb{R}^{N \times N}$ are the mass and Rayleigh damping matrices; $f_\mathrm{int}$ is the (possibly nonlinear) restoring-force operator; $u(t) \in \mathbb{R}^N$ is the relative floor-displacement vector; $\iota = [1,\ldots,1]^T$.

The surrogate modelling goal is to approximate the operator mapping ground acceleration and structural parameters to the full response trajectory:

$$\mathcal{G}: \mathcal{A} \times \mathcal{S} \to \mathcal{U}, \quad u(t) = \{\bar{u}_i(t)\}_{i=1}^N$$

$\mathcal{A}$ is the space of continuous ground-acceleration functions; $\mathcal{S}$ is the structural parameter space (masses, stiffnesses, storey count, damping). This mapping should ideally be resolution-invariant — evaluating at arbitrary discrete time points shouldn't require retraining when the time discretisation changes — which is part of why the operator-learning framing (rather than a fixed-length sequence model) made sense for this problem.

---

## 3. Buildings, ground motions, and the partitioning scheme

Buildings were idealised as lumped-mass shear frames: storey mass 30,000 kg, inter-storey stiffness 15–60 MN/m, storey height 3.5 m, 5% Rayleigh damping tuned to the first two modal frequencies, and a bilinear elastoplastic restoring-force law with 5% kinematic hardening. Six archetypes span fundamental periods from 0.35 s to 1.20 s:

| Archetype | Storeys | T₁ (s) | Role |
|---|---|---|---|
| 3S_T035 | 3 | 0.35 | Training / ID |
| 3S_T060 | 3 | 0.60 | Training / ID |
| 3S_T090 | 3 | 0.90 | Training / ID — upper training T₁ for 3-storey |
| 5S_T055 | 5 | 0.55 | Training / ID |
| 5S_T085 | 5 | 0.85 | Training / ID — upper training T₁ for 5-storey |
| **5S_T120** | **5** | **1.20** | **Held-out OOD-B — unseen structure, the main test** |

Ground motions were drawn from the PEER NGA-West2 database, covering a mix of historical shallow-crustal events, with records split into disjoint training and held-out sets by event identity. Rather than a random split, partitioning ran along two independent axes — structural archetype and earthquake identity — giving four evaluation partitions with genuinely different physical meaning:

| Partition | Structural group | Earthquake group | Runs | What it tests |
|---|---|---|---|---|
| In-distribution (ID) | known archetypes, T₁ ≤ 0.90 s | training records | 120 | ordinary accuracy |
| OOD-A | known archetypes, T₁ ≤ 0.90 s | held-out records | 300 | unseen ground motion |
| OOD-B | unseen structural period — the main test | training records | 240 | unseen structural period |
| OOD-C | held-out archetype (T₁=1.20 s) | held-out records | 60 | combined structural + seismic extrapolation |
| **Total** | | | **720** | |

*720 structure-earthquake combinations, checked programmatically for zero overlap in structure or earthquake identity across partitions before any model was trained.*

---

## 4. First attempt: a fixed-grid Fourier Neural Operator, and why it failed

The natural starting point was a standard 2D Fourier Neural Operator (Li et al., 2021), which has worked well on regular Cartesian domains such as fluid-flow fields. I built a fixed 5×2048 input tensor — five storeys (the tallest archetype) by 2,048 time steps — and zero-padded shorter buildings to fill the unused storeys. Five-storey buildings, which naturally filled the grid, reached a reasonable 19.29% median relative L₂ error. Three-storey buildings, padded across two zeros, reached a median error of **99.6%** — essentially garbage.

The reason is mathematically straightforward once you look at it in the frequency domain. A discrete spatial Fourier transform implicitly assumes periodic boundary conditions across the lattice. Padding a three-storey building with two zero floors creates a sharp step between the last real floor and the padded region, and the spectral convolution reads that step as "nothing there" but as a high-frequency ringing leads corrupted spectral energy back into the real floors below. This wasn't a training problem; the longer training and more layers didn't help, because the representation itself imposed an invalid assumption: a building is a small, irregular graph of masses connected by columns, not a padded image.

**Moving from a fixed grid to a graph representation resolves the zero-padding failure:**

| Representation | 3-storey median relative L₂ |
|---|---|
| Fixed-grid FNO2D (EXP4) | **99.6%** |
| Topology-native GNO (EXP5) | **22.1%** |

---

## 5. A topology-native fix: graph neural operators

The fix was to stop forcing buildings onto a shared grid. Each building now owns its own graph $\mathcal{G} = (V, E)$: one node per real storey ($|V| = N_\mathrm{stories}$), edges between physically adjacent floors carrying the column stiffness. Each graph block computes message passing and temporal spectral convolution:

$$m_v^{(l+1)}(t) = \phi\bigl(h_v^{(l)}(t),\,\sum_{u\in\mathcal{N}(v)} \psi(h_u^{(l)}(t), e_{uv})\bigr)$$

followed by a 1D temporal Fourier spectral layer applied node-wise to handle the dynamics:

$$h_v^{(l+1)}(t) = \phi\bigl(W h_v^{(l+1)}(t) + \mathcal{F}^{-1}[R_l(k)\cdot\mathcal{F}(h_v^{(l+1)})(k)](t)\bigr)$$

This structure eliminates zero-padding artifacts while coupling spatial inter-storey information exchange with temporal wave propagation on a shared latent representation.

---

## 6. What graph topology alone can't fix

That was satisfying, but it uncovered a second, subtler problem. Evaluating the same unconditioned graph model on the held-out flexible archetype (T₁ = 1.20 s, outside the training range which extends to T₁ = 0.90 s) gave a peak displacement error of around 35.2%, and full-trajectory relative L₂ error above 100%. Looking at the FFT of the predicted displacement records showed the model was oscillating at approximately 1.12 Hz — near the edge of what it was trained on — rather than tracking the true resonance at 0.83 Hz.

That was satisfying and unsettling simultaneously: modal conditioning explains *why* the model oscillates at the wrong frequency, but the L₂ error result is a lot worse than peak error suggests. When two sinusoidal signals of identical amplitude but slightly different frequencies are compared pointwise, they drift into phase opposition. Once phase opposition is reached and true waveforms drift into opposition, the L₂ difference is mathematically ~141%. Modal conditioning does not warp the temporal frequency basis functions — that's still fixed and frozen per the whole record. A model that only sees the initial stiffness can't track the resonance of a structure as its stiffness changes through inelastic yielding.

---

## 7. Conditioning on what we already know: modal invariants via FiLM

Given a building's mass and stiffness matrices, mode shapes and natural periods are computable before any shaking happens, by solving the generalised eigenvalue problem:

$$K\phi_i = \omega_i^2 M\phi_i, \quad T_i = 2\pi/\omega_i$$

Because this only uses the known, pre-earthquake $[M]$ and $[K]$, it leaks no information about the future ground motion or response trajectory, using Feature-wise Linear Modulation (Perez et al., 2018) to let it rescale the network's internal representations per layer:

$$[y, \beta] = \mathrm{MLP}(c), \quad h^{(l+1)} = (1 + \gamma) \odot h^{(l)} + \beta$$

I compared an unconditioned baseline against two conditioned variants — one using the fundamental period T₁ only, one using the first three periods and frequencies together — and a falsification control described below.

**Peak-displacement error across all four partitions:**

| Model | ID Peak Disp | OOD-A (unseen EQ) | OOD-B (unseen structure) | OOD-C (both unseen) |
|---|---|---|---|---|
| EXP5 (unconditioned) | 12.70% | 15.86% | 35.21% | 38.66% |
| EXP6-B (T₁ only) | 2.09% | 8.81% | 13.47% | 14.36% |
| EXP6-C (multi-modal) | 8.87% | **7.63%** | **13.06%** | 17.01% |
| EXP6-D (shuffled T₁) | 2.65% | 8.94% | 24.33% | 36.16% |

*All values are median peak-displacement error (%). Conditioning helps in every partition, but the gap is largest exactly where it matters most — the unseen-structure partitions.*

---

### Checking this wasn't just extra capacity

Before trusting the improvement, I wanted to rule out the boring explanation: maybe the FiLM branch just gave the network more parameters to work with, and any auxiliary vector — physically meaningful or not — would have helped similarly. I re-ran evaluation with the conditioning vectors shuffled across buildings within a batch, so each sample received a *different* building's modal descriptor. If the network were exploiting genuine physical correspondence, shuffling should hurt performance; if it were just using extra capacity, it shouldn't matter much.

Performance under shuffling degraded to 24.33% on OOD-B and 36.16% on OOD-C — worse than either conditioned variant, though still somewhat better than the unconditioned baseline. That pattern is consistent with the model relying on physical correspondence rather than scalar capacity, though I'd stop short of calling it proof of causality — it's evidence, not certainty, and I'd want a dimensionally matched but physically meaningless control vector rather than a shuffled real one to state the Section 5 result more rigorously.

---

## 8. Computational performance

Timings were measured on Apple Silicon (MPS) with process-wide device locks to avoid benchmark noise from concurrent Metal command buffers:

| Model / setting | Latency | Throughput | Speedup vs. OpenSeesPy |
|---|---|---|---|
| OpenSeesPy reference, batch 1 | 54.68 ms | 18.3 sim/s | 1.0× |
| Fixed-grid FNO2D (EXP4, batch 1) | 6.60 ms | 151.5 sim/s | 8.3× |
| Topology GNO (EXP5, batch 1) | 16.25 ms | 61.5 sim/s | 3.4× |
| T₁-conditioned GNO (EXP6-B, batch 1) | 21.45 ms | 46.6 sim/s | 2.6× |
| T₁-conditioned GNO (EXP6-B, batch 32) | 30.19 ms | 1,060 sim/s | **~58× (throughput vs. single OpenSeesPy)** |

The fixed-grid model is fastest in isolation, which isn't surprising — it's also the model that doesn't work. Among the models that actually generalise, the single-sample speedup (around 2.5–3.4×) is useful for an interactive design tool, not transformative. The batched throughput becomes practically interesting: over a thousand building evaluations per second on regional screening tasks, assuming the accuracy caveats above are acceptable for that use case (peak-demand estimation, not exact hysteretic response reconstruction).

---

## 9. Architecture summary

```
Structure [M, K]
     │
     ▼
Eigen-solve ──────────────► Modal invariants c = [T₁..₃, ω₁..₃]
(pre-earthquake)                        │
     │                                  ▼
     ▼                           FiLM-conditioned
Graph build                       GNO blocks (×4)
G=(V,E), no padding  ──────────►  ─────────────────► Predicted
                                  u(t), IDR(t), V(t)
     │
     ▼
OpenSeesPy
(NLTHA reference)
```

End-to-end pipeline: structural properties feed both the eigen-solver and the graph construction; the resulting modal invariants condition every FiLM block; GNO predictions are checked against OpenSeesPy ground truth.

---

## 10. Where I'd take this next

- Replace the static 1D Fourier temporal layers with causal state-space models or neural ODEs to address the phase-drift limitation. This is the most pressing technical gap.
- Add a confidence-output head that separately penalises amplitude and phase errors, instead of a single L₂ objective that conflates the two.
- Extend from 2D planar to 3D asymmetric structures with torsional response and bidirectional ground motion, which is where most real buildings actually live outside this idealisation.
- Track stiffness degradation online as plastic hinges form, rather than conditioning only on the undamaged pre-earthquake modal state.
- Run a cleaner falsification ablation — conditioning on a dimensionally matched but physically meaningless control vector, rather than a shuffled real one — to state the Section 7 result more rigorously.

---

## 11. Reproducibility

Data partitions were checked programmatically for zero overlap in structure or earthquake identity across the four partitions before any model saw them. EXP4, EXP5, and EXP6 variants are all frozen and hash-verified against the evaluation logs referenced above, and the repository's unit test suite — covering the numerical solvers, split disjointness, and model I/O shapes — passes in full at the time of writing. The codebase, evaluation scripts, and an interactive demo comparing live model output against OpenSeesPy ground truth are available on request.

---

## References

Li, Z., Kovachki, N., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *International Conference on Learning Representations (ICLR).*

Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI Conference on Artificial Intelligence.*

Pacific Earthquake Engineering Research Center (PEER). NGA-West2 Ground Motion Database.
