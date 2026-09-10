# SeismoFNO: Physics-Grounded Neural Operators for Seismic Structural Response

*Raghvendra Singh Gahlot · B.Tech (Civil Engineering) · September 2026*

---

## Why this problem

Simulating a multi-story building's response to an earthquake—tracking every floor's displacement over a 20-second shaking event—takes OpenSeesPy between 30 and 200 milliseconds per structure, depending on how much plastic yielding happens. That sounds fast, but regional seismic risk assessment involves tens of thousands of unique buildings across a city grid. Running high-fidelity non-linear time-history analyses for each one is prohibitive, which is why the field has historically relied on simplified empirical relations that discard most of the interesting physics.

Neural operators offer an alternative: learn the mapping from earthquake input to structural response once, then evaluate it near-instantly. The underlying governing equation is a coupled second-order matrix ODE:

$$M\ddot{u}(t) + C\dot{u}(t) + f_\mathrm{int}(u(t),\dot{u}(t)) = -M\iota\, a_g(t)$$

From an operator-learning standpoint this is a mapping between infinite-dimensional function spaces—ground acceleration $a_g \in L^2([0,T])$ and multi-story trajectory $u \in L^2([0,T])^N$—conditioned on a structural parameter set $\mathcal{S}$ that specifies floor masses, inter-story stiffnesses, and topology. The challenge is that $N$ (story count) and $\mathcal{S}$ vary across buildings, which standard FNO architectures were not designed to handle gracefully.

---

## 1. Foundations: getting the physics right first

Before training any surrogate model, I needed confidence in the ground-truth generator. EXP1 through EXP3 were entirely about that.

**EXP1** validated single-degree-of-freedom elastoplastic dynamics against analytical solutions, and separately validated Bouc-Wen non-linear hysteretic loops. The point was not just to confirm that OpenSeesPy produced reasonable numbers—it was to build a ground-truth pipeline I could trust well enough to stake everything downstream on it. If the reference data is wrong, every model trained against it is wrong in an unfalsifiable way.

**EXP2** extended this to continuous hysteretic energy dissipation tracking:

$$E_h(t) = \int_0^t f_\mathrm{int}(u,\dot{u})\,\dot{u}\,d\tau$$

This mattered because hysteretic energy is the quantity most directly related to structural damage. Getting it right meant the model would later have to predict not just displacement peaks but the full loop shape.

**EXP3** addressed the data-partitioning question—arguably the most methodologically consequential decision in the project. The naive approach (random train/test split on individual simulation records) leaks structural geometry and earthquake character across partitions. I switched to group-based splits: training structures and training earthquake records are disjoint from test structures and test earthquake records. This distinction between OOD-A (held-out earthquakes, known structures), OOD-B (held-out structure, known earthquakes), and OOD-C (both held out) is what makes any generalization claim meaningful.

---

## 2. First model attempt and its failure (EXP4)

With a validated ground-truth pipeline and clean data splits, I trained a standard 2D Fourier Neural Operator on a fixed $5 \times 1024$ spatial-temporal tensor. Three-story buildings were zero-padded to fill stories 4 and 5.

On five-story buildings this worked: median Relative $L_2$ error of 19.29%. On three-story buildings, it broke entirely: **99.60% median error**.

The cause took some digging to confirm. A 2D Fourier transform assumes periodic boundary conditions on both axes. Zero-padding stories 4 and 5 creates a hard step from physical values to zero—the sort of sharp discontinuity that Fourier representations are worst at. The global spectral kernels spread this boundary artifact back down into the physical floors, producing severe ringing. No amount of training could fix it because the representation itself was wrong for variable-height structures.

This wasn't a disappointment so much as a diagnostic: the model revealed exactly which assumption it was violating.

---

## 3. Graph representation and a new failure mode (EXP5)

The fix was to stop forcing buildings onto a shared grid. In EXP5, each building is its own graph: $\mathcal{G} = (V, E)$ where nodes are floor slabs and edges are inter-story columns. Three-story buildings have 3 nodes; five-story buildings have 5. No padding, no artificial discontinuity.

Each spatiotemporal block combines graph message-passing along structural edges:

$$m_v(t) = \sum_{u \in \mathcal{N}(v)} W_\mathrm{val}\, h_u(t) \odot \sigma(W_\mathrm{edge}\, e_{uv}) + W_\mathrm{self}\, h_v(t)$$

with a 1D temporal Fourier convolution applied node-wise across the 1,024 time steps. The two operations complement each other: graph message-passing handles the discrete spatial structural connectivity; Fourier convolution handles the continuous temporal wave propagation.

Three-story error dropped from 99.60% to **22.09%**—a 77-percentage-point improvement. That validated the topology hypothesis.

What I did not anticipate was a new failure mode on structurally out-of-distribution frames. The held-out flexible archetype `5S_T120` has a fundamental period of $T_1 = 1.20\,\mathrm{s}$; the highest-period training structure is the five-story `5S_T085` at $T_1 = 0.85\,\mathrm{s}$, giving a 0.35-second extrapolation gap. On this structure, EXP5 produced a **115.7% Relative $L_2$ error**, despite predicting peak displacement within **35.2%**.

Forensic FFT analysis of the predictions showed the model was oscillating at roughly 1.12 Hz ($T_1 \approx 0.89\,\mathrm{s}$)—near the training envelope upper limit—instead of the true 0.83 Hz. Two sinusoidal signals of identical amplitude but slightly different frequencies drift into phase opposition after a number of cycles proportional to their frequency difference. Over 20.48 seconds, a 0.29 Hz discrepancy is enough to reach complete anti-phase, giving a mathematical relative error near $\sqrt{2} \approx 141\%$. This is why the phase error was large while the amplitude error was manageable: the unconditioned model found the right energy content but the wrong frequency.

---

## 4. Modal conditioning via FiLM (EXP6)

The fix has a straightforward physical motivation. Before any earthquake happens, a structural engineer can compute the building's natural frequencies from the undamped eigenvalue problem:

$$K\phi_n = \omega_n^2 M\phi_n, \quad T_n = 2\pi/\omega_n$$

These quantities—$T_1, T_2, T_3, \omega_1, \omega_2, \omega_3$—are knowable pre-earthquake from the structural drawings. They contain no information about future ground motion or dynamic response, so injecting them into the model leaks nothing. They do, however, tell the model where the building's resonances are.

In EXP6, these modal invariants are fed through a small MLP that produces per-layer scale and shift vectors, following Feature-wise Linear Modulation:

$$[\gamma^l, \beta^l] = \mathrm{MLP}^l(c), \quad h_\mathrm{mod} = (1 + \gamma^l) \odot h + \beta^l$$

Modulation is applied to both the spatial graph message-passing and the temporal spectral convolution in each of the four operator blocks. The idea is that the building's natural frequency should inform how the temporal Fourier kernels weight different frequency bands.

Results on the held-out `5S_T120` archetype:

| Model | Peak disp. error (OOD-B) | Rel $L_2$ (OOD-B) | Pearson $r$ (OOD-B) |
|---|---|---|---|
| Unconditioned GNO (EXP5) | 35.21% | 115.70% | 0.048 |
| $T_1$-conditioned GNO | 13.47% | 157.64% | 0.089 |
| Multi-modal GNO ($T_{1\text{-}3}, \omega_{1\text{-}3}$) | **13.06%** | 124.07% | **0.091** |
| Shuffled $T_1$ (ablation) | 24.33% | 119.54% | 0.082 |

The 62.9% relative reduction in peak displacement error is the headline result. The ablation is the check: when the same conditioning architecture receives a randomly shuffled $T_1$ (drawn from a different building in the batch), OOD-B peak error rises to 24.33%—halfway back to the unconditioned baseline. This rules out the explanation that FiLM's MLP is simply providing extra parameter capacity.

---

## 5. What still doesn't work

The Relative $L_2$ error on OOD-B remains above 100% even with modal conditioning. The peak envelope is controlled; the waveform phase is not. The reason is structural: global 1D Fourier layers apply a static complex multiplication over the full 20.48-second record. They cannot warp the frequency basis dynamically as the building softens under yielding, so cumulative phase drift persists.

This isn't a flaw in how the model was trained—it's a fundamental limitation of any approach built on static global Fourier convolutions. The natural successor is a continuous-time state-space model (S4, Mamba) or a neural ODE integrator that processes time causally step-by-step, allowing frequency to adapt in response to instantaneous structural state.

---

## 6. Infrastructure and reproducibility

The project runs 305 unit tests covering the numerical solvers, split disjointness enforcement, model forward-pass dimensions, and metric computations. All evaluation numbers cited above are computed live from the raw CSV files under `results/experiments/exp6/evaluation/` and are verified by an automated forensic audit script. Latency benchmarks were measured on an Apple Silicon GPU (`torch.device("mps")`) with process-wide device synchronization to prevent concurrent Metal command interference.

Inference latency (single sample): **21.45 ms** for the $T_1$-conditioned GNO versus **54.68 ms** for the OpenSeesPy solver—a **2.55× speedup** on this hardware configuration.

The codebase, checkpoints, evaluation CSVs, and forensic audit report are available in the repository. An interactive web dashboard allows real-time model evaluation against OpenSeesPy ground truth.

---

*All numbers are traceable to experiment log files and evaluation CSVs. No results are estimated or interpolated.*
