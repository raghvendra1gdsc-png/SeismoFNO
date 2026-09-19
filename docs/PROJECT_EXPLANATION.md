# SeismoFNO — Multi-Audience Project Explanations

**Project Title:** SeismoFNO: Physics-Grounded Neural Operators for Seismic Structural Dynamics  
**Target Evaluation:** Graduate Research Review  
**Core Motto:** *Scientific honesty first — measure, explain, and bound every claim.*

---

## 1. 30-Second Version (Elevator Pitch)

> "SeismoFNO is a Scientific ML project investigating neural operators as fast surrogates for non-linear seismic structural simulation.
> 
> Standard Fourier Neural Operators fail on buildings with variable floor counts because zero-padding introduces artificial spatial boundary artifacts (incurring a 99.6% error on 3-story frames). We resolved this by building a topology-native Spatiotemporal Graph Neural Operator that models physical structural connectivity natively, reducing 3-story error to 22.1%.
> 
> When structures extrapolate beyond the training stiffness distribution, unconditioned models suffer large peak response errors. By conditioning the graph operator on structural eigenvalue invariants via FiLM, we reduced peak displacement error on unseen flexible structures from 35.2% down to 13.1%—a 62.9% relative reduction. However, because global Fourier layers still suffer cumulative phase drift over 20-second transient horizons, we transparently document the phase extrapolation limitation."

---

## 2. 2-Minute Version (Executive Technical Overview)

> "Seismic structural analysis simulates how buildings deform and dissipate energy during earthquakes by solving non-linear dynamic equations of motion in finite element solvers like OpenSeesPy. While rigorous, these numerical solutions are computationally intensive.
> 
> Neural operators offer an attractive surrogate framework because they learn mappings between infinite-dimensional function spaces. However, deploying them to structural dynamics presents two fundamental computer science challenges:
> 
> First, **variable topology**: Standard FNOs assume a regular Euclidean grid. When buildings of different heights are forced onto a fixed grid using zero-padding, global Fourier convolutions enforce periodicity across the artificial discontinuity, causing a 99.60% median error on 3-story buildings. In EXP5, we formulated a Spatiotemporal Graph Neural Operator (GNO) that maps physical stories to graph nodes and column connections to edges without any zero-padding, cutting 3-story error down to 22.09%.
> 
> Second, **out-of-distribution modal shift**: Civil structures encounter buildings with stiffnesses and natural vibration periods well outside the training set. In EXP6, we showed that unconditioned GNOs suffer 35.21% peak displacement error on unseen flexible structures ($T_1 = 1.20\text{ s}$). We developed a physics-conditioned GNO that extracts pre-earthquake modal eigenvalue invariants ($T_i, \omega_i$) and modulates the spatial graph convolutions and temporal spectral channels via Feature-wise Linear Modulation (FiLM). This dropped peak error to 13.06%—a 62.9% relative improvement.
> 
> Crucially, an ablation where conditioning vectors were randomly shuffled degraded performance back to 24.33%, falsifying the hypothesis that the gain was an artifact of extra network capacity. At the same time, we openly report that full waveform Relative $L_2$ error remains elevated (>100%) due to long-horizon phase drift in static Fourier bases. The code is backed by 284 passing tests and an independent forensic audit."

---

## 3. CSE / AI Professor Version (Theoretical & Algorithmic Rigor)

> "From a Computer Science and Machine Learning perspective, SeismoFNO investigates inductive biases, representation learning, and spectral operator approximation for coupled hyperbolic/parabolic partial and differential equations under severe distribution shift.
> 
> **Representation & Inductive Bias:**
> Standard neural operators (e.g., FNO, Li et al.) rely on the Fast Fourier Transform on uniform rectangular lattices. For civil structures with discrete stories and arbitrary connectivity, enforcing a fixed grid with zero-padding violates the smooth periodic boundary conditions assumed by the discrete Fourier transform, causing catastrophic high-frequency leakage. By treating the structure as an irregular spatial graph $\mathcal{G}=(V, E)$ coupled with a 1D continuous temporal operator along the time axis, we preserve permutation and topology invariance across variable-story structures while avoiding boundary interpolation artifacts.
> 
> **Conditioned Operator Approximation:**
> We formulate the forward mapping as an invariant-conditioned operator:
> $$\mathcal{G}_\theta: (A, \mathcal{G}, c) \mapsto U$$
> where $A \in L^2([0, T]; \mathbb{R})$ is the seismic excitation, $\mathcal{G}$ is the structural graph, and $c \in \mathbb{R}^{d_{\text{cond}}}$ represents pre-earthquake structural eigenvalue invariants ($K \phi_i = \omega_i^2 M \phi_i$). We inject $c$ into the operator blocks via dual-branch FiLM modulation. This enables the operator to dynamically rescale both spatial message-passing representations and complex temporal Fourier weights according to the system's natural frequencies.
> 
> **Rigorous Falsification & Honest Science:**
> To ensure the network was not merely benefiting from increased scalar model capacity, we implemented a randomized batch-shuffled ablation (Ablation D). Shuffling the modal vector significantly degrades out-of-distribution performance, confirming that the operator exploits the true physical correspondence between structural eigenvalues and dynamic response.
> 
> **Algorithmic Limitation:**
> We deliberately distinguish between envelope estimation and phase tracking. Global Fourier layers apply a static complex multiplication $W(k) \cdot \hat{h}(k)$. When the fundamental structural period shifts significantly out-of-distribution, slight frequency estimation errors integrate over 1,024 time steps ($20.48\text{ s}$), leading to cumulative phase drift. We document this mechanistic boundary as a clear argument for future work on hybrid continuous-time state-space models (e.g., S4/Mamba)."

---

## 4. Civil & Structural Engineering Professor Version (Mechanics & Validation)

> "In structural earthquake engineering, non-linear time-history analysis (NLTHA) is the gold standard for evaluating seismic performance under earthquake ground motions, solving:
> $$M \ddot{u}(t) + C \dot{u}(t) + f_{\text{int}}(u(t), \dot{u}(t)) = -M r a_g(t)$$
> While OpenSeesPy provides high-fidelity non-linear response with kinematic hardening, fiber sections, and Rayleigh damping, its computational expense hinders regional vulnerability assessment and rapid post-earthquake reconnaissance.
> 
> SeismoFNO explores neural operator surrogates with strict structural engineering validation:
> 
> 1. **Zero-Leakage Structural Splitting:**
>    Unlike common ML papers that randomly split time-history records (leaking building geometry and earthquake events across partitions), our benchmarks strictly isolate structural archetypes and seismic events. In OOD-A, ground motions RSN0011–12 are completely held out. In OOD-B, the flexible structural archetype `5S_T120` ($T_1 = 1.20\text{ s}$, 240 simulations) is entirely held out from training and validation.
> 
> 2. **Engineering Demand Parameter (EDP) Capture:**
>    In structural design, peak displacement ($u_{\max}$) and peak interstory drift ratio ($\text{IDR}_{\max}$) dictate life safety and structural damage limits. On unseen archetype `5S_T120`, our physics-conditioned GNO achieves a median peak displacement error of **13.06%**, compared to **35.21%** for an unconditioned GNO—a **62.9% relative error reduction**.
> 
> 3. **Modal Invariants as Prior Knowledge:**
>    We do not treat the building as a pure black box. Structural engineers always know the initial mass matrix $M$ and elastic stiffness matrix $K$. By computing eigenvalue natural periods ($T_1, T_2, T_3$) and modal frequencies ($\omega_1, \omega_2, \omega_3$) prior to dynamic analysis, we condition the neural operator on known elastic invariants without leaking any dynamic response data.
> 
> 4. **Limitations on Yielding & Phase:**
>    Because our modal descriptors are based on the initial elastic state, structures experiencing extreme plastic yielding experience dynamic period elongation. Combined with global Fourier phase drift, full waveform Relative $L_2$ errors remain high on extreme OOD cases. We present this clearly as an engineering boundary of spectral operators."

---

## 5. Interview Version (Interactive Pair-Programming & Defense)

> **Interviewer:** "Tell me about a challenging project you've worked on recently."
> 
> **Candidate:** 
> "I developed SeismoFNO, a Scientific Machine Learning project investigating neural operators for predicting non-linear structural building vibrations during earthquakes, validated against OpenSeesPy.
> 
> What made it challenging wasn't just training a model, but discovering and rigorously diagnosing unexpected failure modes across research phases:
> 
> In EXP4, we tested a 2D Fourier Neural Operator. It worked well on 5-story buildings, but completely broke on 3-story buildings—achieving 99.6% error. By conducting a systematic error analysis, I traced this to zero-padding: the 2D FNO assumed a fixed grid, and zero-padding floors 4 and 5 created severe spatial boundary discontinuities that global Fourier kernels couldn't resolve.
> 
> To fix this, in EXP5 I replaced the grid with a topology-native Spatiotemporal Graph Neural Operator where each floor is a graph node and columns are edges. This cut 3-story error from 99.6% down to 22.1%.
> 
> But when we stressed the GNO on out-of-distribution structures—like a flexible 5-story frame with a fundamental period of 1.2 seconds versus 0.85 seconds in training—it hit a 35.2% peak displacement error. In EXP6, I introduced physics conditioning via FiLM layers using pre-earthquake modal eigenvalue invariants ($T_1, \omega_1$). That dropped peak error down to 13.1%—a 62.9% relative reduction.
> 
> To prove the improvement wasn't just from adding parameters, I implemented a shuffled-conditioning ablation, which caused error to jump back up to 24.3%.
> 
> What I'm most proud of is that the project is scientifically honest: rather than claiming '100% accuracy', we clearly document that while peak displacement is well captured, long-horizon waveform phase still drifts because global Fourier layers can't dynamically adapt to out-of-distribution frequencies over 20-second records. The entire repository is covered by 284 automated tests and an independent forensic audit."
