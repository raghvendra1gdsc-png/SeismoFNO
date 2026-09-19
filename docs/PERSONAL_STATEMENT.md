# Physics-Grounded Neural Operators for Structural Dynamics

*Raghvendra Singh Gahlot · B.Tech (Civil Engineering), IIT · September 2026*

---

## 1. Motivation: Bridging Structural Simulation and Operator Learning

I came to this project sideways. While running nonlinear time-history analyses in OpenSeesPy for a seismic design course, I became fascinated by the computational bottleneck: simulating a modest five-story shear frame under moderate shaking required ~50 milliseconds. While negligible for an isolated check, regional seismic resilience assessment demands evaluating tens of thousands of unique building archetypes across diverse earthquake scenarios. The resulting computational burden forces engineers to rely on simplified empirical curves that sacrifice structural fidelity.

Scientific machine learning has shown immense promise in fluids and climate modelling, and I wanted to test whether neural operators could offer a rigorous surrogate for structural dynamics. What I did not anticipate was how much the journey would teach me about why standard deep learning recipes fail when applied naively to structural physics.

---

## 2. Diagnostic Milestones: The Zero-Padding Discovery

The most instructive moment was the failure in EXP4. I had taken an intuitive computer vision approach: pad variable-height buildings to a shared 5-story Cartesian grid. While 5-story buildings performed adequately (19.29% median error), 3-story buildings completely broke down (**99.60% median error**). Tracing this failure to first principles revealed that the 2D Fourier transform assumes periodic boundaries; a hard step to zero creates artificial spectral discontinuities that global kernels reflect as Gibbs ringing into the physical floors. Moving to a topology-native graph in EXP5 resolved this, dropping error to **22.09%**. Diagnosing that representational failure taught me far more than if the initial baseline had simply converged.

---

## 3. Current Limitations: The Phase-Drift Problem

SeismoFNO uncovered a deeper physical challenge: modal phase drift. Conditioning on undamped modal invariants ($T_1, \omega_1$) reduced out-of-distribution peak displacement error from 35.21% to **13.47%**. However, full-trajectory relative $L_2$ error remained high (>100%). Frequency analysis confirmed why: static global Fourier layers apply fixed complex weights across the entire time series and cannot dynamically track period elongation as columns yield plastically. Solving this requires causal architectures—such as state-space models (S4, Mamba) or neural ODEs—that process structural dynamics sequentially.

---

## 4. Methodological Growth: Rigorous Falsification and UQ

This project forced me to build habits of scientific honesty. Rather than masking out-of-distribution errors, I designed a shuffled-conditioning ablation (EXP6-D) to prove the model relied on true modal physics rather than excess network capacity. Moving forward, I want to learn how to formulate tighter falsification bounds, implement rigorous Bayesian uncertainty quantification, and work on problems where surrogate predictions directly safeguard physical infrastructure.

---

## 5. Technical Preparation and Independent Foundations

I bring an end-to-end working framework built from scratch: OpenSeesPy validation pipelines, spatiotemporal graph neural operators, physics-informed modal FiLM modulation, 305 passing unit tests, and hash-verified reproducible evaluation logs. While early in my academic career, I possess the technical grounding, scientific curiosity, and independence to contribute meaningfully to research at the intersection of structural mechanics and scientific machine learning.

---

*Repository: **SeismoFNO** (github.com/raghvendra1gdsc-png/SeismoFNO) · Full forensic audit, evaluation CSVs, and interactive demo available.*
