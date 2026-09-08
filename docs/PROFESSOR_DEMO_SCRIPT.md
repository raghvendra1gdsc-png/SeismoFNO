# SEISMOFNO: PROFESSOR DEMONSTRATION SCRIPT
## 3–5 Minute Structured Oral Defense & Walkthrough

---

### [00:00 – 00:30] SECTION 1: THE RESEARCH QUESTION & MOTIVATION
*(Action: Stand in front of the SeismoFNO Research Header, pointing to the research status indicator: `RESEARCH CORE: FROZEN / AUDITED`)*

> "Professor, thank you for your time today.
> 
> The core research question in SeismoFNO is whether continuous neural operators can learn to predict non-linear structural dynamic responses when a building experiences **both discrete topological variations and distribution shifts in its structural modal properties**.
> 
> In conventional structural engineering, assessing a multi-story building under earthquake excitation requires solving coupled non-linear differential equations via numerical integration—such as Newmark-$\beta$ in OpenSeesPy. While rigorous, this takes tens to hundreds of milliseconds per ground motion and does not scale across regional building portfolios.
> 
> Our objective was not simply to fit a deep neural network to time series. Our objective was to uncover representation failures in standard operator learning architectures and introduce physics-grounded modal conditioning to overcome them."

---

### [00:30 – 01:15] SECTION 2: EXP4 & THE TOPOLOGY REPRESENTATION FAILURE
*(Action: Click on the 3-Story structure `3S_T035`, select `EXP4: Fixed-Grid FNO2D`, and point to the floor displacement plot)*

> "We began in EXP4 by testing standard 2D Fourier Neural Operators.
> 
> Standard FNOs assume a regular Cartesian grid. To feed multi-story buildings into this fixed grid, variable-floor structures had to be accommodated by zero-padding higher floors—for example, padding a 3-story frame with two zero-mass, zero-stiffness levels.
> 
> As you can see on this screen, the result was a catastrophic representation failure: the 3-story structure exhibited a Relative $L_2$ error of **99.60%**.
> 
> Why did this happen? The spatial Discrete Fourier Transform enforces periodic boundary conditions. Clamping the upper levels to zero introduced an artificial, non-physical step discontinuity. This induced severe high-frequency Gibbs ringing that completely corrupted the physical response of the lower floors.
> 
> This proved conclusively that Euclidean grid-based neural operators cannot represent variable structural topologies."

---

### [01:15 – 02:15] SECTION 3: EXP5 & TOPOLOGY-NATIVE GRAPH REPRESENTATION
*(Action: Switch the model selector to `EXP5: Graph Neural Operator`, and point to the Graph Representation card)*

> "To eliminate this boundary artifact, in EXP5 we reformulated the problem using a **topology-native Graph Neural Operator**.
> 
> As visualized here in the graph panel, every structural floor degree of freedom is mapped to a physical graph node, and columns and inter-story connections are mapped to physical edges. For a 3-story building, there are exactly 3 nodes. For a 5-story building, exactly 5 nodes. There is zero padding and no fictitious boundary.
> 
> The spatial dimension is processed via topology-native message-passing layers, while the temporal dimension is resolved via 1D Fourier spectral convolutions.
> 
> Notice what happened to the error: the 3-story Relative $L_2$ error dropped immediately from **99.60% down to 22.09%**—a 77.5 percentage-point error reduction.
> 
> This verified that variable structural topology is naturally resolved through graph-structured operator learning."

---

### [02:15 – 03:15] SECTION 4: EXP6 & PHYSICS-INFORMED MODAL CONDITIONING
*(Action: Select the held-out OOD-B structure `5S_T120`, select `EXP6-C: Multi-Modal GNO`, and click on the vertical mode-shape visualizer)*

> "However, EXP5 uncovered a deeper scientific obstacle: **modal distribution shift**.
> 
> When we evaluated the baseline GNO on an unseen, highly flexible 5-story structure—`5S_T120` with a fundamental period of $T_1 = 1.20$ seconds, well outside the training envelope of $T_1 \le 0.85$ seconds—the baseline peak displacement error spiked to **35.21%**.
> 
> To address this in EXP6, we introduced **pre-earthquake modal conditioning**.
> 
> Before any ground motion begins, we extract undamped eigenvalue invariants—natural periods $T_{1-3}$ and circular frequencies $\omega_{1-3}$—purely from the structural mass $[M]$ and initial stiffness $[K]$ matrices. These represent intrinsic physical descriptors of the system.
> 
> We inject these modal invariants into both the spatial graph layers and temporal spectral kernels using feature-wise linear modulation (FiLM).
> 
> As you see on the live displacement chart and the Out-of-Distribution matrix below, the median peak displacement error on the unseen flexible structure plummets from **35.21% down to 13.06%**—a **62.9% relative error reduction** across 2,160 physical simulations.
> 
> Furthermore, look at our falsification control: when we randomly shuffled the $T_1$ conditioning vector across batch instances during training, the error degraded back to **24.33%**. This demonstrates that the network is truly utilizing the physical structural correspondence, rather than merely exploiting auxiliary MLP capacity."

---

### [03:15 – 03:45] SECTION 5: HONEST FAILURE ANALYSIS & WAVEFORM PHASE DRIFT
*(Action: Scroll to the 'Where the Model Fails' section and highlight the high-horizon trajectory metrics)*

> "Crucially, we do not claim that neural operators solve all aspects of structural dynamics. In this demo, we explicitly document where the model fails.
> 
> Look at the trajectory metrics under extreme modal extrapolation: while peak displacement amplitude is accurately bounded within 13.06%, the global trajectory Relative $L_2$ error remains elevated over 100%, and Pearson correlation drops to approximately $0.05\text{--}0.09$.
> 
> This is caused by **waveform phase drift**. Global 1D Fourier layers apply static spectral filters across the entire 20.48-second simulation horizon. When the vibration period extrapolates far beyond training support, microscopic period discrepancies accumulate over 1,024 timesteps. Over a long horizon, this leads to complete waveform phase opposition.
> 
> Identifying this limitation is one of our primary contributions: it shows that global Fourier operators are excellent envelope scalers, but time-frequency wavelets or state-space recurrence are necessary for long-horizon phase coherence."

---

### [03:45 – 04:15] SECTION 6: COMPUTATIONAL SPEEDUP & CONCLUSION
*(Action: Point to the Computational Advantage panel and the reproducible audit telemetry)*

> "Finally, regarding computational speed: on single-building inference benchmarks running on Apple Silicon unified memory, the neural operator evaluates in **21.45 milliseconds**, compared to **54.68 milliseconds** for OpenSeesPy NLTHA—a **2.55× wall-clock speedup** for a single simulation, which scales by an order of magnitude under batch processing.
> 
> In summary, this research delivers three concrete findings:
> 1. Variable structural topologies demand graph-native representations rather than Euclidean zero-padded grids.
> 2. Pre-earthquake modal eigenvalue conditioning provides a principled mechanism to bound non-linear peak displacements under distribution shift.
> 3. Global Fourier layers suffer from cumulative phase drift under extreme modal extrapolation, defining a clear frontier for future state-space neural operators.
> 
> Every number in this interface is traceable to frozen checkpoints and audited datasets, with all 304 unit tests passing."

---

### [04:15 – 04:45] SECTION 7: FROM RESEARCH PROTOTYPE TO REAL-WORLD ENGINEERING WORKFLOW
*(Action: Click on the `00 LIVE EARTHQUAKE` workspace in the navigation sidebar, selecting an observed USGS earthquake event)*

> "To demonstrate how this prototype connects to physical engineering practice, we incorporated an observed real-world earthquake event layer using public USGS GeoJSON services.
> 
> However, we maintain strict scientific boundaries:
> 
> Observed earthquake → event metadata → compatible ground motion → structural model → SeismoFNO → rapid response estimate → engineering screening.
> 
> The system explicitly notifies the engineer that event metadata does not provide a direct acceleration waveform. The model evaluates rapid surrogate responses using verified ground motions under the structural scenario, screening high-risk buildings for targeted non-linear time-history analysis.
> 
> Future work would require validated waveform acquisition, site-specific ground-motion characterization, structural sensing, uncertainty quantification, and extensive external validation before operational engineering deployment.
> 
> Thank you, and I welcome your questions."
