# Related Work Notes — SeismoFNO
*Phase 0 literature synthesis. All claims below are sourced from the findings
provided; no additional claims have been invented or inferred.*

---

## 1  FNO Accuracy Degrades Sharply with Nonlinearity

A 2026 systematic evaluation of FNOs on single-degree-of-freedom (SDOF) and
five-degree-of-freedom (5DOF) systems measured **mean relative L2 error
within the training regime** across three constitutive models:

| Constitutive law | Mean relative L2 error (in-distribution) |
|---|---|
| Linear-elastic | ~0.02 (2 %) |
| Cubic-hardening | ~0.28 (28 %) |
| Bilinear-hysteretic | ~0.38 (38 %) |

Even within the distribution the surrogate was trained on, hysteretic
behaviour is nearly 20x harder to capture than linear response.

> **IMPORTANT**: These numbers are from an in-distribution evaluation, not a
> zero-shot generalisation test. The error on held-out earthquakes or held-out
> structures is expected to be higher. Any generalisation claim in this project
> must use the held-out splits defined in `src/data_pipeline/splits.py`, not
> these in-distribution figures.

---

## 2  Why Bilinear Hysteresis is Hard: Path Dependence

Bilinear hysteresis introduces **path dependence**: after a structural
element yields, the same displacement value can correspond to different
restoring forces depending on whether the element is currently loading,
unloading, or reloading.

Concretely, if the FNO receives only the ground-acceleration time series as
input, it must implicitly infer the internal state (e.g. whether the
structure has previously yielded and at what maximum displacement) from the
history of the input signal. A vanilla FNO with a finite receptive field
may not integrate enough history to distinguish these cases — this is the
likely mechanistic cause of the large in-distribution error in Section 1.

**Implication for this project:** The surrogate model must have some mechanism
for accumulating and representing history-dependent internal state. This is a
design constraint, not just a performance concern.

---

## 3  Discretisation Invariance Breaks Down in Practice

Discretisation invariance (evaluating at a resolution different from the
training resolution) is a theoretical property of FNOs via their spectral
parameterisation. In practice, FNOs trained at coarse resolution and
evaluated at a finer resolution show degraded accuracy — the invariance
often fails.

**Implication:** Resolution must be treated as a hyperparameter matched
between training data and inference data unless explicit multi-resolution
training is used. Any speedup benchmark (Phase 9) must compare OpenSeesPy
and the FNO at the *same* time-step resolution.

---

## 4  Closing the Hysteresis Gap: Explicit State > Loss Engineering

Recent architectural work — including Attention-augmented FNOs (AttFNO),
DeepONet-FNO hybrids, and physics-model + neural-operator compositions —
converges on a common design principle: give the model **explicit access to
history or internal state**, rather than relying on additional loss terms
alone.

Physics-informed loss terms help regularise training but do not substitute
for an architecture that can track state. Phase 5 (hysteretic FNO core)
should therefore prioritise explicit history-representation mechanisms over
physics-loss tuning.

---

## Open Scientific Concerns (Phase 0)

Flagged now per AGENTS.md Rule 6 to avoid silent propagation into later phases:

1. **History-length sensitivity:** If the FNO input window is shorter than
   the duration of strong shaking, the model cannot observe all yielding
   events. The maximum window length must be documented relative to the
   longest ground-motion record in the database.

2. **Energy error accumulation:** Relative L2 on displacement may hide large
   energy errors, since energy is a quadratic functional of displacement and
   velocity. Phase 5 should report both.

3. **Apples-to-apples benchmark caution:** Any comparison of FNO inference
   speed vs. OpenSeesPy must control for hardware (CPU vs GPU), numerical
   integrator step size, and number of DOFs. An unreported speedup is not a
   valid speedup claim (AGENTS.md Rule 3).
