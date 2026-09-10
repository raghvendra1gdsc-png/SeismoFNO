# Why this, and what I hope to contribute

*Raghvendra Singh Gahlot · B.Tech (Civil Engineering) · September 2026*

---

## Why this

I came to this project sideways. I was running nonlinear time-history analyses in OpenSeesPy for a coursework module on seismic design, and I found myself distracted by a simple question: how long each run took. A single five-story shear frame with moderate ground motion was taking around 50 milliseconds. That feels negligible until you think about what regional risk assessment actually requires—tens of thousands of buildings across a city, dozens of earthquake scenarios, multiple intensity levels. The arithmetic makes the problem clear quickly.

I knew neural networks were being applied to surrogate modelling in fluid mechanics and climate science under the label "Scientific ML," and I wondered whether the same idea could work for structural dynamics. What I didn't anticipate was how much the project would teach me about *why* specific modelling choices fail before they teach me how to fix them.

The three-story failure in EXP4—where a Fourier Neural Operator achieved 99.6% error on buildings it had technically been trained on—was the most instructive moment in the project. I had done something that looked reasonable: pad variable-height buildings to a common grid size. It took a systematic error analysis to trace back to why that was wrong (the Fourier transform assumes periodic boundaries; a hard step to zero is the worst possible violation of that assumption). Discovering and diagnosing that failure felt more valuable than if the model had just worked from the start.

I want to keep doing this kind of work—chasing the physical meaning of model failures, and trying to build representations that respect what the physics actually requires.

---

## What I aim to do

SeismoFNO is a winter project that took longer than I expected, mostly because I kept finding things I couldn't explain yet. The modal phase-drift problem—where the model gets the peak displacement right but gets the frequency wrong—I still haven't solved. I know the mechanistic reason (global Fourier layers can't adapt their frequency basis dynamically), and I know the likely fix (state-space models or neural ODEs that process time causally), but building and evaluating that properly would take more time and expertise than I have right now.

That's precisely the situation where working with researchers who actually specialise in this matters. I want to spend a focused period—ideally a full research internship or summer project—working on problems like this under supervision. Not to pad a CV, but because the combination of structural dynamics and scientific machine learning is genuinely where I want to work, and I'm at the point where I've learned as much as I can in isolation.

---

## What I wish to learn

The parts of this project I did worst were the ablation design and uncertainty quantification. The shuffled-conditioning ablation in EXP6 was the closest I came to a proper falsification check, but it's blunt—shuffling within a batch doesn't test all the ways a model could be exploiting spurious correlations. I want to learn how to design ablations that are actually tight, and how to quantify confidence bounds on empirical results rather than just reporting point estimates.

More broadly, I want to work on problems where a result being wrong matters—where the output is used to make decisions about buildings or infrastructure, not just to beat a benchmark leaderboard.

---

## What I bring

A working end-to-end pipeline I built myself: OpenSeesPy ground-truth generation, graph neural operator architecture, physics-informed conditioning, a zero-leakage data split protocol, 305 passing unit tests, and frozen checkpoints with hash-verified evaluation CSVs. The ability to read structural dynamics literature and translate it into code without needing every step explained. A baseline of intellectual honesty about what doesn't work—the phase drift issue is documented as prominently in the codebase as the results that look good.

I'm early in my training, so I'm not pretending to bring theoretical depth I don't have. What I can bring is a project that's real, a genuine interest in why it breaks where it does, and enough independence to be useful quickly rather than needing to be walked through the basics.

---

*Repository: SeismoFNO (available on GitHub). Full forensic audit, evaluation CSVs, and interactive demo are available for inspection.*
