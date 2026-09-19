# AGENTS.md — SeismoFNO

## Mission
Build a Fourier Neural Operator surrogate for nonlinear SDOF/MDOF seismic
response and hysteretic energy dissipation, validated against OpenSeesPy
ground truth. This is a research project for an undergraduate research
application — scientific honesty matters more than a clean-looking result.

## Hard rules (do not violate these even if it seems slower)
1. Follow the phase order below. Do not write any code in src/models/ or
   src/losses/ until every ground-truth validation test in tests/ for the
   current system (SDOF or MDOF) passes.
2. Never evaluate "zero-shot generalization" using a random train/test
   split. Only use the held-out-earthquake and held-out-structure splits
   defined in src/data_pipeline/splits.py for any generalization claim.
3. Every number that ends up in results/ or a report (error %, speedup,
   R², etc.) must be traceable to a specific script + config + run log.
   Never write down a number "estimated" or "typical" — only measured.
4. When a result looks worse than the original pitch (e.g. hysteretic
   error is much higher than linear-elastic error), do NOT quietly tune
   until it looks better without recording what was tried. Log every
   experiment in experiments/<date>_<name>/, including failed ones.
5. Before implementing anything from scratch, check if it already exists
   in src/. Do not duplicate a spectral conv layer, a loss function, etc.
6. Flag scientific concerns explicitly rather than proceeding silently —
   e.g. if a data split leaks information, if a benchmark isn't apples-
   to-apples, if a physics loss term isn't actually being minimized.

## Phase order
0. Literature notes -> 1. SDOF ground truth (validated) -> 2. Ground
motion database -> 3. SDOF dataset + splits -> 4. FNO MVP (linear) ->
5. FNO core (hysteretic + physics loss, ablated) -> 6. Baselines
(LSTM/MLP) -> 7. Zero-shot tests -> 8. MDOF extension -> 9. Speed
benchmark -> 10. Error analysis + report -> 11. Dashboard (optional,
last).

## Current phase
Current phase: Hackathon Integration Complete (NVIDIA Nemotron ReAct agent and 8 deterministic engineering tools unified into main FastAPI application gateway and exposed via Judge Mode /judge workstation; live NVIDIA API integration verified; all 288 unit and integration tests passing)