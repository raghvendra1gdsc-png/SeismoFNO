# Digital Signal Processing Theory & Mathematical Validation

This document details the mathematical formulations, normalization criteria, and spectral estimation theory implemented in the **Multi-Channel Experimental Signal & Spectral Analyzer**.

---

## 1. Discrete Fourier Transform & Single-Sided Scaling

Given a discrete, uniformly sampled real-valued signal $x[n] = x(n \cdot \Delta t)$ for $n = 0, 1, \dots, N-1$ with sampling frequency $F_s = \frac{1}{\Delta t}$, the Discrete Fourier Transform (DFT) is defined as:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N_{\text{fft}}} k n}, \quad k = 0, 1, \dots, N_{\text{fft}}-1$$

Because $x[n] \in \mathbb{R}$, its discrete spectrum exhibits Hermitian symmetry:
$$X[N_{\text{fft}} - k] = X^*[k]$$

### 1.1 Single-Sided Amplitude Normalization
For physical engineering interpretation, negative frequencies are folded into the positive half-spectrum $[0, F_s/2]$. The single-sided amplitude spectrum $A[k]$ is defined over bins $k = 0, 1, \dots, \lfloor N_{\text{fft}}/2 \rfloor$:

$$
A[k] = \begin{cases}
\frac{1}{N \cdot CG} |X[k]|, & k = 0 \text{ (DC component)} \\
\frac{2}{N \cdot CG} |X[k]|, & 0 < k < \frac{N_{\text{fft}}}{2} \text{ (AC components)} \\
\frac{1}{N \cdot CG} |X[k]|, & k = \frac{N_{\text{fft}}}{2} \text{ (Nyquist, if } N_{\text{fft}} \text{ is even)}
\end{cases}
$$

where $CG$ is the **Coherent Gain** of the applied time window (see Section 2). This ensures that a continuous sinusoid $x(t) = A_0 \cos(2\pi f_0 t + \phi)$ produces a peak of magnitude exactly equal to $A_0$ in engineering units.

### 1.2 Logarithmic Decibel (dB) Representation
To identify low-energy higher modes simultaneously with high-amplitude fundamental modes, decibel scaling is computed as:

$$A_{\text{dB}}[k] = 20 \log_{10} \left( \frac{\max(A[k], \epsilon)}{A_{\text{ref}}} \right)$$

where $A_{\text{ref}} = 1.0$ (default engineering unit reference) and $\epsilon = 10^{-12}$ prevents numerical underflow.

### 1.3 Power Spectral Density (PSD)
For broadband random vibration analysis, the one-sided Power Spectral Density $S_{xx}[k]$ (in $\text{Units}^2/\text{Hz}$) is computed via the windowed periodogram:

$$
S_{xx}[k] = \begin{cases}
\frac{1}{F_s \sum_{n=0}^{N-1} w[n]^2} |X[k]|^2, & k = 0, \frac{N_{\text{fft}}}{2} \\
\frac{2}{F_s \sum_{n=0}^{N-1} w[n]^2} |X[k]|^2, & 0 < k < \frac{N_{\text{fft}}}{2}
\end{cases}
$$

### 1.4 Record-Limited Physical Resolution vs. Zero-Padded Discrete Bin Spacing
A crucial theoretical distinction exists between the physical frequency resolution and the discrete bin spacing of the zero-padded transform:
- **Nominal Record-Limited Frequency Scale (Rayleigh Limit):**
  $$\Delta f_{\text{record}} = \frac{1}{T} = \frac{F_s}{N}$$
  This defines a nominal record-limited frequency scale established by the finite observation record length $T = N \cdot \Delta t$. For the test dataset ($N = 13,336$ at $F_s = 200.0\text{ Hz}$, $T = 66.68\text{ s}$), $\Delta f_{\text{record}} \approx 0.0150\text{ Hz}$.
- **Zero-Padded Discrete Bin Spacing:**
  $$\Delta f_{\text{bin}} = \frac{F_s}{N_{\text{fft}}}$$
  Evaluating the FFT over $N_{\text{fft}} = 2^{\lceil \log_2(N) \rceil} = 16,384$ points provides a denser evaluation grid ($\Delta f_{\text{bin}} \approx 0.0122\text{ Hz}$) by sinc-interpolating the continuous DTFT spectrum. Zero-padding enhances visual peak localization and parabolic peak fitting precision, but does **not** increase the fundamental physical resolution beyond the nominal record-limited scale $\Delta f_{\text{record}}$.

---

## 2. Windowing Functions & Coherent Gain Correction

Truncating finite-duration experimental signals induces **spectral leakage** because the rectangular truncation boundary acts as a convolution with a Dirichlet sinc kernel $\frac{\sin(\pi f)}{\pi f}$ whose first side-lobe is only $-13.3\text{ dB}$ down.

### 2.1 Window Definitions
1. **Rectangular Window:**
   $$w[n] = 1, \quad 0 \le n \le N-1$$
2. **Hann Window (Raised Cosine):**
   $$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right)$$
   Side-lobe attenuation: $-31.5\text{ dB}$, roll-off: $-18\text{ dB/octave}$.
3. **Hamming Window:**
   $$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)$$
   Side-lobe attenuation: $-42.7\text{ dB}$.
4. **Blackman Window:**
   $$w[n] = 0.42 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right) + 0.08 \cos\left(\frac{4\pi n}{N-1}\right)$$
   Side-lobe attenuation: $-58.1\text{ dB}$.
5. **Flat Top Window:**
   $$w[n] = a_0 - a_1\cos(z) + a_2\cos(2z) - a_3\cos(3z) + a_4\cos(4z)$$
   where $z = \frac{2\pi n}{N-1}$ and coefficients $[a_0, \dots, a_4] = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]$. Designed specifically to eliminate scallop loss in peak amplitude calibration.

### 2.2 Coherent Gain Normalization
Multiplying $x[n]$ by $w[n]$ attenuates energy near the record boundaries, reducing the peak height of the discrete transform. To restore exact amplitude calibration:

$$CG = \frac{1}{N} \sum_{n=0}^{N-1} w[n]$$

- For Rectangular: $CG = 1.0000$
- For Hann: $CG \approx 0.5000$
- For Hamming: $CG \approx 0.5400$
- For Blackman: $CG \approx 0.4200$
- For Flat Top: $CG \approx 0.2156$

Dividing by $CG$ ensures that regardless of which window the user selects, the calibrated linear amplitude remains invariant.

---

## 3. Preprocessing: DC Offset & Sensor Bias Removal

Experimental acquisition hardware (accelerometers, LVDTs, strain gauges) inevitably suffers from static DC voltage bias and low-frequency thermal drifts. 

In the provided sample dataset `NonClassical_200Hz`:
- Channel 2 exhibits a static offset of $\approx -379.7\text{ units}$ (described approximately as $-375.4\text{ units}$).
- Channel 3 exhibits a static offset of $\approx -97.4\text{ units}$ (described approximately as $-98.2\text{ units}$).
- Channel 4 exhibits a static offset of $\approx -90.8\text{ units}$ (described approximately as $-94.7\text{ units}$).

If unprocessed, the DC spike at $0\text{ Hz}$ has magnitude $A[0] \approx 380\text{ units}$, which completely dwarfs the dynamic modal response (whose peak amplitudes are $\approx 0.5 - 3.0\text{ units}$) by over $40 - 60\text{ dB}$. 

The application provides:
1. **Mean Subtraction:**
   $$x'[n] = x[n] - \frac{1}{N}\sum_{i=0}^{N-1} x[i]$$
2. **Linear Detrending:**
   Solves the least-squares regression $\hat{x}[n] = \alpha \cdot n + \beta$ and computes $x'[n] = x[n] - \hat{x}[n]$, removing linear sensor drift over long acquisition intervals.

---

## 4. Modal Parameter Extraction & Peak Picking

### 4.1 Sub-Bin Parabolic Interpolation
Because the FFT produces discrete frequency bins with spacing $\Delta f = \frac{F_s}{N_{\text{fft}}}$, a physical resonant mode rarely falls exactly on a bin center. To avoid bin quantization error, the three spectral values around local maximum bin $k$ ($\alpha = A[k-1]$, $\beta = A[k]$, $\gamma = A[k+1]$) are fitted to a parabola:

$$\delta = \frac{1}{2} \frac{\alpha - \gamma}{\alpha - 2\beta + \gamma}, \quad \delta \in [-1, 1]$$

The refined modal frequency and peak amplitude are:

$$f_{\text{peak}} = (k + \delta) \cdot \Delta f$$
$$A_{\text{peak}} = \beta - \frac{1}{4}(\alpha - \gamma)\delta$$

### 4.2 Modal Damping Ratio Estimation ($\zeta$)
For each isolated resonant peak, the **half-power bandwidth method (3-dB drop)** is applied:

$$A_{\text{half}} = \frac{A_{\text{peak}}}{\sqrt{2}} \approx 0.7071 \cdot A_{\text{peak}}$$

Denoting $f_a$ and $f_b$ as the frequencies on either side of the resonance where the amplitude falls to $A_{\text{half}}$:

$$\Delta f_{3\text{dB}} = f_b - f_a$$
$$\zeta = \frac{\Delta f_{3\text{dB}}}{2 \cdot f_{\text{peak}}}$$

### 4.3 Classification as Observed Spectral Peaks
Unless structural geometry, spatial sensor coordinates, and cross-channel mode shape phase relationships are independently provided and verified, identified spectral maxima are classified strictly as **observed spectral peaks** rather than unverified physical structural mode shapes (such as bending or torsion).

---

## 5. Non-Classical Damping Considerations & Dataset Labeling

The sample dataset is distributed under the label `NonClassical_200Hz`. In structural dynamics theory, "non-classical" (non-proportional) damping describes a linear dynamic system where the viscous damping matrix $\mathbf{C}$ cannot be simultaneously diagonalized by the classical undamped mode shapes:

$$\mathbf{C} \ne \alpha \mathbf{M} + \beta \mathbf{K}$$

In genuinely non-classically damped structural systems:
1. Modal vectors are complex-valued; individual degrees of freedom do not pass through zero displacement at the same instant in time (introducing phase shifts between sensor channels).
2. Energy dissipation couples nominal normal modes rather than remaining decoupled along classical normal coordinates.

However, the file designation `NonClassical_200Hz` is treated here strictly as a dataset label. Without an independent structural finite-element model, state-space system matrices, or calibrated spatial phase measurements of the physical test specimen, non-classical damping cannot be definitively established from response channels alone. Multi-channel spectral inspection provides empirical amplitudes and cross-sensor frequency content to support subsequent operational modal analysis (OMA) workflows.

