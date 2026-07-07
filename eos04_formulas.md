# EOS-04 Radiometric Calibration – Mathematical Formulas

This document summarizes all mathematical equations implemented in the
`eos04_radiometric_preprocessing.py` script based on the **EOS-04 Data Products Formats v1.2.5** documentation.

---

# 1. Calibration Constant Conversion

The calibration constant provided in `BAND_META.txt` is stored in decibels (dB).

Convert it into linear scale before calibration.

\[
K_{cal}^{linear}=10^{\frac{K_{cal}^{dB}}{10}}
\]

---

# 2. Digital Number (DN)

## Ground Range / Level-2 Products

The Digital Number is directly stored in the GeoTIFF.

\[
DN = \text{Pixel Value}
\]

---

## SLC Products

For complex SAR products,

\[
DN=\sqrt{DN_I^2+DN_Q^2}
\]

where

- \(DN_I\) = In-phase component
- \(DN_Q\) = Quadrature component

---

# 3. Noise Bias Removal

If IMAGE_NOISE_BIAS is applied,

\[
DN_{corrected}^2 = DN^2 - IMAGE\_NOISE\_BIAS
\]

Optionally,

\[
DN_{corrected}^2=\max(DN_{corrected}^2,0)
\]

to avoid negative power values.

---

# 4. Beta Naught (β⁰)

Equation (1)

\[
\beta^0=\frac{DN^2}{K_{cal}^{linear}}
\]

---

# 5. Sigma Naught (σ⁰)

Equation (2)

\[
\sigma^0=\frac{DN^2\sin(i)}{K_{cal}^{linear}}
\]

where

- \(i\) = Local Incidence Angle

---

# 6. Gamma Naught (γ⁰)

Equation (3)

\[
\gamma^0=\frac{DN^2\tan(i)}{K_{cal}^{linear}}
\]

---

# 7. Linear to Decibel Conversion

For power quantities,

\[
Power_{dB}=10\log_{10}(Power_{linear})
\]

---

For amplitude quantities,

\[
Amplitude_{dB}=20\log_{10}(Amplitude)
\]

---

# 8. Radar Cross Section (Integration Method)

Equation (5)

First,

\[
ScatteringArea=
OutputLineSpacing
\times
OutputPixelSpacing
\]

Then,

\[
\sigma=
\frac{
\left(\sum DN^2\right)
\times
ScatteringArea
}
{K_{cal}^{linear}}
\]

---

# 9. Radar Cross Section (Peak Method)

Equation (6)

First,

\[
ScatteringArea=
OutputAzimuthResolution
\times
OutputRangeResolution
\]

Then,

\[
\sigma=
\frac{
DN_{peak}^2
\times
ScatteringArea
}
{K_{cal}^{linear}}
\]

---

# 10. Level-2B Terrain Normalized Gamma Naught (dB)

Equation (9)

\[
\gamma^0_{dB}
=
20\log_{10}(DN)
-
K_{cal}^{dB}
\]

---

# 11. Level-2B Terrain Normalized Gamma Naught (Linear)

Equation (11)

\[
\gamma^0=
\frac{DN^2}{K_{cal}^{linear}}
\]

---

# 12. Undo Radiometric Terrain Correction (RTC)

These equations recover the un-normalized backscatter.

---

## Beta Naught

Equation (13)

\[
\beta^0=
\gamma^0
\times
LocalIlluminationArea
\]

---

## Sigma Naught

Equation (14)

\[
\sigma^0=
\beta^0
\sin(i)
\]

---

## Gamma Naught (Un-normalized)

Equation (15)

\[
\gamma^0=
\beta^0
\tan(i)
\]

---

# 13. Incidence Angle Conversion

Degrees are converted into radians before trigonometric operations.

\[
i_{rad}
=
i_{deg}
\times
\frac{\pi}{180}
\]

Used in

- Beta⁰
- Sigma⁰
- Gamma⁰
- Undo RTC

---

# Summary of All Equations

| Equation | Formula |
|-----------|---------|
| Calibration Constant | \(K_{cal}^{linear}=10^{K_{cal}^{dB}/10}\) |
| SLC DN | \(DN=\sqrt{DN_I^2+DN_Q^2}\) |
| Noise Bias | \(DN^2-IMAGE\_NOISE\_BIAS\) |
| Beta⁰ | \(\beta^0=\frac{DN^2}{K_{cal}^{linear}}\) |
| Sigma⁰ | \(\sigma^0=\frac{DN^2\sin(i)}{K_{cal}^{linear}}\) |
| Gamma⁰ | \(\gamma^0=\frac{DN^2\tan(i)}{K_{cal}^{linear}}\) |
| Linear → dB | \(10\log_{10}(x)\) |
| Amplitude → dB | \(20\log_{10}(x)\) |
| RCS (Integration) | \(\sigma=\frac{\sum DN^2\times Area}{K_{cal}^{linear}}\) |
| RCS (Peak) | \(\sigma=\frac{DN_{peak}^2\times Area}{K_{cal}^{linear}}\) |
| Level-2B Gamma⁰ (dB) | \(20\log_{10}(DN)-K_{cal}^{dB}\) |
| Level-2B Gamma⁰ (Linear) | \(\frac{DN^2}{K_{cal}^{linear}}\) |
| Undo RTC Beta⁰ | \(\beta^0=\gamma^0\times LocalIlluminationArea\) |
| Undo RTC Sigma⁰ | \(\sigma^0=\beta^0\sin(i)\) |
| Undo RTC Gamma⁰ | \(\gamma^0=\beta^0\tan(i)\) |
