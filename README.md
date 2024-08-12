# Adaptive Guidance Law for Pursuer with Accelerating Target

This code implements three guidance laws from [this paper](https://arc.aiaa.org/doi/abs/10.2514/1.G007664) [1]. The three guidance laws are ProNav (PNG), finite time convergence guidance (FTCG), and adaptive finite time nonlinear guidance (AFTNG).

Each of the three guidance laws in action can be seen below. All three exhibit a successful collision with the target, but at different times:

![3_laws](https://github.com/user-attachments/assets/2d3becb9-04ca-486f-b801-e4176e0ca879)

## Using the code
Individual sim files (in the form of python scripts) can be run in order to evaluate performance under different acceleration profiles of the target.

## References
[1] A. J. Calise, “Adaptive Finite Time Intercept Guidance,” Journal of Guidance, Control, and Dynamics, vol. 46, no. 10, pp. 1975–1980, Oct. 2023, doi: 10.2514/1.G007664.

