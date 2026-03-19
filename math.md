# Mathematical Formulation

This document outlines the mathematical foundations of the double pendulum system, the baseline neural network model, and the Hamiltonian Neural Network (HNN) used in this project.

---

## 1. System Representation

We represent the double pendulum using generalized coordinates and conjugate momenta:

- $\theta_1, \theta_2$: angular positions  
- $p_1, p_2$: conjugate momenta  

The full state vector is:

$$
z = (\theta_1, \theta_2, p_1, p_2) \in \mathbb{R}^4
$$

The system evolves according to:

$$
\frac{dz}{dt} = f(z)
$$

The double pendulum is a nonlinear, chaotic system, meaning small perturbations in initial conditions lead to large deviations over time.

---

## 2. Hamiltonian Mechanics

In Hamiltonian mechanics, the dynamics of a system are derived from a scalar function called the Hamiltonian:

$$
H(\theta, p)
$$

which represents the total energy of the system:

$$
H = T + V
$$

where:
- $T$ is kinetic energy  
- $V$ is potential energy  

---

## 3. Double Pendulum Hamiltonian (Conceptual Form)

For a double pendulum, the Hamiltonian consists of:

- kinetic energy contributions from both masses  
- gravitational potential energy  
- nonlinear coupling terms due to geometry  

While the full closed-form expression is complex, the key idea is:

$$
H(\theta_1, \theta_2, p_1, p_2)
$$

encodes all dynamics of the system.

---

## 4. Hamilton’s Equations

The time evolution of the system is given by Hamilton’s equations:

$$
\frac{d\theta_i}{dt} = \frac{\partial H}{\partial p_i}
$$

$$
\frac{dp_i}{dt} = -\frac{\partial H}{\partial \theta_i}
$$

for $i = 1, 2$.

---

## 5. Compact Form (Symplectic Structure)

We can write the system in vector form:

$$
\dot{z} = J \nabla H(z)
$$

where $J$ is the symplectic matrix:

$$
J =
\begin{bmatrix}
0 & I \\
- I & 0
\end{bmatrix}
$$

and $\nabla H(z)$ is the gradient of the Hamiltonian.

---

## 6. Energy Conservation

A key property of Hamiltonian systems is conservation of energy.

We can show:

$$
\frac{dH}{dt} = \nabla H \cdot \dot{z}
$$

Substituting $\dot{z} = J \nabla H$:

$$
\frac{dH}{dt} = \nabla H \cdot (J \nabla H)
$$

Since $J$ is skew-symmetric:

$$
J^T = -J
$$

we obtain:

$$
\nabla H \cdot (J \nabla H) = 0
$$

Thus:

$$
\frac{dH}{dt} = 0
$$

This means the total energy remains constant over time.

---

## 7. Numerical Integration (RK4)

To simulate trajectories, we use the 4th-order Runge–Kutta method.

Given step size $h$:

$$
k_1 = f(z_t)
$$

$$
k_2 = f\left(z_t + \frac{h}{2} k_1\right)
$$

$$
k_3 = f\left(z_t + \frac{h}{2} k_2\right)
$$

$$
k_4 = f(z_t + h k_3)
$$

$$
z_{t+1} = z_t + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

This provides accurate and stable time integration for the system.

---

## 8. Baseline Neural Network

The baseline model learns the dynamics directly:

$$
\frac{dz}{dt} \approx f_\theta(z)
$$

where $f_\theta$ is a neural network.

The training objective is:

$$
\mathcal{L} = \| f_\theta(z) - \dot{z}_{\text{true}} \|^2
$$

### Limitation

This model does not enforce physical structure. Therefore:

- energy is not conserved  
- errors accumulate over time  
- trajectories become unstable  

---

## 9. Hamiltonian Neural Network (HNN)

Instead of learning dynamics directly, the HNN learns:

$$
H_\theta(z)
$$

Then dynamics are computed via:

$$
\dot{z} = J \nabla H_\theta(z)
$$

### Key Idea

The neural network outputs a scalar energy function, and the dynamics are derived using Hamilton’s equations.

---

## 10. Why HNN Improves Stability

Because the model enforces:

$$
\dot{z} = J \nabla H_\theta(z)
$$

it inherits:

$$
\frac{dH}{dt} = 0
$$

approximately.

This leads to:

- reduced energy drift  
- improved long-term stability  
- physically consistent trajectories  

---

## 11. Evaluation Metrics

### 1. Trajectory Error (MSE)

$$
\text{MSE} = \frac{1}{T} \sum_{t=1}^T \| z_t^{\text{pred}} - z_t^{\text{true}} \|^2
$$

### 2. Energy Drift

$$
\text{Drift} = \max_t |H(z_t) - H(z_0)|
$$

### 3. Phase Space Fidelity

Qualitative comparison of trajectories in phase space to evaluate structural preservation.

---

## 12. Connection to Experimental Results

The theoretical properties explain the observed performance:

- HNN exhibits significantly lower energy drift due to enforced conservation  
- Standard NN accumulates error due to lack of structure  
- Improved trajectory accuracy follows from improved stability  

---

## 13. Summary

- The double pendulum is a chaotic Hamiltonian system  
- Standard neural networks approximate dynamics without constraints  
- HNNs enforce physical structure through energy modeling  
- This leads to superior long-term prediction performance  