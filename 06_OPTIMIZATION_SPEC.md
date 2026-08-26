# 06_OPTIMIZATION_SPEC.md

## Mathematical Models
*AI Note: Do not modify these formulas without explicit human approval.*

### 1. Standard Units & Constants
- **Distance:** Micrometers (um)
- **Frequency:** Megahertz (MHz)
- **$d_{ref}$:** 10.0 um
- **$\sigma_f$:** 20.0 MHz
- **$w_c$ (Crosstalk Weight):** 0.70
- **$w_r$ (Routing Weight):** 0.30

### 2. Normalized Spatial Risk ($S_{ij}$)
$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$
$d_{norm} = \frac{d_{ij}}{d_{ref}}$
$S_{ij} = \frac{1}{1 + d_{norm}^3}$

### 3. Normalized Spectral Risk ($F_{ij}$)
$\Delta f_{ij} = |f_i - f_j|$
$F_{ij} = \exp\left(-\frac{\Delta f_{ij}^2}{2\sigma_f^2}\right)$

### 4. Base Interaction Risk ($R_{ij}$)
$R_{ij} = S_{ij} \times F_{ij}$

### 5. Intended vs. Unintended Coupling
Let $I_{ij} \in [0, 1]$ be the intended interaction weight from the `Connection` model.
- **Unintended Risk:** $R^{unintended}_{ij} = (1 - I_{ij}) R_{ij}$
- **Routing Cost:** $C_{routing, ij} = I_{ij} \left( \frac{d_{norm}^2}{1 + d_{norm}^2} \right)$

### 6. Layout Quality Score (LQS)
$P_{ij} = w_c R^{unintended}_{ij} + w_r C_{routing, ij}$
$TotalPenalty = \frac{1}{N_{pairs}} \sum P_{ij}$
$LQS = 100 \times (1 - TotalPenalty)$
*(Bounds: $0 \le LQS \le 100$)*

### 7. Hard Constraints (Q-DRC)
The optimizer MUST REJECT any candidate violating these rules:
- $d_{ij} \ge min\_qubit\_spacing\_um$ *(Minimum physical spacing is a universal hard constraint and applies to both intended and unintended qubit pairs.)*
- $min\_boundary\_clearance\_um \le x_i \le (chip\_width\_um - min\_boundary\_clearance\_um)$
- $min\_boundary\_clearance\_um \le y_i \le (chip\_height\_um - min\_boundary\_clearance\_um)$
- **Frequency Collision Rule:** 
  - For unintended pairs ($I_{ij} = 0$): If $d_{ij} < frequency\_check\_distance\_um$, then REQUIRE $\Delta f_{ij} \ge min\_frequency\_separation\_mhz$.
  - For intended pairs ($I_{ij} > 0$): Bypass strict frequency collision rejection to allow routing, but flag as a design warning.

### 8. Optimization Engine (Hill Climbing)
Algorithm parameters: `max_iterations = 500`, `patience = 50`, `step_size_um = 5.0`, `random_seed = 42`.
1. **Initialize:** Set random seed. Calculate baseline LQS.
2. **Select Target:** Select the qubit participating in the pair with the highest objective penalty ($P_{ij}$), prioritizing hard Q-DRC violations when present.
3. **Generate Candidates:** Generate 8 candidate moves for this qubit (N, S, E, W, NE, NW, SE, SW) by `step_size_um`.
4. **Validate:** Reject any candidates that violate Q-DRC hard constraints.
5. **Evaluate:** Calculate LQS for remaining valid candidates.
6. **Accept/Reject:** Accept ONLY if new LQS > current LQS.
7. **Iterate:** Repeat until `max_iterations` is reached, OR no improvement occurs for `patience` iterations.
8. **Finalize:** The complete Q-DRC and risk engine must be rerun entirely on the accepted layout.
