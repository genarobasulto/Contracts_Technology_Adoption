# Solving Finite-Time Dynamic Contracting Models with Network Effects

This repository contains the code for solving the finite-time dynamic contracting model presented in our paper (Contracts and tech. adoption).

### Model Details

The model determines optimal entry decisions, market shares, and contracts based on the following parameters:  
- **Entry costs** \(C_{mt}\)  
- **Marginal production costs** \(c_t\)  
- **Switching costs** \(P_t\)  
- **Industrial consumer-type distribution**  
- **Government incentives**  

### Computational Solution

The computational approach involves the following steps:  

1. **Initialization at Terminal Period:**  
   At time \(T\), compute all possible active market combinations \(\sigma(M, T)\) (e.g., with 2 markets: \((0,0), (0,1), (1,0), (1,1)\)).  

2. **Profit Maximization:**  
   For each state \((\sigma, T)\):  
   - Compute profits \(\Pi_T^m = \Pi_T^m - \text{cost}\) for all markets.  
   - Choose the optimal market entry decision that maximizes profits.  
   - Save the decision and corresponding value functions.  

3. **Backward Induction:**  
   - At \(T-1\), compute all possible active market combinations \(\sigma(M, T-1)\).  
   - Repeat Step 2, considering future value functions computed at \(T\).  
   - Continue backward in time to \(t = 0\).  

4. **Reducing Complexity:**  
   - Restrict firms to enter a limited number of markets at one time (suitable for markets with high entry costs or capacity constraints).  
   - Use specific functional form assumptions to simplify integrations over future profits.

## Features

- Computes optimal entry decisions and market shares.  
- Handles high-dimensional state spaces using simplifying assumptions.  
- Flexible input parameters for entry costs, production costs, consumer distribution, and incentives.

## Repository Contents

- **Code basic_functions_v2:** The program for solving the model.  
- **Example 1-4:** Example datasets to run simulations.
- ** NY calibration and solution: ** Calibration data from NY ATC reporting and solution of the model.
- **_plots:** Code to generate solution plots.
- **No contracts functions:** The program for solving the benchmark model with single prices.  
- **Documentation:** Detailed explanation of functional form assumptions and computational strategies.

## How to Use

1. Clone this repository:  
   ```bash
   git clone https://github.com/genarobasulto/Contracts_Technology_Adoption
   cd Contracts_Technology_Adoption
   ```

2. Install dependencies (if applicable).  

3. Follow the instructions in the documentation to set up input parameters and run the model.

4. We prepared an example of usage in the file run_simulations_github_example.
   The basic functions to solve the model are in basic_functions_price_uncertanty_v2 and
   no_contracts_functions_price_uncertanty. We also included code to plot the solutions. 

## Assumptions and Limitations

- Entry is limited to a certain number of markets per period.  
- Functional form assumptions simplify computations of future profits and uncertainty.  
- Results depend on parameter values and input assumptions.  

## Citation

If you use this code, please cite the repository:  
[Contracts and Technology Adoption](https://github.com/genarobasulto/Contracts_Technology_Adoption)  
