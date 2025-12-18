# Constrained Optimization Logistics Problem

## Overview
This project formulates and solves a constrained optimization logistics problem involving optimal route allocation for organizations under demand and capacity constraints. The objective is to minimize operational cost while satisfying equality and inequality constraints that arise in real-world logistics systems.

The problem is solved using the **Quadratic Penalty Method**, with each penalized subproblem optimized using **Newton’s Method**. The implementation demonstrates how feasibility is progressively enforced while maintaining numerical stability and convergence.

---

## Problem Description
An organization distributes goods through three transportation routes. Each route is characterized by:
- A preferred operating (shipment) level
- A cost weight penalizing deviations from the preferred level
- A maximum capacity constraint

The system must collectively satisfy a fixed total demand while ensuring all operational constraints are respected.

---

## Mathematical Formulation

### Decision Variables
Let:
- \( x_1, x_2, x_3 \) denote the shipment quantities allocated to the three routes.

---

### Objective Function
The cost function penalizes deviations from preferred shipment levels:
f(x) = w1(x1 − t1)² + w2(x2 − t2)² + w3(x3 − t3)²
where:
- \( t_i \) are preferred shipment levels  
- \( w_i \) are cost weights  

---

### Constraints

#### Equality Constraint (Demand Satisfaction)
x1 + x2 + x3 − D = 0

#### Inequality Constraints (Capacity and Non-Negativity)
0 ≤ xi ≤ ci , i = 1, 2, 3

---

## Quadratic Penalty Method
To handle constraints, the problem is transformed into a sequence of unconstrained problems using a quadratic penalty function:
Φ(x; μ) = f(x) + (μ/2) h(x)² + (μ/2) ∑ max(0, gi(x))²
where:
- \( h(x) \) represents the equality constraint
- \( g_i(x) \) represent inequality constraints
- \( μ \) is the penalty parameter

As \( μ \) increases, constraint violations are penalized more heavily, driving the solution toward feasibility.

---

## Optimization Strategy
- Analytical gradients and Hessians are derived for the penalized objective
- Newton’s Method is applied to minimize each penalized subproblem
- The penalty parameter is increased iteratively
- Optimization terminates when constraint violations fall below a specified tolerance

This approach balances cost minimization with constraint enforcement.

---

## Convergence and Outputs
The implementation produces:
- Iterative optimization logs
- Penalty function growth across iterations
- Equality constraint convergence plots
- Final optimal and feasible route allocation

These outputs illustrate how feasibility improves as the penalty parameter increases.

---

## Technologies Used
- Python
- NumPy
- Matplotlib

---

## Files
- `logistics.py` – complete implementation of the quadratic penalty method
- `Constrained_Optimization_Logistics_Report.pdf` – mathematical derivation, numerical results, and discussion

---

## Academic Context
This project was completed as part of **AIT203 – Optimization** and demonstrates:
- Constrained optimization modeling
- Penalty-based methods
- Newton’s Method for numerical optimization
- Trade-offs between feasibility enforcement and numerical stability

