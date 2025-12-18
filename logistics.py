import numpy as np
import matplotlib.pyplot as plt  

#1. User Input
def get_user_inputs():
    print("Enter preferred shipment levels:")
    t1 = float(input("t1: "))
    t2 = float(input("t2: "))
    t3 = float(input("t3: "))

    print("\nEnter cost weights:")
    w1 = float(input("w1: "))
    w2 = float(input("w2: "))
    w3 = float(input("w3: "))

    print("\nEnter route capacities:")
    c1 = float(input("Capacity c1: "))
    c2 = float(input("Capacity c2: "))
    c3 = float(input("Capacity c3: "))

    D = float(input("\nEnter daily demand D: "))

    print("\nEnter initial guess (x1, x2, x3):")
    x1 = float(input("x1_0: "))
    x2 = float(input("x2_0: "))
    x3 = float(input("x3_0: "))
    #return all user inputs as numpy arrays
    return np.array([t1, t2, t3]), np.array([w1, w2, w3]), np.array([c1, c2, c3]), D, np.array([x1, x2, x3])

#2. Problem Functions
def f(x, t, w):
    #quadratic cost function: penalizes deviation from preferred levels
    return w[0]*(x[0]-t[0])**2 + w[1]*(x[1]-t[1])**2 + w[2]*(x[2]-t[2])**2

def grad_f(x, t, w):
    #gradient of the cost function
    return np.array([2*w[0]*(x[0]-t[0]), 2*w[1]*(x[1]-t[1]), 2*w[2]*(x[2]-t[2])])

def h(x, D):
    #equality constraint: total shipments minus demand
    return np.sum(x) - D

def grad_h():
    #gradient of equality constraint
    return np.ones(3)

def g_list(x, c):
    #inequality constraints: capacities and non-negativity
    return np.array([x[0] - c[0], x[1] - c[1], x[2] - c[2], -x[0], -x[1], -x[2]])

def grad_g_list():
    #gradients of all inequality constraints
    return np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])

def hessian_f(w):
    #Hessian of the cost function
    return np.diag([2*w[0], 2*w[1], 2*w[2]])

#3. Quadratic Penalty Method
def Phi(x, t, w, c, D, mu):
    #penalized objective function
    g_vals = g_list(x, c)
    return f(x, t, w) + 0.5*mu*h(x, D)**2 + 0.5*mu*np.sum(np.maximum(0, g_vals)**2)

def grad_Phi(x, t, w, c, D, mu):
    #gradient of penalized function
    grad = grad_f(x, t, w) + mu*h(x, D)*grad_h()
    g_vals = g_list(x, c)
    grad_g = grad_g_list()
    for gi, gi_grad in zip(g_vals, grad_g):
        if gi > 0:
            grad += mu * gi * gi_grad
    return grad

def hessian_Phi(x, t, w, c, D, mu):
    #Hessian of penalized objective function
    H = hessian_f(w)
    gh = grad_h().reshape(-1,1)
    #adds penalty from equality constraint
    H += mu * (gh @ gh.T)
    #adds penalty contributions from violated inequalities
    g_vals = g_list(x, c)
    grad_g = grad_g_list()
    for gi, gg in zip(g_vals, grad_g):
        if gi > 0:
            gg = gg.reshape(-1,1)
            H += mu * (gg @ gg.T)
    return H

def newton_for_Phi(x0, t, w, c, D, mu, tol=1e-8):
    #Newton's method to minimize penalized function for fixed mu
    x = x0.copy()
    while True:
        g = grad_Phi(x, t, w, c, D, mu)
        H = hessian_Phi(x, t, w, c, D, mu)
        step = np.linalg.solve(H, g)
        x_new = x - step
        if np.linalg.norm(step) < tol:
            return x_new
        x = x_new

def penalty_method(x0, t, w, c, D, mu0=0.1, beta=10, tol=1e-3):
    #outer penalty loop: increases mu until constraints are satisfied
    #maximum of 15 iterations allowed — penalty methods typically converge in very few
    #outer steps for most real-world constrained problems, since μ grows rapidly (1→10→100→...)
    #Beyond this point the Hessian becomes ill-conditioned, so 15 iterations is more than 
    #sufficient for almost all practical cases while avoiding numerical instability
    mu = mu0
    x = x0.copy()

    obj_history = [] #penalty function values Φ(x;μ)
    h_history = []   #|h(x)| violations

    for k in range(15):
        print(f"\n--- Iteration {k}, mu={mu} ---")
        x = newton_for_Phi(x, t, w, c, D, mu)

        print("x =", x.tolist())   #current shipment vector
        print("h(x) =", h(x, D))   #equality constraint violation
        print("max(g_i) =", np.max(np.maximum(0, g_list(x, c))))  #max inequality violation

        #store histories
        obj_history.append(Phi(x, t, w, c, D, mu))
        h_history.append(abs(h(x, D)))

        #stop when constraints are nearly satisfied
        if abs(h(x, D)) <= tol:
            print("\nConstraints satisfied.")
            break

        mu *= beta  #increase penalty

    #plot 1: penalty function Φ(x; μ)
    plt.figure(figsize=(7,4))
    plt.plot(obj_history, marker='o')
    plt.title("Penalty Function Growth Φ(x; μ)")
    plt.xlabel("Iteration")
    plt.ylabel("Φ(x; μ)")
    plt.grid(True)
    plt.show()

    #plot 2: equality constraint convergence |h(x)|
    plt.figure(figsize=(7,4))
    plt.plot(h_history, marker='o')
    plt.title("Convergence of Equality Constraint |h(x)|")
    plt.xlabel("Iteration")
    plt.ylabel("|h(x)|")
    plt.grid(True)
    plt.show()

    return x

#4. Main program
t, w, c, D, x0 = get_user_inputs()
print("\n=== CONSTRAINED PENALTY METHOD SOLUTION ===")
x_star = penalty_method(x0, t, w, c, D)

print("\n--- FINAL OPTIMAL SOLUTION ---")
#final optimized solution
print("x* =", x_star.tolist())
#final objective value
print("f(x*) =", f(x_star, t, w))
#equality constraint status
print("Constraint h(x*) =", h(x_star, D))
#inequality constraint status
print("Inequalities g_i(x*) =", g_list(x_star, c).tolist())
