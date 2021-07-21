import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.optimize import fsolve

#This code will numerically solve the Fisher-KPP equation
#   u_t = u_xx + alpha * u * (1 - u)
#by discretizing and using Newton's method.

a = 25.0;   #Equation will be solved on interval (-a,a).
N = 1000;  #Number of mesh points.
h = 2 * a / N;  #Distance between mesh point.
alpha = 1;
c = 2 * np.sqrt(alpha) #We expect the solution to be traveling fronts moving at this speed.
T = 5; #Maximum time for which we will solve the equation.
DeltaT = T / 5; #Plot the solution at times k * DeltaT for k an integer.

def norm(u):
    return np.sum(u**2)

#Initial guess for the solution.
def u0(x):
    return np.exp(-c * x / 2) / (1 + np.exp(-c * x / 2))

#Uses Newton's method to find zeros of f
#At each iteration updates u0 to u0 + p for p minimizing |f(u0)|_{L^2}.
#   u0 is the initial guess for the minimizer of |f(x) - b|_{L^2}
#   J is the jacobian of f computed at the point u0.
#   tol is the tolerance to within which we look for a solution.
def Newton(f, J, u, tol):
    steps = 0

    while (norm(f(u)) > tol and steps < 1000):
        u = u + np.linalg.solve(J(u), - f(u))
        steps += 1
    
    return u

def update(u, alpha, h, a, tol):
    #solving system of equations given by
    #    -cu' - u'' = alpha u (1 - u)
    #where left side is discretized with dt = h**2 = (dx)**2. Leads to looking for solutions u_{k+1} to
    #   u_{k+1}(x_j) - u_k(x_j) - (u_{k+1}(x_{j+1}) - 2 * u_{k+1}(x_j) + u_{k+1}(x_{j-1})) = h**2 * alpha * u_{k+1}(x_j) * (1 - u_{k+1}(x_j))
    def f(v, u, alpha, h):
    
    #  Accounting for boundaries
        vl = np.roll(v, 1)
        vl[0] = u0(-a)
        
        vr = np.roll(v, -1)
        vr[len(v) - 1] = u0(a)
        
        pde_operator = (v - u) / h**2 - (vl - 2 * v + vr) / h**2 - alpha * v * (np.ones(len(v)) - v)
        return pde_operator
    
    def J(v, alpha, h):
        return np.identity(len(v)) / h**2 - (np.diag(np.ones(len(v) - 1), 1) - 2 * np.identity(len(v)) + np.diag(np.ones(len(v) - 1), -1)) / h**2 - alpha * np.diag(np.ones(len(v)) - 2 * v)
    
    return Newton(lambda v: f(v, u, alpha, h), lambda v: J(v, alpha, h), u, tol)

#Extra space on the mesh to account for spilling over the boundaries when computing discretized u_xx.
x = np.arange(-a + 2 * a / N, a - 2 * a / N, 2 * a / N)

initial_guess = np.vectorize(u0)

u = initial_guess(x)
tol = 10**(-6)  #Looking for solutions to within this error.
t = 0

while (t < T):
    u = update(u, alpha, h, a, tol)
    if (np.floor(t / DeltaT) > np.floor((t - h**2) / DeltaT)):
        plt.plot(x, u, label='time %s' %t)
    
    t += h**2

plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()
