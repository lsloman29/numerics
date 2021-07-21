import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fft import rfft, irfft, rfftfreq
from scipy import integrate
from scipy.optimize import curve_fit

#This code will numerically solve the international trade equation
#       u_t = u_xx + alpha * (1 - M) * (1 - u)
#where M = -integral_{-\infty}^x ((1 - u(t,y+delta))^{n-1} + (n-1) * (1 - u(t,y-delta)) * (1 - u(t,y))^{n-2}) * u_y(t,y) dy.
#by approximating it as the sum of solutions to the two equations
#       u_t = u_xx,     u_t = alpha * (1 - M) * (1 - u).
#This equation describes the evolution of a country's productivity distribution when it participates in global trade according to the model of Alvarez, Buera, and Lucas (2013).

N = 2**12;  #Number of mesh points.
a = 10.0;   #Interval on which the problem will be solved (-a,a).
h = 2 * a / (N-1);  #Distance between mesh points.
alpha = 1;
n = 5; #Number of countries.

maxT = 5.0; #Time up to which we model the solution.
deltaT = maxT / 20;  #Save the data at times k * deltaT.

delta = math.ceil(2 / h) * h;   #Model parameter, related to the shipping taxes countries impose on one another.
c = 2 * np.sqrt(n * alpha); #Speed at which we expect the solution to travel.
lambdac = c / 2; #Parameter based on c for estimating shape of the solution.

def innerProduct(u, v):
    if (len(u) != len(v)):
        print("Error! Lengths of arguments must be equal.")
        return "Error: Arguments had unequal lengths."
    else:
        product = np.zeros(len(u[0]))
        
        for j in range(0, len(u)):
            if (len(u[j]) != len(u[0])):
                print("Error! Arguments must be rectangular arrays.")
                return "Error: Arguments have inconsistent dimensions."
            else:
                product += u[j] * v[j]
            
        return product
        
#Location at which we expect the traveling fronts to be centered.
def BramsonCorrection(t, c):
    return c * t - 3 * np.log(t) / c

def u0(x, c):
#initial value of u
    return np.exp(-lambdac * x) / (1 + np.exp(-lambdac * x))

#returns vector u containing u(x) for a mesh x shifted to u(x + delta)
def plusDelta(u, delta, h):
    deltaShift = math.ceil(delta / h)
    return np.concatenate((u[deltaShift:], [u[len(u)-1]] * deltaShift))
    
#returns vector u containing u(x) for a mesh x shifted to u(x - delta)
def minusDelta(u, delta, h):
    deltaShift = math.ceil(delta / h)
    return np.concatenate(([u[0]] * deltaShift, u[:len(u) - deltaShift]))
    
#this returns I where M = integral from -inf to x of I * F_y
def integrandM(u, n, h, delta):
    F = np.ones(len(u)) - u;
    return plusDelta(F, delta, h)**(n-1) + (n-1) * minusDelta(F, delta, h) * (F**(n-2))

#Computes vector of M(x,t) for x in the specified mesh where
#M = integral of ( F(y+delta)^(n-1) + (n-1) F(y-delta) F(y)^(n-2) ) f(y) dy
#with f = F_y
def M(x, u, n, delta):
    h = x[1] - x[0]
    f = - np.diff(np.append(u, u[len(u)-1])) / h
    I = integrandM(u, n, h, delta);
    return integrate.cumtrapz(I * f, x, initial=0)
    
def diffusionTerm(u, x):
    h = x[1] - x[0]
    u_periodic = np.concatenate((u[::-1], u))
    freqs = rfftfreq(len(u_periodic), h)
    
    #diffusion contribution to Fourier transform at time t
    #using that diffusion obeys
    #   f_s = f_xx,  f(x,0) = u(x,t)
    #so that F(f)(k,s) = F(u(x,t))(k,0) exp(-n_k^2 s)
    #Fourier transform of diffusion term at time dt
    uDiffused = irfft(rfft(u_periodic) * np.exp(- (2 * np.pi * freqs * h) ** 2))
    
    return uDiffused[len(u):] - u
    
def reactionTerm(u, x, M, alpha):
    #nonlinear term of
    #    u_t - u_xx = alpha (1 - M) (1 - u)
    #given by solution to g_s = alpha (1 - M)(1 - g) for 0 < s < dt, g(x,0) = u(x,t)
    #using Taylor series with k terms, so error = O(h^(2k+2)) since dt = h^2
    h = x[1] - x[0]
    
    F = np.ones(len(u)) - u
    f = np.diff(np.append(F, F[len(u)-1])) / h;
    Id = np.ones(len(u))
    
    #The order-1 coefficient is u_t = alpha * (1 - M) F
    coeff1 = alpha * (Id - M) * F

    #The order-2 coefficient is u_tt = alpha * ((1 - M)F)_t = -alpha * (M_t F + alpha (1 - M)^2 F)
    #Since M is an integral of I(y,t)f(y,t) from -inf to x, M_t is as well. So first we calculate the integrand:M,
    #M_xt = I_t f + I f_t
    #     = I_t f(y) + I F_ty
    #     = I_t f + I (-alpha * (1 - M)F )_y
    #     = I_t f - alpha * I (- IF + 1 - M )f
    I = integrandM(u, n, h, delta)
    
    #with vecM and vecF as defined below,
    # d^k/dt^k I = <(vecM**(k-1))_t, vecF> + <vecM**k, vecF>
    vecM = - alpha * np.vstack(((n-1) * (Id - plusDelta(M, delta, h)), (n-1) * Id - (n-2) * M - minusDelta(M, delta, h)))
    vecF = np.vstack((plusDelta(F, delta, h)**(n-1), minusDelta(F, delta, h) * F**(n-2)))
    
    It = innerProduct(vecM, vecF)
    ft = - alpha * ((Id - M) * f - I * F)

    Mxt = It * f + I * ft
    Mt = integrate.cumtrapz(Mxt, x, initial=0)

    coeff2 = -alpha * (Mt + alpha * (Id - M)**2) * F
    
    vecMt = alpha * np.vstack(((n-1) * plusDelta(Mt, delta, h), (n-2) * Mt + minusDelta(Mt, delta, h)))
    
    Itt = innerProduct(vecMt + vecM**2, vecF)
    ftt = alpha * (Mxt * F + Mt * f - 2 * alpha * (Id - M) * I * F * f + alpha * (Id - M)**2 * f)
    Mxtt = Itt * f + 2 * It * ft + I * ftt
    Mtt = integrate.cumtrapz(Mxtt, x, initial=0)
    
    coeff3 = - alpha * F * (Mtt - 3 * alpha * (Id - M) * Mt - alpha**2 * (Id - M)**3)

    return coeff1 * (h**2) + coeff2 * (h**2)**2 + coeff3 * (h**2)**3
    
def update(u, x, M, alpha):
    return u + reactionTerm(u, x, M, alpha) + diffusionTerm(u, x)
    
def distance(u, v):
    return np.sum(np.square(u - v))
    
#u should be a decreasing function, so our numerics fail as soon as it stops decreasing. This is due to error in the Taylor series solution for the reaction equation
#   u_t = alpha * (1 - M) * (1 - u).
def decreasing(u, epsilon):
    return all(w < v + epsilon for v, w in zip(u, u[1:]))

x = np.linspace(-a, a , N, dtype = 'float64')

u = u0(x, lambdac)
Mvec = M(x, u, n, delta)

t = 0.0
correctionVec = []
timeVec = []

shift = 0
epsilon = 10**(-5)

#We let the equation evolve until the errors get too large or until maxT has been reached.
#Due to the approximation in the Taylor series, the numerics won't be valid for all time. Increasing the timescale requires more terms in the Taylor series.
while (t < maxT and decreasing(update(u, x, Mvec, alpha), epsilon)):
    timeVec = np.append(timeVec, t)

    if (t // deltaT > (t - h**2) // deltaT):
        plt.plot(x, u, label='time %s' %t)
        np.save("u_ITE_delta_%d_alpha_%d_c_%d_t_%d.npy" % (delta, alpha, c, t), u)
        print("Time elapsed: ", t)
        

    u = update(u, x, Mvec, alpha)
    Mvec = M(x, u, n, delta)

#    we want the wavefronts to stay at a_0 and -a_0; a_0 = -a + (a _0 + a)/h * h so for
#    K = math.ceil((a_0 + a)/h) we should have u[K] < 1/2
    while (u[(N - 1) // 2] > 0.52):
        u = np.delete(u, [0])
        u = np.append(u, u[N-2])
        shift += h;

    while (u[N // 2] < 0.48):
        u = np.delete(u, [N - 1])
        u = np.insert(u, 0, [u[0]])
        shift -= h;

    t += h**2

print("Time evolved is: ", t)

plt.xlabel("x - shift(t)")
plt.legend()
plt.show()

plt.plot(timeVec, correctionVec, label="shift vs. time")
plt.plot(timeVec, - c * timeVec + 3 * np.log(timeVec) / (2 * lambdac), label="Bramson Correction")
plt.xlabel("t")
plt.xlabel("Distance Traveled by Wavefront")
plt.legend()
plt.show()

params, cov = curve_fit(BramsonCorrection, timeVec, correctionVec)
print("c = ", params[0], "gamma = ", params[1] * lambdac)
