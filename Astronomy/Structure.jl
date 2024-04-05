# Solving the differential equations for the stellar structure model. For more references https://en.wikipedia.org/wiki/Stellar_structure. First aim is to get the density, temperature, pressure, and luminosity profiles of the 1M_sun star.

# The differential equations are:
# 1. dP/dr = -G * m(r) * ρ(r) / r^2
# 2. dT/dr = -G * m(r) * ρ(r) / r^2 * T / P * ∇
# 3. dL/dr = 4 * π * r^2 * ρ(r) * ε
# 4. dm/dr = 4 * π * r^2 * ρ(r)

using DifferentialEquations
using Plots
# I'm gonna solve these 4 equations using the DifferentialEquations.jl package. The first step is to define the equations as functions.

function dPdr(r, P, T, m, ρ)
    return -G * m * ρ / r^2
end

function dTdr(r, P, T, m, ρ)
    return -G * m * ρ / r^2 * T / P * ∇
end

function dLdr(r, P, T, m, ρ, ε)
    return 4 * π * r^2 * ρ * ε
end

function dmdr(r, P, T, m, ρ)
    return 4 * π * r^2 * ρ
end
# ρ needs to be calculated from P and T using the equation of state at every step. 
"""
function ρ(P, T)
    return μ * m_H *P / (k * T)-aT^4/3
"""
ρ=[]
X= 0.7
Y= 0.27
Z=0.02
X_CNO=0.01

# from the proton-proton chain
function τ(T::Vector{Float64})
    return 33.8 * (T / 1e6) .^ (-1/3)
end

# from the CNO cycle
function τ_CNO(T::Vector{Float64})
    return 152.3 * (T / 1e6) .^ (-1/3)
end
# ε is the TOTAL energy generation rate
function ϵ(X::Float64, X_CNO::Float64, τ::Vector{Float64}, τ_CNO::Vector{Float64}, ρ::Float64)
    return X * ρ * (0.00212 * X * τ.^2 .* exp.(-τ) .+ 3.7e16 * X_CNO * τ_CNO .* exp.(-τ_CNO))
end

function κ(Z::Float64, X::Float64, ρ::Float64, T::Float64)
    return 3e21 * Z * (1 + X) * ρ * T^(-3.5) + 0.02 * (1 + X)
end




# I am going to integrate from the center of the star to the surface. The initial conditions are:
# P(0) = P_c, T(0) = T_c, L(0) = 0, m(0) = 0

P_c = 4e15
T_c = 2.9e7
L_c = 0
r_c = 0
m_c = 0

print("slay")