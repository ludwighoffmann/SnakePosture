module Beam

using ..Convolution: conv_setup, convu!#, DirectLinearConvolution!

using OrdinaryDiffEq
using SparseArrays
using StaticArrays
using LinearSolve
using ForwardDiff
using Symbolics
using DataInterpolations


# ====================================================
#                     Curve solver
# ====================================================

@doc raw"""
    curve(θ, sgrid)

    Compute the curve (x(s), y(s)) from the angle θ(s) by solving the first order ODE
    ```math
    x'(s) = \cos(\theta(s) - \pi/2), y'(s) = \sin(\theta(s) - \pi/2)
    ```
    where θ(s) is linearly interpolated from the values `θ` at the points `sgrid`.
"""
function curve(θ::AbstractVector{<:Real}, sgrid::AbstractVector{<:Real};
        ode_alg::OrdinaryDiffEqAlgorithm = Tsit5(), kwargs...)

    # ODEProblem requires continuously defined input; must interpolate θ(s)
    θ_itp = DataInterpolations.LinearInterpolation(θ, sgrid)

    function curve_f(u, p ,t)
        SA[cos(θ_itp(t) + π/2.0), sin(θ_itp(t) + π/2.0)]
    end

    # integrate with respect to s
    u0 = SA[0.0, 0.0]
    tspan = (sgrid[begin], sgrid[end])
    prob = ODEProblem(curve_f, u0, tspan)
    curve_sol = solve(prob, ode_alg; kwargs...)

    return curve_sol
end

"""
    Compute the curve (x(s), y(s)) from the solution `u` of the beam equation
"""
curve(u::SciMLBase.ODESolution, sgrid::AbstractVector{<:AbstractFloat}, t::AbstractFloat;
        kwargs...) = curve(u(t)[:, 1], sgrid; kwargs...)

"""
    Compute the curve (x(s), y(s)) from the solution `u` of the beam equation
"""
curve(u::SciMLBase.ODESolution, sgrid::AbstractVector{<:AbstractFloat}, i::Integer;
        kwargs...) = curve(u[:, 1, i], sgrid; kwargs...)



# =====================================================
#                     Beam equation
# =====================================================

# ----------------------------------------------
#           Beam equation coefficients
# ----------------------------------------------

mutable struct BeamCoeffs
    a::Float64      # gravitational strength
    b::Float64      # feedback strength
    c::Float64      # wave speed
    d::Float64      # damping coefficient
    α::Float64      # coefficient in Robin b.c.
    λ::Float64      # length scale of nonlocal feedback kernel
    μ::Float64      # speed of delayed feedback
end

BeamCoeffs(; a, b, c, d=0.0, α=1.0, λ=1.0, μ=1.0) = BeamCoeffs(a, b, c, d, α, λ, μ)

#const FloatOrDual = Union{AbstractFloat, ForwardDiff.Dual}


# ------------------------------------------
#           Spatial discretization
# ------------------------------------------

function beam_eq_loop!(dv::AbstractVector{<:Real}, θ::AbstractVector{<:Real},
        sgrid::AbstractVector{<:AbstractFloat}, cc::Real, a::Real,
        start::Integer, stop::Integer)

    @inbounds for j in start:stop
        jl, jr = j-1, j+1
        s = sgrid[j]

        dv[j] += cc * (θ[jl] + θ[jr] - 2.0*θ[j]) + a*(1.0 - s)*sin(θ[j])
    end
end


# ----- Robin-Neumann b.c. -----
"""
    beam_eq_RN!(dv, θ, sgrid, ds, c, a, dsα)

Computes a second order finite different approximation of c²∂_x^2 θ - a(1-s) cos(θ)
with Robin boundary conditions on the left endpoint and Neumann boundary conditions
on the right endpoint.
"""
function beam_eq_RN!(dv::AbstractVector{<:Real}, θ::AbstractVector{<:Real},
        sgrid::AbstractVector{<:AbstractFloat}, dsα::Real,
        cc::Real, a::Real)

    # Robin b.c. θ = α θ' at left endpoint
    # θ_ghost = θ[begin+1] - 2.0 * (dsα) * θ[begin]     # dsα = ds/α
    dv[begin] += ( cc * 2.0 * (θ[begin+1] - (1.0 + dsα)*θ[begin]) +
                  a*(1.0 - sgrid[begin])*sin(θ[begin]) )

    beam_eq_loop!(dv, θ, sgrid, cc, a, 2, lastindex(sgrid)-1)

    # Neumann b.c. at right endpoint
    dv[end] += cc * 2.0 * (θ[end-1] - θ[end]) + a*(1.0 - sgrid[end])*sin(θ[end])

    return
end

beam_eq_RN!(dv::AbstractVector{<:Real}, θ::AbstractVector{<:Real},
            sgrid::AbstractVector{<:AbstractFloat}, ds::Real, coeffs::BeamCoeffs) =
beam_eq_RN!(dv, θ, sgrid, ds/coeffs.α, (coeffs.c/ds)^2, coeffs.a)


# -------------------------------------------
#           Temporal discretization
# -------------------------------------------

"""
    beam(tmax, p, ds, sgrid, u0, [cfl, ode_alg])

Computes a solution to the beam equation with initial condition given by `θ_init`, `v_init`
and the rhs given by `p.rhs`.
"""
function beam(tmax::AbstractFloat, p::NamedTuple, ds::Real,
        sgrid::AbstractVector{<:AbstractFloat}, u0::AbstractMatrix{<:AbstractFloat};
        cfl::Real = 0.95, ode_alg = Tsit5(), kwargs...)

    function beam_f!(du, u, p, t)

        θ = @view u[:, 1]
        v = @view u[:, 2]
        dθ = @view du[:, 1]
        dv = @view du[:, 2]

        # dynamical equations and b.c.
        @. dθ = v

        rhs = p.rhs
        @. dv = rhs(sgrid, t) - p.coeffs.d * v

        p.beam_eq!(dv, θ, sgrid, ds, p.coeffs)
    end

    # Max dt to satisfy CFL condition
    cfl_dt = cfl*ds/(p.coeffs.c)

    # Solve using method of lines
    prob = ODEProblem(beam_f!, u0, (0.0, tmax), p)
    u_sol = solve(prob, ode_alg, dtmax=cfl_dt; kwargs...)

    return u_sol, sgrid
end

"""
    beam(tmax, smax, n, p, θ_init_fn, v_init_fn, bc)

Computes a solution to the beam equation up to time `tmax` on a grid of size `n` with
initial condition given by `θ_init_fn`, `v_init_fn` and the rhs f given by `p.rhs`.
"""
function beam(tmax::AbstractFloat, smax::AbstractFloat, n::Int, p::NamedTuple,
        θ_init_fn::Function, v_init_fn::Function; kwargs...)

    sgrid = range(0.0, smax, length=n)
    ds = Float64(sgrid.step)

    # Initial condition
    u0 = Matrix{Float64}(undef, n, 2)
    @. u0[:, 1] = θ_init_fn(sgrid)      # θ0
    @. u0[:, 2] = v_init_fn(sgrid)      # v0

    return beam(tmax, p, ds, sgrid, u0; kwargs...)

end



# =================================================
#                     Feedback
# =================================================

function _conv_bdry!(out::AbstractVector{<:Real}, conv_out::AbstractVector{<:Real},
        θ::AbstractVector{<:Real}, dgs::AbstractMatrix{<:Real},
        ds::Real, b::Real)

    n = length(out)
    fb = @view conv_out[n:2*n-1]

    # Compute the boundary terms
    for j in eachindex(out)

        out[j] += b * (ds * fb[j] + dgs[j, 1] * θ[end] - dgs[j, 2] * θ[begin])
    end
end


function nonlocal_feedback(tmax::AbstractFloat, p::NamedTuple, ds::Real,
        sgrid::AbstractVector{T}, u0::AbstractMatrix{T},
        dgs::AbstractMatrix{T}, ddgs::AbstractVector{T},
        θbar::AbstractVector{T}, conv_out::AbstractVector{T},
        conv_input::Tuple, multipliers::AbstractVector{T}; cfl::Real = 0.95,
        ode_alg = Tsit5(), kwargs...) where {T<:AbstractFloat}

    if iseven(length(sgrid))
        throw(ArgumentError("length of `sgrid` must be odd"))
    end
    n = length(sgrid)

    function beam_f!(du, u, p, t)

        θ = @view u[:, 1]
        v = @view u[:, 2]
        dθ = @view du[:, 1]
        dv = @view du[:, 2]

        @. dθ = v

        rhs = p.rhs
        @. dv = rhs(sgrid, t) - p.coeffs.d * v

        # - compute the feedback integral ∫g'(s-ξ) θ'(ξ)dξ by parts

        # compute the convolution ∫ g''(s-ξ) θ(ξ) dξ using a
        # Simpson's rule FFT method described in 10.1016/j.cpc.2009.10.005
        @. θbar = multipliers * θ
        convu!(conv_out, conv_input..., θbar)

        # Compute the boundary terms
        _conv_bdry!(dv, conv_out, θ, dgs, ds, p.coeffs.b)

        # ----

        # Compute acceleration dv
        beam_eq_RN!(dv, θ, sgrid, ds, p.coeffs)
    end

    # Max dt to satisfy CFL condition
    cfl_dt = cfl*ds/p.coeffs.c

    # Solve using method of lines
    prob = ODEProblem(beam_f!, u0, (0.0, tmax), p)
    u_sol = solve(prob, ode_alg, dtmax=cfl_dt; kwargs...)

    return u_sol, sgrid
end


function nonlocal_feedback_setup(smax::Real, n::Int, p::NamedTuple,
        θ_init_fn::Function, v_init_fn::Function)

    if iseven(n)
        throw(ArgumentError("grid length `n` must be odd"))
    end

    sgrid = range(0.0, smax, length=n)
    ds = Float64(sgrid.step)

    # Initial condition
    u0 = Matrix{Float64}(undef, n, 2)
    @. u0[:, 1] = θ_init_fn(sgrid)      # θ0
    @. u0[:, 2] = v_init_fn(sgrid)      # v0

    # First and second derivatives of Gaussian kernel
    λ = p.coeffs.λ

    dg(s) = -1.0/(λ^3) * s * exp(-s^2/(2.0*λ^2))
    ddg(s) = 1.0/(λ^5) * (s^2 - λ^2) * exp(-s^2/(2.0*λ^2))

    # Allocations for FFT convolutions
    sgrid2 = range(-sgrid[end], sgrid[end], length=2*n-1)
    ddgs = @. ddg(sgrid2)
    θbar = similar(sgrid)

    conv_allocs = conv_setup(ddgs, sgrid)
    conv_out = conv_allocs[1]
    conv_input = conv_allocs[2:end-2]

    simp_multipliers = Vector{Float64}(undef, n)
    @. simp_multipliers[2:2:end-1] = 4.0/3.0
    @. simp_multipliers[3:2:end-1] = 2.0/3.0
    simp_multipliers[1] = 1.0/3.0
    simp_multipliers[end] = 1.0/3.0

    # Allocations for boundary terms
    dgs = Matrix{Float64}(undef, n, 2)
    @. dgs[:, 1] = dg(sgrid - sgrid[end])
    @. dgs[:, 2] = dg(sgrid - sgrid[begin])

    return ds, sgrid, u0, dgs, ddgs, θbar, conv_out, conv_input, simp_multipliers

end


@doc raw"""
    nonlocal_feedback(tmax, smax, n, p, θ_init_fn, v_init_fn)

Computes a solution to the beam equation
```math
(\partial_t^2 - c^2 \partial_s^2) \theta + a (1-s) \cos \theta = f
```
up to time `tmax` on a grid of size `n` with initial condition given by
`θ_init_fn`, `v_init_fn` and the rhs f given by `p.rhs`.
"""
function nonlocal_feedback(tmax::Real, smax::Real, n::Int, p::NamedTuple,
        θ_init_fn::Function, v_init_fn::Function; kwargs...)

    setup = nonlocal_feedback_setup(smax, n, p, θ_init_fn, v_init_fn)

    return nonlocal_feedback(tmax, p, setup...; kwargs...)
end


# ========================================================
#                     Sparse Jacobians
# ========================================================

"""
    beam(tmax, p, ds, sgrid, u0, jac_sparsity, [cfl, ode_alg])

Computes a solution to the beam equation with initial condition given by `θ_init`, `v_init`
and the rhs given by `p.rhs`.
"""
function sparse_beam(tmax::AbstractFloat, p::NamedTuple, ds::Real,
        sgrid::AbstractVector{<:AbstractFloat}, u0::AbstractMatrix{<:AbstractFloat};
        jac_sparsity::AbstractSparseMatrix{<:Real, <:Integer},
        ode_alg = TRBDF2(linsolve = KLUFactorization()), kwargs...)

    function beam_f!(du, u, p, t)

        θ = @view u[:, 1]
        v = @view u[:, 2]
        dθ = @view du[:, 1]
        dv = @view du[:, 2]

        # dynamical equations and b.c.
        @. dθ = v

        rhs = p.rhs
        @. dv = rhs(sgrid, t) - p.coeffs.d * v

        p.beam_eq!(dv, θ, sgrid, ds, p.coeffs)
    end

    # Solve using method of lines
    sparse_f! = ODEFunction(beam_f!; jac_prototype = float.(jac_sparsity))
    prob = ODEProblem(sparse_f!, u0, (0.0, tmax), p)
    u_sol = solve(prob, ode_alg; kwargs...)

    return u_sol, sgrid
end


function sparse_nonlocal_feedback(tmax::AbstractFloat, p::NamedTuple, ds::Real,
        sgrid::AbstractVector{<:AbstractFloat}, u0::AbstractMatrix{<:AbstractFloat},
        dgs::AbstractMatrix{<:AbstractFloat}, ddgs::AbstractVector{<:AbstractFloat},
        θbar::AbstractVector{<:AbstractFloat}, conv_out::AbstractVector{<:AbstractFloat},
        conv_input::Tuple, multipliers::AbstractVector{<:AbstractFloat};
        jac_sparsity::AbstractSparseMatrix{<:Real, <:Integer}, cfl::Real = 0.95,
        ode_alg = TRBDF2(autodiff=false, linsolve = KLUFactorization()), kwargs...)

    if iseven(length(sgrid))
        throw(ArgumentError("length of `sgrid` must be odd"))
    end
    n = length(sgrid)

    function beam_f!(du, u, p, t)

        θ = @view u[:, 1]
        v = @view u[:, 2]
        dθ = @view du[:, 1]
        dv = @view du[:, 2]

        @. dθ = v

        rhs = p.rhs
        @. dv = rhs(sgrid, t) - p.coeffs.d * v

        # - compute the feedback integral ∫g'(s-ξ) θ'(ξ)dξ by parts

        # compute the convolution ∫ g''(s-ξ) θ(ξ) dξ using a
        # Simpson's rule FFT method described in 10.1016/j.cpc.2009.10.005
        @. θbar = multipliers * θ
        convu!(conv_out, conv_input..., θbar)

        # Compute the boundary terms
        _conv_bdry!(dv, conv_out, θ, dgs, ds, p.coeffs.b)

        # ----

        # Compute acceleration dv
        beam_eq_RN!(dv, θ, sgrid, ds, p.coeffs)
    end

    # Solve using method of lines
    sparse_f! = ODEFunction(beam_f!; jac_prototype = float.(jac_sparsity))
    prob = ODEProblem(sparse_f!, u0, (0.0, tmax), p)
    u_sol = solve(prob, ode_alg; kwargs...)

    return u_sol, sgrid
end

"""
    beam_jac_sparsity(tmax, p, ds, sgrid, u0, [cfl, ode_alg])

Computes the sparsity pattern
"""
function beam_jac_sparsity(p::NamedTuple, ds::Real,
        sgrid::AbstractVector{<:AbstractFloat}, u0::AbstractMatrix{<:AbstractFloat})

    function beam_f!(du, u, p, t)

        θ = @view u[:, 1]
        v = @view u[:, 2]
        dθ = @view du[:, 1]
        dv = @view du[:, 2]

        # dynamical equations and b.c.
        @. dθ = v

        rhs = p.rhs
        @. dv = rhs(sgrid, t) - p.coeffs.d * v

        p.beam_eq!(dv, θ, sgrid, ds, p.coeffs)
    end

    # Compute the sparsity pattern
    du0 = copy(u0)
    jac_sparsity = Symbolics.jacobian_sparsity(
                                               (du, u) -> beam_f!(du, u, p, 0.0),
                                               du0, u0)
    return jac_sparsity
end

# end module Beam
end
