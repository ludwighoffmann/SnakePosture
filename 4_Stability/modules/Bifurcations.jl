module Bifurcations

using ..Beam
using OrdinaryDiffEq
using Roots
using CSV

function bifurcation_diagram_one(tmax=1000.0, d=0.5; stab_tol=1e-4, num_points=500)#2000)

    """
    Function to create the data for the bifurcation diagram in Fig. 4a. 
    """

    # ----- Setup -----

    # Settings
    smax = 1.0
    n = 2^8 + 1
    tol = 1.0e-9
    amin = 6.0
    amax = 20.0

    θ_amplitude = 0.0001
    α = 1e-6        # Robin b.c.

    # PDE coefficients
    c = 1.0
    a = 1.0
    coeffs = Beam.BeamCoeffs(a=a, b=0.0, c=c, d=d, α=α)

    # Discretization
    sgrid = range(0.0, smax, length=n)
    ds = Float64(sgrid.step)

    # Initial condition
    θ_init(s) = -θ_amplitude/(2/π-α) * ( α * cos(π*s) + 2.0/π * sin(pi/2 * s) )
    v_init(s) = 0.0

    # rhs and b.c.
    rhs(s, t) = 0.0
    eq! = Beam.beam_eq_RN!
    p = (coeffs=coeffs, rhs=rhs, beam_eq! = Beam.beam_eq_RN!)

    # Setup
    setup = Beam.nonlocal_feedback_setup(smax, n, p, θ_init, v_init)
    u0 = setup[3]

    sp = float.(Beam.beam_jac_sparsity(p, ds, sgrid, u0))

    # ----- Bifurcation solver -----

    # Use this function to solve the equation in the case lambda = 0. Return the max angle.
    function stability_f(a, b)
        p.coeffs.a = a
        p.coeffs.c = sqrt(c^2 + b)

        beam_sol, _ = Beam.sparse_beam(tmax, p, ds, sgrid, u0;
                                       jac_sparsity=sp,
                                       abstol=tol, reltol=tol, save_everystep=false)
                                       
                                       
        θs = beam_sol[:, 1, end]

        return maximum(abs, θs) - stab_tol
    end
    
    # Use this function instead to solve the equation in the case lambda != 0. Return the max angle.
    function nl_stability_f(a, b, λ)
        p.coeffs.a = a
        p.coeffs.c = c
        p.coeffs.b = b
        p.coeffs.λ = λ
        
        setup = Beam.nonlocal_feedback_setup(smax, n, p, θ_init, v_init)

        beam_sol, _ = Beam.sparse_nonlocal_feedback(tmax, p, setup...;
                                                        jac_sparsity=sp, abstol=tol, reltol=tol,
                                                        save_everystep=false)
        θs = beam_sol[:, 1, end]

        return maximum(abs, θs) - stab_tol
    end
    
    # ----- Run -----
    
    #Range we consider for alpha
    as = LinRange(amin, amax, num_points)
    
    #Fix a value for the active moment. In the non-local case we divide by sqrt(2*π) to account for the normalization of the kernel
    b = 0.5
    b_nonlocal = b/sqrt(2*π)
    
    #Save the data for the bifurcation diagram for six different values for lambda
    @time stabs0 = @. stability_f(as, b) + stab_tol
    CSV.write("Bifurcation_Diagram_Data/stabs0.csv", (data = stabs0, ))
    @time stabs1 = @. nl_stability_f(as, b_nonlocal, 0.1) + stab_tol
    CSV.write("Bifurcation_Diagram_Data/stabs1.csv", (data = stabs1, ))
    @time stabs2 = @. nl_stability_f(as, b_nonlocal, 0.25) + stab_tol
    CSV.write("Bifurcation_Diagram_Data/stabs2.csv", (data = stabs2, ))
    @time stabs3 = @. nl_stability_f(as, b_nonlocal, 0.5) + stab_tol
    CSV.write("Bifurcation_Diagram_Data/stabs3.csv", (data = stabs3, ))
    @time stabs4 = @. nl_stability_f(as, b_nonlocal, 1) + stab_tol
    CSV.write("Bifurcation_Diagram_Data/stabs4.csv", (data = stabs4, ))
    @time stabs5 = @. nl_stability_f(as, b_nonlocal, 4) + stab_tol
    CSV.write("Bifurcation_Diagram_Data/stabs5.csv", (data = stabs5, ))

    return
end

function bifurcation_diagram_two(tmax=50.0, d=0.5; stab_tol=1e-4, num_points=1000)

    """
    Function to create the data used for the critical curves in 4b.
    Same way of finding bifurcation solution as previous function, see there for more comments.
    """

    # ----- Setup -----

    # Settings
    smax = 1.0
    n = 2^8 + 1
    tol = 1.0e-9
    amin = 0.1
    amax = 16.0

    θ_amplitude = 0.0001
    α = 1e-6        # Robin b.c.

    # PDE coefficients
    c = 1.0
    a = 1.0
    coeffs = Beam.BeamCoeffs(a=a, b=0.0, c=c, d=d, α=α)

    # Discretization
    sgrid = range(0.0, smax, length=n)
    ds = Float64(sgrid.step)

    # Initial condition
    θ_init(s) = -θ_amplitude/(2/π-α) * ( α * cos(π*s) + 2.0/π * sin(pi/2 * s) )
    v_init(s) = 0.0

    # rhs and b.c.
    rhs(s, t) = 0.0
    eq! = Beam.beam_eq_RN!
    p = (coeffs=coeffs, rhs=rhs, beam_eq! = Beam.beam_eq_RN!)

    # Setup
    setup = Beam.nonlocal_feedback_setup(smax, n, p, θ_init, v_init)
    u0 = setup[3]
    sp = float.(Beam.beam_jac_sparsity(p, ds, sgrid, u0))

    # ----- Bifurcation solver -----

    function stability_f(a, b)
        p.coeffs.a = a
        p.coeffs.c = sqrt(c^2 + b)

        beam_sol, _ = Beam.sparse_beam(tmax, p, ds, sgrid, u0;
                                       jac_sparsity=sp,
                                       abstol=tol, reltol=tol, save_everystep=false)
                                       
                                       
        θs = beam_sol[:, 1, end]
        return maximum(abs, θs) - stab_tol
    end

    function nl_stability_f(a, b, λ)
        p.coeffs.a = a
        p.coeffs.c = c
        p.coeffs.b = b
        p.coeffs.λ = λ
        
        setup = Beam.nonlocal_feedback_setup(smax, n, p, θ_init, v_init)

        beam_sol, _ = Beam.sparse_nonlocal_feedback(tmax, p, setup...;
                                                        jac_sparsity=sp, abstol=tol, reltol=tol,
                                                        save_everystep=false)
        θs = beam_sol[:, 1, end]
        return maximum(abs, θs) - stab_tol
    end
    
    # ----- Run -----

    as = LinRange(amin, amax, num_points)
    
    #Find bifurcation curve for these values of b and lambda
    bval_array = ["b_25","b_50","b_100"]
    lambda_array = [3,2,1,0.9,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.1,0.05]
    
    count = 1
    
    for bval in [0.25,0.5,1.0]
        b_local = bval
        b_nonlocal = bval/sqrt(2*π)
        
        iter = 0
        
        for lambdaval in lambda_array
            
            println(lambdaval)
            
            @time stabs3 = @. nl_stability_f(as, b_nonlocal, lambdaval) + stab_tol
            CSV.write("Crit_Curves_Data/" + bval_array[count] * "/stabs_$(iter).csv", (data = stabs3, ))
            
            iter += 1
        end
        
        count += 1
        
    end
    
    return
    
end


function stability_diagram_nl(num_points=11, tmax=2000.0, d=0.5; stab_tol=1e-4)

    """
    Function to get data for the three-dimensional stability diagram in Fig. 4c. 
    Solve stability problem as before. irst, save spatial data for three curves.
    Then solve the stability for different values of lambda.
    """

    # ----- Setup -----

    # Settings
    smax = 1.0
    n = 2^8 + 1
    tol = 1.0e-9
    bmin = -1.0
    bmax = 1.0
    amin = 0.0
    amax = 25.0

    θ_amplitude = 0.0001
    α = 1e-6        # Robin b.c.

    # PDE coefficients
    c = 1.0
    a = 1.0
    coeffs = Beam.BeamCoeffs(a=a, b=0.0, c=c, d=d, α=α)

    # Discretization
    sgrid = range(0.0, smax, length=n)
    ds = Float64(sgrid.step)

    # Initial condition
    θ_init(s) = -θ_amplitude/(2/π-α) * ( α * cos(π*s) + 2.0/π * sin(pi/2 * s) )
    v_init(s) = 0.0
    u0 = Matrix{Float64}(undef, n, 2)
    @. u0[:, 1] = θ_init(sgrid)      # θ0
    @. u0[:, 2] = v_init(sgrid)      # v0

    # rhs and b.c.
    rhs(s, t) = 0.0
    eq! = Beam.beam_eq_RN!
    p = (coeffs=coeffs, rhs=rhs, beam_eq! = Beam.beam_eq_RN!)

    sp = float.(Beam.beam_jac_sparsity(p, ds, sgrid, u0))

    # ----- Bifurcation solver -----
    
    
    println("Saving curves for three examples...")

    function nl_stability_f_print_angle(a, b, λ)
        p.coeffs.a = a
        p.coeffs.c = c
        p.coeffs.b = b
        p.coeffs.λ = λ
        
        setup = Beam.nonlocal_feedback_setup(smax, n, p, θ_init, v_init)

        beam_sol, _ = Beam.sparse_nonlocal_feedback(tmax, p, setup...;
                                                        jac_sparsity=sp, abstol=tol, reltol=tol,
                                                        save_everystep=false)
    
        θs = beam_sol[:, 1, end]
        
        return θs
    end

    th = nl_stability_f_print_angle(5., 0.5/sqrt(2*π), 0.25)
    CSV.write("ExampleCurves/curve_1.csv", (data = th, ))
    th = nl_stability_f_print_angle(12., 0.5/sqrt(2*π), 0.25)
    CSV.write("ExampleCurves/curve_2.csv", (data = th, ))
    th = nl_stability_f_print_angle(15., 0.1/sqrt(2*π), 0.1)
    CSV.write("ExampleCurves/curve_3.csv", (data = th, ))

    function nl_stability_f(a, b, λ)
        p.coeffs.a = a
        p.coeffs.c = c
        p.coeffs.b = b
        p.coeffs.λ = λ
        
        setup = Beam.nonlocal_feedback_setup(smax, n, p, θ_init, v_init)

        beam_sol, _ = Beam.sparse_nonlocal_feedback(tmax, p, setup...;
                                                        jac_sparsity=sp, abstol=tol, reltol=tol,
                                                        save_everystep=false)
        
        θs = beam_sol[:, 1, end]
        
        return maximum(abs, θs) - stab_tol
    end
    
    # ----- Run -----
    
    println("---------")
    println("Stability for different values of lambda.")
    println("Values of b:")
    for (i, b) in enumerate(bs)
        println(b)
    end
    println("---")
    
    
    bs = LinRange(bmin, bmax, num_points)
    as = zeros(length(bs))
    
    
    for lambda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        a_guess = find_zero( x-> nl_stability_f(x, 0.0, lambda), (amin, amax))
        print("lambda = ")
        println(lambda)
        println("...")
        for (i, b) in enumerate(bs)
            b_nonlocal = b/sqrt(2*π)
            if i == 6
               a_root =  7.8374114936913815
            else
                a_root = find_zero( x-> nl_stability_f(x, b_nonlocal, lambda), 1.01*a_guess)
            end
            as[i] = a_root
            println(a_root)
            a_guess = a_root
        end
        println("----")
    end

    return
end

function stability_diagram_nl_exp(num_points=11, tmax=2000.0, d=0.5; stab_tol=1e-4)

    """
    Function to get data for the three-dimensional stability diagram in Fig. S9. 
    Same as previous function but for different range of b and lambda.
    """

    # ----- Setup -----

    # Settings
    smax = 1.0
    n = 2^8 + 1
    tol = 1.0e-9
    bmin = 8.0
    bmax = 12.0
    amin = 0
    amax = 200.0

    θ_amplitude = 0.0001
    α = 1e-6        # Robin b.c.

    # PDE coefficients
    c = 1.0
    a = 1.0
    coeffs = Beam.BeamCoeffs(a=a, b=0.0, c=c, d=d, α=α)

    # Discretization
    sgrid = range(0.0, smax, length=n)
    ds = Float64(sgrid.step)

    # Initial condition
    θ_init(s) = -θ_amplitude/(2/π-α) * ( α * cos(π*s) + 2.0/π * sin(pi/2 * s) )
    v_init(s) = 0.0
    u0 = Matrix{Float64}(undef, n, 2)
    @. u0[:, 1] = θ_init(sgrid)      # θ0
    @. u0[:, 2] = v_init(sgrid)      # v0

    # rhs and b.c.
    rhs(s, t) = 0.0
    eq! = Beam.beam_eq_RN!
    p = (coeffs=coeffs, rhs=rhs, beam_eq! = Beam.beam_eq_RN!)

    #beam_args = (tmax, p, ds, sgrid, u0)
    sp = float.(Beam.beam_jac_sparsity(p, ds, sgrid, u0))

    # ----- Bifurcation solver -----
    
    function stability_f(a, b)
        p.coeffs.a = a
        p.coeffs.c = sqrt(c^2 + b)

        beam_sol, _ = Beam.sparse_beam(tmax, p, ds, sgrid, u0;
                                       jac_sparsity=sp,
                                       abstol=tol, reltol=tol, save_everystep=false)
        θs = beam_sol[:, 1, end]

        return maximum(abs, θs) - stab_tol
    end

    function nl_stability_f(a, b, λ)
        p.coeffs.a = a
        p.coeffs.c = c
        p.coeffs.b = b
        p.coeffs.λ = λ
        
        setup = Beam.nonlocal_feedback_setup(smax, n, p, θ_init, v_init)

        beam_sol, _ = Beam.sparse_nonlocal_feedback(tmax, p, setup...;
                                                        jac_sparsity=sp, abstol=tol, reltol=tol,
                                                        save_everystep=false)
        θs = beam_sol[:, 1, end]
        return maximum(abs, θs) - stab_tol
    end

    bs = LinRange(bmin, bmax, num_points)
    as = zeros(length(bs))
    
    # ----- Run -----
    
    println("Stability for different values of lambda.")
    println("Values of b:")
    for (i, b) in enumerate(bs)
        println(b)
    end
    println("---")
    
    
    println("lambda = 0")
    println("...")
    
    a_guess = 7.84*(1+bmin)
    
    for (i, b) in enumerate(bs)
        a_root = find_zero( x-> stability_f(x, b), 1.05*a_guess)
        as[i] = a_root
        println(a_root)
        a_guess = a_root
    end
    
    
    a_guess = 7.84*(1+bmin-1) #modifying a_guess can make finding solutions faster/possible
    for lambda in [0.04, 0.08, 0.12]
        print("lambda = ")
        println(lambda)
        println("...")
        for (i, b) in enumerate(bs)
            b_nonlocal = b/sqrt(2*π)
            a_root = find_zero( x-> nl_stability_f(x, b_nonlocal, lambda), 1.03*a_guess)
            as[i] = a_root
            println(a_root)
            a_guess = a_root
        end
        println("----")
        a_guess -= 0.5
    end

    return
end


# end module
end
