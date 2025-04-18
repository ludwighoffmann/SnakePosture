module Convolution
# This module is a slight modification of the DSP FFT convolution code
#   https://github.com/JuliaDSP/DSP.jl/blob/master/src/dspbase.jl
# in order to reduce repeated allocations when performing multiple
# convolutions with same-size vectors.

using FFTW
using LinearAlgebra
using DSP: nextfastfft
export conv!

# ========================================================
#                     FFTW convolution
# ========================================================


"""
    _zeropad!(padded::AbstractVector, u::AbstractVector)

Copies `u` to the start of `padded` and sets the rest of `padded` to zero.
"""
@inline function _zeropad!(
        padded::AbstractVector,
        u::AbstractVector,
    )
    datasize = length(u)
    copyto!(padded, 1, u, 1, datasize)
    padded[1 + datasize : end] .= 0

    padded
end

"""
    _zeropad(u, padded_size)

Creates and returns a new arrray of size `padded_size`, copies `u`
to the start of the vector and sets the rest of the vector to zero.
"""
function _zeropad(u, padded_size)
    padded = similar(u, padded_size)
    _zeropad!(padded, u)
end


"""
    _apply_fftplan(P, x, y)

Applies the pre-planned FFT P to the vector x and saves the result to y.
This is an alternative to y = P * x without allocations.
"""
function _apply_fftplan!(P, x, y)
    FFTW.assert_applicable(P, x)
    FFTW.unsafe_execute!(P, x, y)
end


function _conv_kern_fft!(
        # output
        out,
        # preallocated
        padded, P, IP, uf, vf, raw_out,
        # input
        u::AbstractArray{T, N},
        v::AbstractArray{T, N}) where {T<:Real, N}

    # Fourier transform of u
    _zeropad!(padded, u)
    _apply_fftplan!(P, padded, uf) # equivalent to `uf = P * padded`
    # Fourier transform of v
    _zeropad!(padded, v)
    # Compute convolution in Fourier space
    _apply_fftplan!(P, padded, vf) # equivalent to `vf = P * padded`
    uf .*= vf

    # Inverse Fourier transform
    # the following lines are equivalent to `raw_out = IP * uf`
    _apply_fftplan!(IP.p, uf, raw_out)
    LinearAlgebra.rmul!(raw_out, IP.scale)

    copyto!(out,
            CartesianIndices(out),
            raw_out,
            CartesianIndices(UnitRange.(1, length(out)))
           )
end



"""
    conv!(out, padded, P, IP, uf, vf, raw_out, u, v)

Computes the convolution of the vectors `u` and `v` using FFT convolution
without allocations. The result is saved to `out`. The arrays should be
allocated beforehand with `conv_setup`.
Example usage:
```
    u, v = rand(10), rand(10)
    input = conv_setup(u, v)
    out = input[1]
    conv!(input...)
```
which is equivalent to `out = DSP.conv(u, v)`, but the allocated
temporary arrays can be reused.
"""
function conv!(
        out::Vector{<:Real},
        padded::Vector{<:Real}, P, IP, uf, vf, raw_out,
        u::Vector{<:Real}, v::Vector{<:Real}
    )

    lu = length(u)
    lv = length(v)
    outsize = lu + lv - 1
    if length(out) != outsize
        throw(ArgumentError("output size $(size(out)) must equal outsize $(outsize)"))
    end

    _conv_kern_fft!(out, padded, P, IP, uf, vf, raw_out, u, v)

    out
end


"""
    convu!(out, padded, P, IP, uf, vf, raw_out, v)

Computes the convolution of the vectors `v` and the vector with Fourier transform `uf`
without allocations. The result is saved to `out`. The arrays should be allocated
beforehand with `conv_setup`.
Example usage:
```
    u = rand(10), rand(10)
    input = conv_setup(u, v)
    out = input[1]
    w = rand(10)
    convu!(out, input[2:end-2]..., w)
```
which is equivalent to `out = DSP.conv(u, w)`, but the allocated
temporary arrays can be reused.
"""
function convu!(
        # output
        out,
        # preallocated
        padded, P, IP, uf, vf, raw_out,
        # input
        v::AbstractArray{<:Real})

    # Fourier transform of v
    _zeropad!(padded, v)
    _apply_fftplan!(P, padded, vf) # equivalent to `vf = P * padded`

    # Compute convolution in Fourier space
    vf .*= uf

    # Inverse Fourier transform
    # the following lines are equivalent to `raw_out = IP * vf`
    _apply_fftplan!(IP.p, vf, raw_out)
    LinearAlgebra.rmul!(raw_out, IP.scale)

    copyto!(out,
            CartesianIndices(out),
            raw_out,
            CartesianIndices(UnitRange.(1, length(out)))
           )

    return out

end

"""
    convu!(out, padded, P, IP, uf, vf, raw_out, u, v)

Convenience function. Example usage:
```
    u = rand(10), rand(10)
    input = conv_setup(u, v)
    out = input[1]
    convu!(input)
```
which is equivalent to `out = DSP.conv(u, w)`, but the allocated
temporary arrays can be reused.
"""
function convu!(
        out::Vector{<:Real},
        padded::Vector{<:Real}, P, IP, uf, vf, raw_out,
        u::Vector{<:Real}, v::Vector{<:Real}
    )

    convu!(out, padded, P, IP, uf, vf, raw_out, v)
end


"""
    conv_setup(u, v)

Computes the Fourier transform `uf` of the array `u` and allocates
the necessary arrays to compute the convolution of `u` and `v` with
`conv!` or `convu!`. Note that `vf` is not the Fourier transform of `v`
but is vector of the correct length.
See [`conv!`](@ref) for usage.
"""
function conv_setup(u::AbstractVector, v::AbstractVector)

    lu = length(u)
    lv = length(v)
    outsize = lu + lv - 1

    out = similar(u, outsize)

    nffts = nextfastfft(outsize)

    padded = _zeropad(u, nffts)
    P = plan_rfft(padded)
    uf = P * padded
    vf = similar(uf)
    IP = plan_irfft(uf, nffts)
    raw_out = IP.p * uf

    return out, padded, P, IP, uf, vf, raw_out, u, v

end

## modify vbar so that conv(u, v) * dx/3 computes a simpson's rule
## approximation of the convolution integral
#function simpson_conv_setup!(vbar::AbstractVector, v::AbstractVector)
#end


# ---------------------------------------------------------------------------------------

# ===== Simpson's rule =====

function simpson(y, h=1.0)

    if iseven(length(y))
        throw(ArgumentError("length of `y` must be odd"))
    end

    sm = y[1] + y[end]
    sm += 4.0 * sum(y[2:2:end])
    sm += 2.0 * sum(y[3:2:end-1])

    return sm/3 * h
end

function simpson_convolution(f::Function, v::AbstractVector, grid::StepRangeLen)

    n = length(grid)
    if length(v) != n
        throw(ArgumentError("`v` and `grid` must have equal length"))
    end

    fv = similar(grid)
    result = similar(grid)
    dx = Float64(grid.step)

    for i in 1:n
        x = grid[i]
        @. fv = f(x-grid) * v
        result[i] = simpson(fv, dx)
    end

    return result
end

function fftrect_convolution(f::Function, v::AbstractVector, grid::StepRangeLen)

    n = length(grid)
    if length(v) != n
        throw(ArgumentError("`v` and `grid` must have equal length"))
    end

    dx = Float64(grid.step)
    grid2 = range(-grid[end], grid[end], length=2*n-1)
    fs = @. f(grid2)
    dummy = zeros(n)

    conv_allocs = MyConv.conv_setup(fs, dummy)
    conv_out = conv_allocs[1]
    conv_input = conv_allocs[2:end-2]

    MyConv.convu!(conv_out, conv_input..., v)
    result = conv_out[n:2*n-1] .* dx

    return result
end

function fftsimpson_convolution(f::Function, v::AbstractVector, grid::StepRangeLen)

    n = length(grid)
    if length(v) != n
        throw(ArgumentError("`v` and `grid` must have equal length"))
    end

    dx = Float64(grid.step)
    grid2 = range(-grid[end], grid[end], length=2*n-1)
    fs = @. f(grid2)

    vbar = similar(v)
    vcopy = similar(v)

    @. vcopy = @view v[:]
    #vbar[2:2:end-1] = 4.0 .* @view v[2:2:end-1]
    #vbar[3:2:end-1] = 2.0 .* @view v[3:2:end-1]
    @. vbar[2:2:end-1] = 4.0 * @view vcopy[2:2:end-1]
    @. vbar[3:2:end-1] = 2.0 * @view vcopy[3:2:end-1]
    vbar[1] = v[1]
    vbar[end] = v[end]

    conv_allocs = MyConv.conv_setup(fs, v)
    conv_out = conv_allocs[1]
    conv_input = conv_allocs[2:end-2]

    MyConv.convu!(conv_out, conv_input..., vbar)
    result = conv_out[n:2*n-1] .* dx/3.0

    return result
end



# ---------------------------------------------------------------------------------------

# Just for computing sparsity pattern
function DirectLinearConvolution!(out, u, v)

    for i in eachindex(out)
        for j in eachindex(u)
            if 1 <= i - j + 1 <= length(v)
                out[i] += u[j] * v[i-j+1]
            end
        end
    end

    return out
end

function DirectLinearConvolution(u, v)

    out = zeros(eltype(v), length(u)+length(v)-1)
    for i in eachindex(out)
        for j in eachindex(u)
            if 1 <= i - j + 1 <= length(v)
                out[i] += u[j] * v[i-j+1]
            end
        end
    end

    return out
end


# end module MyConv
end
