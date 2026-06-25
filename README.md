
# LatticeMatrices.jl

This package offers very simple functionality for accelerating kernel matrices
for which (1) the kernel is shift-invariant, so that it accepts a distance/lag
argument, and (2) the locations at which the kernel is evaluated are on a gappy
lattice in one or two dimensions. While there are existing packages in the
ecosystem for Toeplitz-like matrix-vector product acceleration (like
[ToeplitzMatrices.jl](https://github.com/JuliaLinearAlgebra/ToeplitzMatrices.jl)),
this package is focused on applications where you have a kernel matrix
corresponding to a (potentially gappy) lattice of measurement locations. As well
as handling those gaps and still exploiting FFT acceleration, this package also
offers significantly better preconditioners via the optional
[Vecchia.jl](https://github.com/cgeoga/Vecchia.jl) extension. Here is a quick
demonstration:
```julia
using StaticArrays, LatticeMatrices
using  Krylov, Vecchia # optional extensions if you want fast \ as well as *.

# Example points on a lattice with some gaps (excuse the ugly gap generation).
grid1d = range(0.0, 1.0, length=64)
pts    = vec(SVector{2,Float64}.(Iterators.product(grid1d, grid1d)))
pts    = pts[setdiff(1:length(pts), sort(unique(rand(1:length(pts), 50))))]

# This object is the equivalent of [exp(-norm(x-y)) for x in pts, y in pts].
# If your kernel is smooth away from the origin and you want efficient linear
# solves, specify a Vecchia preconditioner like so.
M_fast = lattice_kernel_matrix(pts, h->exp(-norm(h)), 
                               VecchiaPreconditioner(k=40))

# You get a nice quasilinear matvec that is exact:
v = Float64.(eachindex(pts))
@show isapprox(M_ref*v, M_fast*v)

# example solve:
isapprox(M_ref\v, M_fast\v, rtol=1e-8) # < 10 iterations, so < 20 FFTs.
```
There is analogous functionality for one-dimensional lattice data. See the tests
for more demonstrations.

