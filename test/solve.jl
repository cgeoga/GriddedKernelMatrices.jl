
using Krylov, Vecchia # optional extensions if you want fast \ as well as *.

# Make points on a gappy lattice.
grid1d = range(0.0, 1.0, length=64)
pts    = vec(SVector{2,Float64}.(Iterators.product(grid1d, grid1d)))
pts    = pts[setdiff(1:length(pts), sort(unique(rand(1:length(pts), 50))))]

# Create the exponential covariance matrix for them.
M_ref  = [exp(-norm(x-y)) for x in pts, y in pts]

# Create the fast implicit operator for them
M_fast = lattice_kernel_matrix(pts, h->exp(-norm(h)), 
                               VecchiaPreconditioner(k=40))

v = Float64.(eachindex(pts))
@test isapprox(M_fast\v, M_ref\v, rtol=1e-7)

