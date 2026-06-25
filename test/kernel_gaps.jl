
#
# 1D:
#

# gappy lattice in 1D:
pts = [1.0, 3, 6, 7, 8, 10]

# Simple exponential decay kernel
kernel(dt) = exp(-0.5 * abs(dt))

# Make the reference and implicit matrices
M_fast  = [kernel(abs(x-y)) for x in pts, y in pts]
M_exact = lattice_kernel_matrix(pts, kernel)

v  = collect(1.0:length(pts))
v2 = hcat(v, v, v)
@test M_exact*v  ≈ M_fast*v
@test M_exact*v2 ≈ M_fast*v2


#
# 2D:
#

# Make a complete grid of points, and then throw some away for demonstration.
grid1d = range(0.0, 1.0, length=32)
pts    = vec(SVector{2,Float64}.(Iterators.product(grid1d, grid1d)))
pts    = pts[setdiff(1:length(pts), sort(unique(rand(1:length(pts), 50))))]

# define a kernel function. Note that this kernel 
kernel(h::SVector{2,Float64}) = exp(-norm(h))

# Make the reference and implicit matrices
M_exact = [kernel(x-y) for x in pts, y in pts]
M_fast  = lattice_kernel_matrix(pts, kernel)

v  = collect(1.0:length(pts))
v2 = hcat(v, v, v)
@test M_exact*v  ≈ M_fast*v
@test M_exact*v2 ≈ M_fast*v2

