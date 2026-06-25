
# Make a complete grid of points, and then throw some away for demonstration.
grid1d = range(0.0, 1.0, length=32)
pts    = vec(SVector{2,Float64}.(Iterators.product(grid1d, grid1d)))
pts    = pts[setdiff(1:length(pts), sort(unique(rand(1:length(pts), 50))))]

# define a kernel function. Note that this kernel 
const L = @SMatrix [1.1 0.0
                    2.1 3.1]
aniso_kernel(h::SVector{2,Float64}) = exp(-norm(L*h))

# Make the reference and implicit matrices
M_exact = [kernel(x-y) for x in pts, y in pts]
M_fast  = lattice_kernel_matrix(pts, kernel)

v  = collect(1.0:length(pts))
v2 = hcat(v, v, v)
@test M_exact*v  ≈ M_fast*v
@test M_exact*v2 ≈ M_fast*v2

