
g1  = range(0.0, 1.0, length=32)
g2  = range(0.0, 1.0, length=42)
pts = vec(SVector{2,Float64}.(Iterators.product(g1, g2)))

M_ref  = [exp(-norm(x-y)) for x in pts, y in pts]
M_fast = lattice_kernel_matrix(pts, h->exp(-norm(h)))

v = collect(1.0:length(pts))
@test M_ref*v ≈ M_fast*v

