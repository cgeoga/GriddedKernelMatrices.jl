
grid1 = range(0.0, 1.0, length=64)
pts   = vec(SVector{2,Float64}.(Iterators.product(grid1, grid1)))

C_exact = [exp(-norm(x-y)^2) for x in pts[1:100], y in pts[101:300]]
C_fast  = lattice_kernel_matrix(pts[1:100], pts[101:300], h->exp(-norm(h)^2))

(U, S, V) = psvd(C_fast; rtol=1e-13)
@test isapprox(U*Diagonal(S)*V', C_exact; rtol=1e-11)

