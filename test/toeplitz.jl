
v  = collect(1.0:10.0)
op = GriddedKernelMatrices.SymToeplitz(v)
M  = [v[abs(j-k)+1] for j in 1:10, k in 1:10]
x  = collect(11.0:20.0)

@test M*x ≈ op*x

