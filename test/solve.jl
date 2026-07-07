
using Krylov, Vecchia # optional extensions if you want fast \ as well as *.

for dim in (1,2) # hits Toeplitz and block Toeplitz.

  local pts = if dim == 1
    _pts = collect(range(0.0, 1.0, length=500))
    _pts[setdiff(1:length(_pts), sort(unique(rand(1:length(_pts), 50))))]
  else
    local grid1d = range(0.0, 1.0, length=64)
    _pts   = vec(SVector{2,Float64}.(Iterators.product(grid1d, grid1d)))
    _pts[setdiff(1:length(_pts), sort(unique(rand(1:length(_pts), 50))))]
  end

  # Create the exponential covariance matrix for them.
  local M_ref  = [exp(-norm(x-y)) for x in pts, y in pts]

  # Create the fast implicit operator for them
  local M_fast = lattice_kernel_matrix(pts, h->exp(-norm(h)), 
                                       VecchiaPreconditioner(k=40))

  local v = Float64.(eachindex(pts))
  @test isapprox(M_fast\v, M_ref\v, rtol=1e-7)

  # Matrix solve:
  local vm = [log1p(i+j) for i in 1:length(pts), j in 1:10]
  @test isapprox(M_fast\vm, M_ref\vm, rtol=1e-7)

end

