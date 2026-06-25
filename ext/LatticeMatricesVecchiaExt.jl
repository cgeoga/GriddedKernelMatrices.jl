
module LatticeMatricesVecchiaExt

  using LatticeMatrices, Vecchia
  using LatticeMatrices.LinearAlgebra
  using LatticeMatrices.StaticArraysCore
  import LatticeMatrices: VecchiaPreconditioner, gen_preconditioner

  function LatticeMatrices.gen_preconditioner(pre::VecchiaPreconditioner, pts, kernel)
    _kernel = (x, y, p) -> kernel(x-y)
    appx    = VecchiaApproximation(pts, _kernel; conditioning=KNNConditioning(pre.k))
    rchol_preconditioner(appx, Float64[])
  end

  function LatticeMatrices.gen_preconditioner(pre::VecchiaPreconditioner, 
                                              pts::Vector{Float64}, kernel)
    _pts = [SVector{1,Float64}(x) for x in pts]
    LatticeMatrices.gen_preconditioner(pre, _pts, kernel)
  end

end

