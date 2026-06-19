
module LatticeMatricesVecchiaExt

  using LatticeMatrices, Vecchia
  using LatticeMatrices.LinearAlgebra
  import LatticeMatrices: VecchiaPreconditioner, gen_preconditioner

  function LatticeMatrices.gen_preconditioner(pre::VecchiaPreconditioner, pts, kernel)
    _kernel = (x, y, p) -> kernel(x-y)
    appx    = VecchiaApproximation(pts, _kernel; conditioning=KNNConditioning(pre.k))
    rchol_preconditioner(appx, Float64[])
  end

end

