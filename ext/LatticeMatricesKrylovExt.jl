
module LatticeMatricesKrylovExt

  using LatticeMatrices, Krylov
  using LatticeMatrices.LinearAlgebra
  import LatticeMatrices: SymToeplitz, SymBTTB, MaskedSymToeplitz, MaskedSymBTTB

  function LatticeMatrices._solve!(buf, M::SymToeplitz, v)
    buf .= cg(M, v; M=M.pre)[1]
    buf
  end

  function LatticeMatrices._solve!(buf, M::SymBTTB, v)
    buf .= cg(M, v; M=M.pre)[1]
    buf
  end

  function LatticeMatrices._solve!(buf, M::MaskedSymToeplitz, v)
    buf .= cg(M, v; M=M.pre)[1]
    buf
  end

  function LatticeMatrices._solve!(buf, M::MaskedSymBTTB, v)
    buf .= cg(M, v; M=M.pre)[1]
    buf
  end

end

