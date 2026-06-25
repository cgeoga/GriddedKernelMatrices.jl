
module LatticeMatricesKrylovExt

  using LatticeMatrices, Krylov
  using LatticeMatrices.LinearAlgebra
  import LatticeMatrices: SymToeplitz, SymBTTB, MaskedSymToeplitz, MaskedSymBTTB

  for T in (SymToeplitz, SymBTTB, MaskedSymToeplitz, MaskedSymBTTB)
    @eval begin

    function LatticeMatrices._solve!(buf, M::$T, v)
      work = CgWorkspace(M, v)
      cg!(work, M, v; M=M.pre)
      buf .= work.x
      buf
    end

    end
  end

end

