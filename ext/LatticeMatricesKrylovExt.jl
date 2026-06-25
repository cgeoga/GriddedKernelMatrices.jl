
module LatticeMatricesKrylovExt

  using LatticeMatrices, Krylov
  using LatticeMatrices.LinearAlgebra
  import LatticeMatrices: SymToeplitz, SymBTTB, MaskedSymToeplitz, MaskedSymBTTB

  for T in (SymToeplitz, SymBTTB, MaskedSymToeplitz, MaskedSymBTTB)
    @eval begin

    function LatticeMatrices._solve!(buf::AbstractVector, M::$T, v::AbstractVector)
      work = CgWorkspace(M, v)
      LatticeMatrices._solve!(buf, M, v, work)
    end

    function LatticeMatrices._solve!(buf::AbstractVector, M::$T, v::AbstractVector, work)
      cg!(work, M, v; M=M.pre)
      buf .= work.x
      buf
    end

    function LatticeMatrices._solve!(buf::AbstractMatrix, M::$T, v::AbstractMatrix)
      work = CgWorkspace(M, view(v, :, 1))
      for j in 1:size(v, 2)
        bufj  = view(buf, :, j)
        vj    = view(v,   :, j)
        cg!(work, M, v; M=M.pre)
        bufj .= work.x
      end
      buf
    end

    end
  end

end

