
module GriddedKernelMatricesKrylovExt

  using GriddedKernelMatrices, Krylov
  using GriddedKernelMatrices.LinearAlgebra
  import GriddedKernelMatrices: SymToeplitz, SymBTTB, MaskedSymToeplitz, MaskedSymBTTB

  for T in (SymToeplitz, SymBTTB, MaskedSymToeplitz, MaskedSymBTTB)
    @eval begin

    function GriddedKernelMatrices._solve!(buf::AbstractVector, M::$T, v::AbstractVector)
      work = CgWorkspace(M, v)
      GriddedKernelMatrices._solve!(buf, M, v, work)
    end

    function GriddedKernelMatrices._solve!(buf::AbstractVector, M::$T, v::AbstractVector, work)
      cg!(work, M, v; M=M.pre)
      buf .= work.x
      buf
    end

    function GriddedKernelMatrices._solve!(buf::AbstractMatrix, M::$T, v::AbstractMatrix)
      work = CgWorkspace(M, view(v, :, 1))
      for j in 1:size(v, 2)
        bufj  = view(buf, :, j)
        vj    = view(v,   :, j)
        cg!(work, M, v; M=M.pre, verbose=5)
        bufj .= work.x
      end
      buf
    end

    end
  end

end

