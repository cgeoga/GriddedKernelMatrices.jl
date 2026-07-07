
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
      @info "For block solves, (P)GMRES is used instead of (P)CG. Please manually loop over individual solves if you specifically need CG." maxlog=1
      work = BlockGmresWorkspace(M, v; memory=30)
      block_gmres!(work, M, v; M=M.pre)
      buf .= work.X
    end

    end
  end

end

