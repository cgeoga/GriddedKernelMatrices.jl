
function _solve! end

# Square mul! methods:
for T in (SymToeplitz, MaskedSymToeplitz, SymBTTB, MaskedSymBTTB)
  @eval begin

  Base.:*(M::$T, x) = mul!(similar(x), M, x)

  end
end

# Non-square mul! methods:
for T in (CrossMaskedSymToeplitz, AdjointCrossMaskedSymToeplitz, CrossMaskedSymBTTB, AdjointCrossMaskedSymBTTB)
  @eval begin

  Base.:*(M::$T, x::AbstractVector) = mul!(zeros(size(M,1)), M, x)
  Base.:*(M::$T, x::AbstractMatrix) = mul!(zeros(size(M,1), size(x, 2)), M, x)

  end
end

# Matrix mul! methods:
for T in (SymToeplitz, MaskedSymToeplitz, SymBTTB, MaskedSymBTTB,
          CrossMaskedSymToeplitz, AdjointCrossMaskedSymToeplitz, 
          CrossMaskedSymBTTB, AdjointCrossMaskedSymBTTB)
  @eval begin

    function LinearAlgebra.mul!(buf::AbstractMatrix, M::$T, v::AbstractMatrix)
      size(buf, 2) == size(v, 2) || error("Number of columns in input and output don't agree.")
      for j in 1:size(buf, 2)
        bufj = view(buf, :, j)
        vj   = view(v,   :, j)
        mul!(bufj, M, vj)
      end
      buf
    end

  end
end

# Square ldiv! methods:
for T in (SymToeplitz, MaskedSymToeplitz, SymBTTB, MaskedSymBTTB)
  @eval begin

  function LinearAlgebra.ldiv!(buf, M::$T, v)
    size(buf) == size(v)    || error("Input and output dimensions don't agree.")
    size(M,1) == size(v, 1) || error("Input and matrix dimensions don't agree.")
    if hasmethod(_solve!, (typeof(buf), $T, typeof(v)))
      _solve!(buf, M, v)
      return buf
    end
    error("Please load the `Krylov.jl` extension for ldiv!")
  end

  Base.:\(M::$T, v) = ldiv!(similar(v), M, v)

  end
end

