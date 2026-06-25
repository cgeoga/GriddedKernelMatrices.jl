
function _solve! end

for T in (SymToeplitz, MaskedSymToeplitz, SymBTTB, MaskedSymBTTB)
  @eval begin

  Base.:*(M::$T, x) = mul!(similar(x), M, x)

  end
end

for T in (CrossMaskedSymToeplitz, CrossMaskedSymBTTB)
  @eval begin

  Base.:*(M::$T, x::AbstractVector) = mul!(zeros(size(M,1)), M, x)

  end
end

# 2. Extend to matrix RHS things
# 3. Extend to AbstractVector and AbstractMatrix (make as general as possible
#    without hitting MethodErrors about ambiguity).
for T in (SymToeplitz, MaskedSymToeplitz, SymBTTB, MaskedSymBTTB)
  @eval begin

  function LinearAlgebra.ldiv!(buf::Vector{Float64}, M::$T, 
                               x::Vector{Float64})
    if hasmethod(_solve!, (Vector{Float64}, $T, Vector{Float64}))
      _solve!(buf, M, x)
      return buf
    end
    error("Please load the `Krylov.jl` extension for ldiv!")
  end

  Base.:\(M::$T, x) = ldiv!(copy(x), M, x)

  end
end

