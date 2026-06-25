
struct MaskedSymToeplitz{F,P}
  toep::SymToeplitz{F}
  ixs::Vector{Int}
  buf1::Vector{Float64}
  buf2::Vector{Float64}
  pre::P
end

Base.eltype(M::MaskedSymToeplitz) = Float64
Base.size(M::MaskedSymToeplitz) = (length(M.ixs), length(M.ixs))
Base.size(M::MaskedSymToeplitz, d::Int) = length(M.ixs)
LinearAlgebra.issymmetric(M::MaskedSymToeplitz) = true
LinearAlgebra.ishermitian(M::MaskedSymToeplitz) = true

function LinearAlgebra.mul!(y::Vector{Float64}, M::MaskedSymToeplitz, x::Vector{Float64})
  fill!(M.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.ixs)
      M.buf1[idx] = x[i]
  end
  mul!(M.buf2, M.toep, M.buf1)
  @inbounds for (i, idx) in enumerate(M.ixs)
      y[i] = M.buf2[idx]
  end
  y
end

struct CrossMaskedSymToeplitz{F}
  toep::SymToeplitz{F}
  ixs_out::Vector{Int}
  ixs_in::Vector{Int}
  buf1::Vector{Float64}
  buf2::Vector{Float64}
end

Base.eltype(M::CrossMaskedSymToeplitz) = Float64
Base.size(M::CrossMaskedSymToeplitz) = (length(M.ixs_out), length(M.ixs_in))
Base.size(M::CrossMaskedSymToeplitz, d::Int) = d == 1 ? length(M.ixs_out) : length(M.ixs_in)

function LinearAlgebra.mul!(y::Vector{Float64}, M::CrossMaskedSymToeplitz, x::Vector{Float64})
  fill!(M.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.ixs_in)
    M.buf1[idx] = x[i]
  end
  mul!(M.buf2, M.toep, M.buf1)
  @inbounds for (i, idx) in enumerate(M.ixs_out)
    y[i] = M.buf2[idx]
  end
  y
end

struct MaskedSymBTTB{B,P}
  bttb::B
  given_ixs::Vector{Int64}
  full_in::Vector{Float64}
  full_out::Vector{Float64}
  pre::P
end

Base.eltype(M::MaskedSymBTTB)  = Float64
Base.size(M::MaskedSymBTTB)    = (length(M.given_ixs), length(M.given_ixs))
Base.size(M::MaskedSymBTTB, j) = size(M)[j]
LinearAlgebra.issymmetric(M::MaskedSymBTTB) = true
LinearAlgebra.ishermitian(M::MaskedSymBTTB) = true

function MaskedSymBTTB(full_first_columns::Vector{Vector{Float64}}, 
                       given_ixs::Vector{Int64}, pre=I)
  bttb = SymBTTB(full_first_columns)
  N_total = length(full_first_columns[1]) * length(full_first_columns)
  full_in = Vector{Float64}(undef, N_total)
  full_out = Vector{Float64}(undef, N_total)
  MaskedSymBTTB(bttb, given_ixs, full_in, full_out, pre)
end

function LinearAlgebra.mul!(buf::Vector{Float64}, M::MaskedSymBTTB, v::Vector{Float64})
  fill!(M.full_in, 0.0)
  for i in eachindex(v)
    @inbounds M.full_in[M.given_ixs[i]] = v[i]
  end
  mul!(M.full_out, M.bttb, M.full_in)
  for i in eachindex(buf)
    @inbounds buf[i] = M.full_out[M.given_ixs[i]]
  end
  buf
end

function Base.:*(M::MaskedSymBTTB, v::Vector{Float64})
  buf = similar(v)
  mul!(buf, M, v)
end

struct CrossMaskedSymBTTB{B,P}
  bttb::SymBTTB{B,P}
  ixs_out::Vector{Int}
  ixs_in::Vector{Int}
  buf1::Vector{Float64}
  buf2::Vector{Float64}
end

Base.eltype(M::CrossMaskedSymBTTB)  = Float64
Base.size(M::CrossMaskedSymBTTB)    = (length(M.ixs_out), length(M.ixs_in))
Base.size(M::CrossMaskedSymBTTB, j) = j == 1 ? length(M.ixs_out) : length(M.ixs_in)
LinearAlgebra.issymmetric(M::CrossMaskedSymBTTB) = false
LinearAlgebra.ishermitian(M::CrossMaskedSymBTTB) = false

function LinearAlgebra.mul!(y::Vector{Float64}, M::CrossMaskedSymBTTB, x::Vector{Float64})
  fill!(M.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.ixs_in)
    M.buf1[idx] = x[i]
  end
  mul!(M.buf2, M.bttb, M.buf1)
  @inbounds for (i, idx) in enumerate(M.ixs_out)
    y[i] = M.buf2[idx]
  end
  y
end

