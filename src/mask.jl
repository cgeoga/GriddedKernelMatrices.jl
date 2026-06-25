
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

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::MaskedSymToeplitz, v::AbstractVector{Float64})
  length(buf) == length(v) || error("Input and output dimensions don't agree.")
  size(M,1)   == length(v) || error("Input and matrix dimensions don't agree.")
  fill!(M.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.ixs)
      M.buf1[idx] = v[i]
  end
  mul!(M.buf2, M.toep, M.buf1)
  @inbounds for (i, idx) in enumerate(M.ixs)
      buf[i] = M.buf2[idx]
  end
  buf
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

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::CrossMaskedSymToeplitz, x::AbstractVector{Float64})
  fill!(M.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.ixs_in)
    M.buf1[idx] = x[i]
  end
  mul!(M.buf2, M.toep, M.buf1)
  @inbounds for (i, idx) in enumerate(M.ixs_out)
    buf[i] = M.buf2[idx]
  end
  buf
end

struct AdjointCrossMaskedSymToeplitz{F}
  parent::CrossMaskedSymToeplitz{F}
end

Base.eltype(M::AdjointCrossMaskedSymToeplitz) = Float64
Base.size(M::AdjointCrossMaskedSymToeplitz) = (length(M.parent.ixs_in), length(M.parent.ixs_out))
Base.size(M::AdjointCrossMaskedSymToeplitz, d::Int) = d == 1 ? length(M.parent.ixs_in) : length(M.parent.ixs_out)

Base.adjoint(M::CrossMaskedSymToeplitz) = AdjointCrossMaskedSymToeplitz(M)
Base.adjoint(M::AdjointCrossMaskedSymToeplitz) = M.parent

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::AdjointCrossMaskedSymToeplitz, 
                            v::AbstractVector{Float64})
  fill!(M.parent.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.parent.ixs_out)
    M.parent.buf1[idx] = v[i]
  end
  mul!(M.parent.buf2, M.parent.toep, M.parent.buf1)
  @inbounds for (i, idx) in enumerate(M.parent.ixs_in)
    buf[i] = M.parent.buf2[idx]
  end
  buf
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

function MaskedSymBTTB(kernel, dx::Float64, dy::Float64, nx::Int, ny::Int,
                       given_ixs::Vector{Int}, pre=I)
  bttb = SymBTTB(kernel, dx, dy, nx, ny)
  N_total = nx * ny
  full_in = Vector{Float64}(undef, N_total)
  full_out = Vector{Float64}(undef, N_total)
  MaskedSymBTTB(bttb, given_ixs, full_in, full_out, pre)
end

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::MaskedSymBTTB, v::AbstractVector{Float64})
  length(buf) == length(v) || error("Input and output dimensions don't agree.")
  size(M,1)   == length(v) || error("Input and matrix dimensions don't agree.")
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

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::CrossMaskedSymBTTB, v::AbstractVector{Float64})
  fill!(M.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.ixs_in)
    M.buf1[idx] = v[i]
  end
  mul!(M.buf2, M.bttb, M.buf1)
  @inbounds for (i, idx) in enumerate(M.ixs_out)
    buf[i] = M.buf2[idx]
  end
  buf
end

struct AdjointCrossMaskedSymBTTB{B,P}
  parent::CrossMaskedSymBTTB{B,P}
end

Base.eltype(M::AdjointCrossMaskedSymBTTB) = Float64
Base.size(M::AdjointCrossMaskedSymBTTB) = (length(M.parent.ixs_in), length(M.parent.ixs_out))
Base.size(M::AdjointCrossMaskedSymBTTB, d::Int) = d == 1 ? length(M.parent.ixs_in) : length(M.parent.ixs_out)

# Wire up the standard adjoint syntax
Base.adjoint(M::CrossMaskedSymBTTB) = AdjointCrossMaskedSymBTTB(M)
Base.adjoint(M::AdjointCrossMaskedSymBTTB) = M.parent

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::AdjointCrossMaskedSymBTTB, 
                            v::AbstractVector{Float64})
  fill!(M.parent.buf1, 0.0)
  @inbounds for (i, idx) in enumerate(M.parent.ixs_out)
    M.parent.buf1[idx] = v[i]
  end
  mul!(M.parent.buf2, M.parent.bttb, M.parent.buf1)
  @inbounds for (i, idx) in enumerate(M.parent.ixs_in)
    buf[i] = M.parent.buf2[idx]
  end
  buf
end

