
struct BCCB{P}
  Λ::Matrix{Float64}
  sizes::Tuple{Int, Int}
  buf::Matrix{ComplexF64}
  plan::P
end

Base.eltype(M::BCCB)  = Float64
Base.size(M::BCCB)    = (prod(size(M.sizes)), prod(size(M.sizes)))
Base.size(M::BCCB, j) = size(M)[j]
LinearAlgebra.issymmetric(M::BCCB) = false
LinearAlgebra.ishermitian(M::BCCB) = false

function BCCB(c::Vector{Float64}, sizes::Tuple{Int, Int})
  buf = Matrix{ComplexF64}(undef, sizes...)
  plan = plan_fft!(buf)
  BCCB(real(fft(reshape(c, sizes))), sizes, buf, plan)
end

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::BCCB, v::AbstractVector{Float64})
  M.buf  .= reshape(v, M.sizes)
  M.plan*M.buf
  M.buf .*= M.Λ
  M.plan\M.buf
  for j in eachindex(buf)
    @inbounds buf[j] = real(M.buf[j])
  end
  buf
end

struct SymBTTB{B,P}
  bccb::BCCB{B}
  buf1::Vector{Float64}
  buf2::Vector{Float64}
  orig_sizes::Tuple{Int, Int}
  pre::P
end

Base.eltype(M::SymBTTB)  = Float64
Base.size(M::SymBTTB)    = (prod(M.orig_sizes), prod(M.orig_sizes))
Base.size(M::SymBTTB, j) = size(M)[j]
LinearAlgebra.issymmetric(M::SymBTTB) = true
LinearAlgebra.ishermitian(M::SymBTTB) = true

function SymBTTB(first_columns::Vector{Vector{Float64}}, pre=I)
  circ_inner = circulant_extend.(first_columns)
  circ_outer = circulant_extend(circ_inner)
  sizes  = (length(circ_outer[1]), length(circ_outer))
  c_flat = reduce(vcat, circ_outer)
  bccb = BCCB(c_flat, sizes)
  buf1 = Vector{Float64}(undef, length(c_flat))
  buf2 = Vector{Float64}(undef, length(c_flat))
  orig_sizes = (length(first_columns[1]), length(first_columns))
  SymBTTB(bccb, buf1, buf2, orig_sizes, pre)
end

function LinearAlgebra.mul!(buf::AbstractVector{Float64}, M::SymBTTB, v::AbstractVector{Float64})
  length(buf) == length(v) || error("Input and output dimensions don't agree.")
  size(M,1)   == length(v) || error("Input and matrix dimensions don't agree.")
  fill!(M.buf1, 0.0)
  v_grid = reshape(v, M.orig_sizes)
  buf1_grid = reshape(M.buf1, M.bccb.sizes)
  for j in 1:M.orig_sizes[2]
    for i in 1:M.orig_sizes[1]
      @inbounds buf1_grid[i, j] = v_grid[i, j]
    end
  end
  mul!(M.buf2, M.bccb, M.buf1)
  buf2_grid = reshape(M.buf2, M.bccb.sizes)
  buf_grid = reshape(buf, M.orig_sizes)
  for j in 1:M.orig_sizes[2]
    for i in 1:M.orig_sizes[1]
      @inbounds buf_grid[i, j] = buf2_grid[i, j]
    end
  end
  buf
end

