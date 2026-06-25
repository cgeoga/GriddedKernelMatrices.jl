
struct SymToeplitz{F,P}
  n::Int
  c_ext_ft::Vector{ComplexF64}
  buf::Vector{ComplexF64}
  plan::F
  pre::P
end

circulant_extend(c) = vcat(c, reverse(c)[2:(end-1)])

function SymToeplitz(v, pre=I)
  c_ext = complex(circulant_extend(v))
  buf   = zeros(complex(eltype(v)), length(c_ext))
  plan  = plan_fft!(buf, 1)
  fft!(c_ext)
  SymToeplitz(length(v), c_ext, buf, plan, pre)
end

Base.eltype(M::SymToeplitz)  = Float64
Base.size(M::SymToeplitz)    = M.n
Base.size(M::SymToeplitz, j) = size(M)[j]
LinearAlgebra.issymmetric(M::SymToeplitz) = true
LinearAlgebra.ishermitian(M::SymToeplitz) = true

function LinearAlgebra.mul!(buf::Vector{Float64}, M::SymToeplitz{F,P}, 
                            x::Vector{Float64}) where{F,P}
  fill!(M.buf, 0.0 + 0.0*im)
  copyto!(view(M.buf, 1:length(x)), x)
  M.plan*M.buf
  for j in 1:size(M.buf,1)
    @inbounds M.buf[j] *= M.c_ext_ft[j]
  end
  M.plan\M.buf
  for j in 1:size(buf, 1)
    @inbounds buf[j] = real(M.buf[j])
  end
  buf
end

