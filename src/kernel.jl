
struct NoPreconditioner end

Base.@kwdef struct VecchiaPreconditioner 
  k::Int = 30
end

gen_preconditioner(::NoPreconditioner, pts, kernel) = I

function lattice_kernel_matrix(pts::Vector{Float64}, kernel, 
                               preconditioner=NoPreconditioner())
  t_min = minimum(pts)
  t_max = maximum(pts)
  dt    = minimum(diff(pts))
  N     = round(Int, (t_max - t_min)/dt) + 1
  ixs   = round.(Int, (pts .- t_min)./dt) .+ 1
  v     = [kernel(d * dt) for d in 0:(N-1)]
  toep  = SymToeplitz(v)
  buf1  = zeros(Float64, N)
  buf2  = zeros(Float64, N)
  pre   = gen_preconditioner(preconditioner, pts, kernel)
  MaskedSymToeplitz(toep, ixs, buf1, buf2, pre)
end

function lattice_kernel_matrix(pts::Vector{SVector{2,Float64}}, kernel,
                               preconditioner=NoPreconditioner())

  ex1 = extrema(x->x[1], pts)
  ex2 = extrema(x->x[2], pts)
  dx1 = minimum(diff(sort(unique(getindex.(pts, 1)))))
  dx2 = minimum(diff(sort(unique(getindex.(pts, 2)))))
  nx  = round(Int, (ex1[2] - ex1[1])/dx1) + 1
  ny  = round(Int, (ex2[2] - ex2[1])/dx2) + 1
  lag1r = range(0.0, step=dx1, length=nx)
  lag2r = range(0.0, step=dx2, length=ny)
  full_columns = map(lag1r) do l1
    [kernel(SVector(l1, l2)) for l2 in lag2r]
  end
  observed_ixs = map(pts) do pt
    i = round(Int, (pt[2] - ex2[1])/ dx2) + 1
    j = round(Int, (pt[1] - ex1[1])/ dx1) + 1
    i + (j - 1)*ny
  end
  pre   = gen_preconditioner(preconditioner, pts, kernel)
  MaskedSymBTTB(full_columns, observed_ixs, pre)
end

