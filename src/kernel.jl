
struct NoPreconditioner end

Base.@kwdef struct VecchiaPreconditioner 
  k::Int = 30
end

gen_preconditioner(::NoPreconditioner, pts, kernel) = I

function identify_lattice(pts::Vector{Float64})
  t_min = minimum(pts)
  t_max = maximum(pts)
  dt    = minimum(diff(pts))
  N     = round(Int, (t_max - t_min)/dt) + 1
  if N > 10 * length(pts)
    @warn "Embedding these points in a grid requires an internal FFT of size $(2*N-1). If this is larger than expected, please double check your provided locations."
  end
  ixs  = round.(Int, (pts .- t_min)./dt) .+ 1
  grid = [d*dt for d in 0:(N-1)]
  (;ixs, grid)
end

function lattice_kernel_matrix(pts::Vector{Float64}, kernel, 
                               preconditioner=NoPreconditioner())
  (;ixs, grid) = identify_lattice(pts)
  toep = SymToeplitz(kernel.(grid))
  pre  = gen_preconditioner(preconditioner, pts, kernel)
  MaskedSymToeplitz(toep, ixs, zeros(length(grid)), zeros(length(grid)), pre)
end

function identify_lattice(pts1::Vector{Float64}, pts2::Vector{Float64})
  all_pts = sort(unique(vcat(pts1, pts2)))
  t_min = all_pts[1]
  t_max = all_pts[end]
  dt = length(all_pts) > 1 ? minimum(diff(all_pts)) : 1.0
  N  = round(Int, (t_max - t_min) / dt) + 1
  if N > 10 * length(all_pts)
    @warn "Embedding these points in a grid requires an internal FFT of size $(2*N-1). If this is larger than expected, please double check your provided locations."
  end
  ixs_out = round.(Int, (pts1 .- t_min) ./ dt) .+ 1
  ixs_in  = round.(Int, (pts2 .- t_min) ./ dt) .+ 1
  grid    = [d*dt for d in 0:(N-1)]
  (;N, ixs_in, ixs_out, grid)
end

function lattice_kernel_matrix(pts1::Vector{Float64}, pts2::Vector{Float64}, kernel)
  (;N, ixs_in, ixs_out, grid) = identify_lattice(pts1, pts2)
  toep = SymToeplitz(kernel.(grid))
  CrossMaskedSymToeplitz(toep, ixs_out, ixs_in, 
                         zeros(length(grid)), zeros(length(grid)))
end

function identify_lattice(pts::Vector{SVector{2,Float64}})
  ex1 = extrema(x->x[1], pts)
  ex2 = extrema(x->x[2], pts)
  dx1 = minimum(diff(sort(unique(getindex.(pts, 1)))))
  dx2 = minimum(diff(sort(unique(getindex.(pts, 2)))))
  nx  = round(Int, (ex1[2] - ex1[1])/dx1) + 1
  ny  = round(Int, (ex2[2] - ex2[1])/dx2) + 1
  if (nx * ny) > 10 * length(pts)
    @warn "Embedding these points in a grid requires an internal FFT of size $(2*nx-1) x $(2*ny-1). If this is larger than expected, please double check your provided locations."
  end
  lag1r = range(0.0, step=dx1, length=nx)
  lag2r = range(0.0, step=dx2, length=ny)
  observed_ixs = map(pts) do pt
    i = round(Int, (pt[2] - ex2[1])/ dx2) + 1
    j = round(Int, (pt[1] - ex1[1])/ dx1) + 1
    i + (j - 1)*ny
  end
  (;lag1r, lag2r, observed_ixs)
end

function lattice_kernel_matrix(pts::Vector{SVector{2,Float64}}, kernel,
                               preconditioner=NoPreconditioner())
  (;lag1r, lag2r, observed_ixs) = identify_lattice(pts)
  bttb = SymBTTB(kernel, step(lag1r), step(lag2r), length(lag1r), length(lag2r))
  buf1 = zeros(Float64, length(lag1r)*length(lag2r))
  buf2 = zeros(Float64, length(lag1r)*length(lag2r))
  pre  = gen_preconditioner(preconditioner, pts, kernel)
  MaskedSymBTTB(bttb, observed_ixs, buf1, buf2, pre)
end

#=
function identify_lattice(pts1::Vector{SVector{2,Float64}}, 
                          pts2::Vector{SVector{2,Float64}})
  all_pts = vcat(pts1, pts2)
  ex1 = extrema(x->x[1], all_pts)
  ex2 = extrema(x->x[2], all_pts)
  dx1 = minimum(diff(sort(unique(getindex.(all_pts, 1)))))
  dx2 = minimum(diff(sort(unique(getindex.(all_pts, 2)))))
  nx  = round(Int, (ex1[2] - ex1[1])/dx1) + 1
  ny  = round(Int, (ex2[2] - ex2[1])/dx2) + 1
  if (nx * ny) > 10 * length(all_pts)
    @warn "Embedding these points in a grid requires an internal FFT of size $(2*nx-1) x $(2*ny-1). If this is larger than expected, please double check your provided locations."
  end
  lag1r = range(0.0, step=dx1, length=nx)
  lag2r = range(0.0, step=dx2, length=ny)
  ixs_out = map(pts1) do pt
    i = round(Int, (pt[2] - ex2[1])/dx2) + 1
    j = round(Int, (pt[1] - ex1[1])/dx1) + 1
    i + (j-1)*ny
  end
  ixs_in = map(pts2) do pt
    i = round(Int, (pt[2] - ex2[1])/dx2) + 1
    j = round(Int, (pt[1] - ex1[1])/dx1) + 1
    i + (j-1)*ny
  end
  (;lag1r, lag2r, ixs_in, ixs_out)
end
=#

function identify_lattice(pts1::Vector{SVector{2,Float64}}, 
                          pts2::Vector{SVector{2,Float64}})
  all_pts = vcat(pts1, pts2)
  ex1 = extrema(x->x[1], all_pts)
  ex2 = extrema(x->x[2], all_pts)
  # TODO (cg 2026/07/03 13:26): think more about how this should work.
  function get_dx(coords)
    diffs = diff(sort(unique(coords)))
    idx = findfirst(>(1e-10), diffs) 
    isnothing(idx) ? 1.0 : diffs[idx]
  end
  dx1 = get_dx(getindex.(all_pts, 1))
  dx2 = get_dx(getindex.(all_pts, 2))
  nx  = round(Int, (ex1[2] - ex1[1])/dx1) + 1
  ny  = round(Int, (ex2[2] - ex2[1])/dx2) + 1
  if (nx * ny) > 10 * length(all_pts)
    @warn "Embedding these points in a grid requires an internal FFT of size $(2*nx-1) x $(2*ny-1). If this is larger than expected, please double check your provided locations."
  end
  lag1r = range(0.0, step=dx1, length=nx)
  lag2r = range(0.0, step=dx2, length=ny)
  ixs_out = map(pts1) do pt
    i = round(Int, (pt[2] - ex2[1])/dx2) + 1
    j = round(Int, (pt[1] - ex1[1])/dx1) + 1
    i + (j-1)*ny
  end
  ixs_in = map(pts2) do pt
    i = round(Int, (pt[2] - ex2[1])/dx2) + 1
    j = round(Int, (pt[1] - ex1[1])/dx1) + 1
    i + (j-1)*ny
  end
  (;lag1r, lag2r, ixs_in, ixs_out)
end

function lattice_kernel_matrix(pts1::Vector{SVector{2,Float64}}, 
                               pts2::Vector{SVector{2,Float64}}, 
                               kernel)
  (;lag1r, lag2r, ixs_in, ixs_out) = identify_lattice(pts1, pts2)
  bttb = SymBTTB(kernel, step(lag1r), step(lag2r), length(lag1r), length(lag2r))
  buf1 = zeros(Float64, length(lag1r)*length(lag2r))
  buf2 = zeros(Float64, length(lag1r)*length(lag2r))
  CrossMaskedSymBTTB(bttb, ixs_out, ixs_in, buf1, buf2)
end

