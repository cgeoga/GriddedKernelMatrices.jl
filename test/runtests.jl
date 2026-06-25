
using Test, GriddedKernelMatrices, StaticArrays, LowRankApprox
using GriddedKernelMatrices.LinearAlgebra

@testset "mul" begin
  include("toeplitz.jl")
  include("different_grids.jl")
  include("kernel_gaps.jl")
  include("cross_mul.jl")
  include("aniso.jl")
end

@testset "solve" begin
  include("solve.jl")
end

@testset "ecosystem" begin
  include("sketch.jl")
end

