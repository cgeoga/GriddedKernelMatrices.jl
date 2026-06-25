
using Test, LatticeMatrices, StaticArrays
using LatticeMatrices.LinearAlgebra

@testset "mul" begin
  include("toeplitz.jl")
  include("block_toeplitz.jl")
  include("masked_mul.jl")
  include("different_grids.jl")
  include("kernel_gaps.jl")
  include("cross_mul.jl")
end

@testset "solve" begin
  include("solve.jl")
end

