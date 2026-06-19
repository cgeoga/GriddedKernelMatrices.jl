module LatticeMatrices

  using LinearAlgebra, FFTW, StaticArraysCore

  include("toeplitz.jl")

  include("block_toeplitz.jl")

  include("mask.jl")

  include("kernel.jl")
  export lattice_kernel_matrix, VecchiaPreconditioner

end 

