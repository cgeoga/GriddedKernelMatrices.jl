
v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]
v3 = [7.0, 8.0, 9.0]

given_ixs = [1,2,4,5,7,9]
op = GriddedKernelMatrices.MaskedSymBTTB([v1, v2, v3], given_ixs)

M1 = [1.0 2.0 3.0
      2.0 1.0 2.0
      3.0 2.0 1.0]

M2 = [4.0 5.0 6.0
      5.0 4.0 5.0
      6.0 5.0 4.0]

M3 = [7.0 8.0 9.0
      8.0 7.0 8.0
      9.0 8.0 7.0]

M  = [M1 M2 M3
      M2 M1 M2
      M3 M2 M1]
M  = M[given_ixs, given_ixs]

v = collect(1.0:9.0)[given_ixs]
@test M*v ≈ op*v

