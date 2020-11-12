module PSDMatrices

using LinearAlgebra
import LinearAlgebra: cholesky, diag, det, logdet

import Base: +, *, /, \, ==, inv, Matrix, kron, copy

# Write your package code here.
include("psdmatrix.jl")

export PSDMatrix

end
