module PSDMatrices

using LinearAlgebra
import Base: +, *, /, \, ==, inv, Matrix, kron, copy

# Write your package code here.
include("psdmatrix.jl")

export PSDMatrix

end
