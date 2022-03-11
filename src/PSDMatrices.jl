
module PSDMatrices

import Base: \, /, size, inv, copy, ==
using LinearAlgebra
import LinearAlgebra: det, logdet

struct PSDMatrix{T,Ltype} <: AbstractMatrix{T}
    L::Ltype
    PSDMatrix(L::AbstractMatrix{T}) where {T} = new{T,typeof(L)}(L)
end

# Base overloads

size(M::PSDMatrix) = (size(M.L, 1), size(M.L, 1))
inv(M::PSDMatrix) = PSDMatrix(inv(M.L'))
\(A::PSDMatrix, B::AbstractVecOrMat) = A.L' \ (A.L \ B)
/(B::AbstractVecOrMat, A::PSDMatrix) = B / A.L' / A.L
copy(M::PSDMatrix) = PSDMatrix(copy(M.L))
==(M1::PSDMatrix, M2::PSDMatrix) = M1.L == M2.L  # todo: same as isequal()?!

# LinearAlgebra overloads

det(M::PSDMatrix) = det(M.L)^2
logdet(M::PSDMatrix) = 2 * logdet(M.L)

function nonnegative_diagonal(R)
    signs = signbit.(diag(R))
    R .*= (1 .- 2 .* signs)
    return R
end

# Custom functions

function todense(M::PSDMatrix)
    return M.L * M.L'
end

function X_A_Xt(; A::PSDMatrix, X::AbstractMatrix)
    return PSDMatrix(X * A.L)
end
X_A_Xt(A::PSDMatrix, X::AbstractMatrix) = X_A_Xt(A=A, X=X)

function add_cholesky(A::PSDMatrix, B::PSDMatrix)
    sum_dense = todense(A) + todense(B)
    factor = cholesky(sum_dense).L
    return PSDMatrix(factor)
end

function add_qr(A::PSDMatrix, B::PSDMatrix)
    stack = hcat(A.L, B.L)
    matrix = PSDMatrix(stack)
    return choleskify_factor(matrix)
end

function choleskify_factor(M::PSDMatrix)
    R = qr(M.L').R
    R = nonnegative_diagonal(R)
    return PSDMatrix(LowerTriangular(R'))
end

export PSDMatrix
export todense
export add_cholesky
export add_qr
export choleskify_factor
export X_A_Xt

end
