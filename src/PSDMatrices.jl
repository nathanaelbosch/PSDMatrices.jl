
module PSDMatrices

import Base: \, /, size, inv, copy, ==, show
using LinearAlgebra
import LinearAlgebra: det, logdet

struct PSDMatrix{T,FactorType} <: AbstractMatrix{T}
    R::FactorType
    PSDMatrix(R::AbstractMatrix{T}) where {T} = new{T,typeof(R)}(R)
end

# Base overloads

size(M::PSDMatrix) = (size(M.R, 2), size(M.R, 2))
inv(M::PSDMatrix) = PSDMatrix(inv(M.R'))
\(A::PSDMatrix, B::AbstractVecOrMat) = A.R \ (A.R' \ B)
/(B::AbstractVecOrMat, A::PSDMatrix) = B / A.R / A.R'
copy(M::PSDMatrix) = PSDMatrix(copy(M.R))
==(M1::PSDMatrix, M2::PSDMatrix) = M1.R == M2.R  # todo: same as isequal()?!
# Base.show(io::IO, x::MyString) = print(io, x.s)
show(io::IO, M::PSDMatrix) = begin
    print(io, "$(size(M,1))x$(size(M,2)) $(typeof(M)); R=")
    show(io, M.R)
end
show(io::IO, m::MIME"text/plain", M::PSDMatrix) = begin
    println(io, "$(size(M,1))x$(size(M,2)) $(typeof(M)) ")
    print(io, " Right square root: ")
    show(io, m, M.R)
end

# LinearAlgebra overloads

det(M::PSDMatrix) = det(M.R)^2
logdet(M::PSDMatrix) = 2 * logdet(M.R)

function nonnegative_diagonal(R)
    signs = signbit.(diag(R))
    R .*= (1 .- 2 .* signs)
    return R
end

# Custom functions

function todense(M::PSDMatrix)
    return M.R' * M.R
end

function X_A_Xt(; A::PSDMatrix, X::AbstractMatrix)
    return PSDMatrix(A.R * X')
end
X_A_Xt(A::PSDMatrix, X::AbstractMatrix) = X_A_Xt(A=A, X=X)

function add_cholesky(A::PSDMatrix, B::PSDMatrix)
    sum_dense = todense(A) + todense(B)
    factor = cholesky(sum_dense).U
    return PSDMatrix(factor)
end

function add_qr(A::PSDMatrix, B::PSDMatrix)
    stack = vcat(A.R, B.R)
    matrix = PSDMatrix(stack)
    return choleskify_factor(matrix)
end

function choleskify_factor(M::PSDMatrix)
    R = qr(M.R).R
    R = nonnegative_diagonal(R)
    return PSDMatrix(UpperTriangular(R))
end

export PSDMatrix
export todense
export add_cholesky
export add_qr
export choleskify_factor
export X_A_Xt

end
