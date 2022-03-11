
module PSDMatrices

import Base: \, /, size, inv, copy, ==
using LinearAlgebra
import LinearAlgebra: det, logdet

struct PSDMatrix{T,Ltype} <: AbstractMatrix{T}
    L::Ltype
    PSDMatrix(L::AbstractMatrix{T}) where {T} = new{T,typeof(L)}(L)
end

# Custom functions

function todense(M::PSDMatrix)
    return M.L * M.L'
end


function X_A_Xt(A::PSDMatrix, X::AbstractMatrix)
    return PSDMatrix(X * A.L)
end


function choleskify_factor(M::PSDMatrix)
    QR = qr(M.L')
    R = triu(QR.factors)
    R = nonnegative_diagonal(R)
    return PSDMatrix(LowerTriangular(R'))
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




# Exports

export PSDMatrix
export todense
export choleskify_factor
export X_A_Xt

end
