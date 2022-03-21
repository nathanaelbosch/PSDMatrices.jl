
module PSDMatrices

import Base: \, /, size, inv, copy, copy!, ==, show, similar, Matrix
using LinearAlgebra
import LinearAlgebra: det, logdet

struct PSDMatrix{T,FactorType} <: AbstractMatrix{T}
    R::FactorType
    PSDMatrix(R::AbstractMatrix{T}) where {T} = new{T,typeof(R)}(R)
end

# Base overloads
Matrix(M::PSDMatrix) = M.R' * M.R
size(M::PSDMatrix) = (size(M.R, 2), size(M.R, 2))
inv(M::PSDMatrix) = PSDMatrix(inv(M.R'))
\(A::PSDMatrix, B::AbstractVecOrMat) = A.R \ (A.R' \ B)
/(B::AbstractVecOrMat, A::PSDMatrix) = B / A.R / A.R'
copy(M::PSDMatrix) = PSDMatrix(copy(M.R))
similar(M::PSDMatrix, element_type::Type=eltype(M)) = PSDMatrix(similar(M.R, element_type))
copy!(dst::PSDMatrix, src::PSDMatrix) = (copy!(dst.R, src.R); dst)
==(M1::PSDMatrix, M2::PSDMatrix) = M1.R == M2.R  # todo: same as isequal()?!
function show(io::IO, M::PSDMatrix)
    print(io, "$(size(M,1))x$(size(M,2)) $(typeof(M)); R=")
    show(io, M.R)
end
function show(io::IO, m::MIME"text/plain", M::PSDMatrix)
    println(io, "$(size(M,1))x$(size(M,2)) $(typeof(M)) ")
    print(io, " Right square root: R=")
    show(io, m, M.R)
end

# LinearAlgebra overloads

function det(M::PSDMatrix)
    confirm_factor_is_square(M)
    return det(M.R)^2
end

function logdet(M::PSDMatrix)
    confirm_factor_is_square(M)
    return 2 * logdet(M.R)
end

function confirm_factor_is_square(M::PSDMatrix)
    if size(M.R, 1) != size(M.R, 2)
        msg = (
            "The requested operation is not available for a PSDMatrix with a non-square factor." *
            "The factor of the received PSDMatrix has dimensions ($(size(M.R,1)), $(size(M.R,2))). " *
            "Try turning the PSDMatrix into a dense matrix first."
        )
        throw(MethodError(msg))
    end
end

# Custom functions

X_A_Xt(; A::PSDMatrix, X::AbstractMatrix) = PSDMatrix(A.R * X')
X_A_Xt(A::PSDMatrix, X::AbstractMatrix) = X_A_Xt(A=A, X=X)
function X_A_Xt!(out::PSDMatrix; A::PSDMatrix, X::AbstractMatrix)
    mul!(out.R, A.R, X')
    return out
end
X_A_Xt!(out::PSDMatrix, A::PSDMatrix, X::AbstractMatrix) = X_A_Xt!(out, A=A, X=X)

function add_cholesky(A::PSDMatrix, B::PSDMatrix)
    sum_dense = Matrix(A) + Matrix(B)
    factor = cholesky(sum_dense).U
    return PSDMatrix(factor)
end

function add_qr(A::PSDMatrix, B::PSDMatrix)
    stack = vcat(A.R, B.R)
    matrix = PSDMatrix(stack)
    return triangularize_factor(matrix)
end

function triangularize_factor(M::PSDMatrix)
    R = qr(M.R).R
    return PSDMatrix(UpperTriangular(R))
end

function choleskify(M::PSDMatrix)
    if M.R isa UpperTriangular
        return Cholesky(nonnegative_diagonal(M.R))
    end
    errormsg = "choleskify() is only defined for PSDMatrix types with triangular factors."
    throw(MethodError(errormsg))
end

function nonnegative_diagonal(R)
    signs = signbit.(diag(R))
    R .*= (1 .- 2 .* signs)
    return R
end

export PSDMatrix
export add_cholesky
export add_qr
export triangularize_factor
export choleskify
export X_A_Xt
export X_A_Xt!

end
