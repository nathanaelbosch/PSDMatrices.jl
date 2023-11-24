
module PSDMatrices

import Base: \, /, size, inv, copy, copy!, ==, show, similar, Matrix, iszero, isapprox
using LinearAlgebra
import LinearAlgebra: det, logabsdet, diag

using PDMats
using PDMats
import PDMats: unwhiten!

struct PSDMatrix{T,FactorType} <: AbstractPDMat{T}
    R::FactorType
end
PSDMatrix(R::AbstractMatrix{T}) where {T} = PSDMatrix{T,typeof(R)}(R)
PSDMatrix{T}(R::AbstractMatrix) where {T} = PSDMatrix{T,typeof(R)}(R)

# Base overloads
Matrix(M::PSDMatrix) = Matrix(M.R' * M.R)
unfactorize(M::PSDMatrix) = M.R' * M.R
size(M::PSDMatrix) = (size(M.R, 2), size(M.R, 2))
ndims(M::PSDMatrix) = 2
inv(M::PSDMatrix) = PSDMatrix(inv(M.R'))
iszero(M::PSDMatrix) = iszero(M.R)
\(A::PSDMatrix, B::AbstractVecOrMat) = A.R \ (A.R' \ B)
/(B::AbstractVecOrMat, A::PSDMatrix) = B / A.R / A.R'
/(v::LinearAlgebra.Transpose{T,<:AbstractVector} where {T}, M::PSDMatrices.PSDMatrix) =
    transpose(M \ transpose(v))
/(v::LinearAlgebra.Adjoint{T,<:AbstractVector} where {T}, M::PSDMatrices.PSDMatrix) =
    adjoint(conj(M) \ adjoint(v))
copy(M::PSDMatrix{T}) where {T} = PSDMatrix{T}(copy(M.R))
similar(M::PSDMatrix{T}, element_type::Type=eltype(M)) where {T} =
    PSDMatrix{T}(similar(M.R, element_type))
copy!(dst::PSDMatrix, src::PSDMatrix) = (copy!(dst.R, src.R); dst)
==(M1::PSDMatrix, M2::PSDMatrix) = M1.R == M2.R  # todo: same as isequal()?!
isapprox(M1::PSDMatrix, M2::PSDMatrix; kwargs...) = isapprox(M1.R, M2.R; kwargs...)
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

function logabsdet(M::PSDMatrix)
    confirm_factor_is_square(M)
    _logabsdet, _sign = logabsdet(M.R)
    return 2 * _logabsdet, _sign^2
end

function diag(M::PSDMatrix)
    out = similar(M.R, size(M.R, 2))
    sum!(abs2, out', M.R)
    return out
end

function confirm_factor_is_square(M::PSDMatrix)
    if size(M.R, 1) != size(M.R, 2)
        msg = (
            "The requested operation is not available for a PSDMatrix with a non-square factor." *
            "The factor of the received PSDMatrix has dimensions ($(size(M.R,1)), $(size(M.R,2))). " *
            "Try turning the PSDMatrix into a dense matrix first."
        )
        error(msg)
    end
end

# PDMats.jl overloads
choleskify(M::PSDMatrix) = Cholesky(nonnegative_diagonal!(qr(M.R).R), 'U', 0)
function nonnegative_diagonal!(R)
    signs = signbit.(diag(R))
    R .*= (1 .- 2 .* signs)
    return R
end
function PDMats.unwhiten!(r::AbstractVecOrMat, M::PSDMatrix, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims size(M.R, 2) == size(x, 1)
    return mul!(r, choleskify(M).L, x)
end
function PDMats.unwhiten(M::PSDMatrix, x::AbstractVecOrMat)
    PDMats.@check_argdims size(M.R, 1) == size(x, 1)
    return choleskify(M).L * x
end
function PDMats.whiten!(r::AbstractVecOrMat, M::PSDMatrix, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    PDMats.@check_argdims size(M.R, 2) == size(x, 1)
    return ldiv!(r, choleskify(M).L, x)
end
function PDMats.whiten(M::PSDMatrix, x::AbstractVecOrMat)
    PDMats.@check_argdims size(M.R, 1) == size(x, 1)
    return choleskify(M).L \ x
end

PDMats.X_A_Xt(A::PSDMatrix, X::AbstractVecOrMat) = PSDMatrix(A.R * X')
PDMats.Xt_A_X(A::PSDMatrix, X::AbstractVecOrMat) = PSDMatrix(A.R * X)
PDMats.X_invA_Xt(A::PSDMatrix, X::AbstractVecOrMat) = PSDMatrix(A.R' \ X')
PDMats.Xt_invA_X(A::PSDMatrix, X::AbstractVecOrMat) = PSDMatrix(A.R' \ X)
PDMats.quad(A::PSDMatrix, x::AbstractVector) = (z = A.R * x; z'z)
PDMats.quad(A::PSDMatrix, X::AbstractMatrix) = diag(PSDMatrix(A.R * X))
PDMats.quad!(out::AbstractArray, A::PSDMatrix, X::AbstractMatrix) = begin
    z = similar(X, size(X, 2))
    for i = 1:size(X, 2)
        mul!(z, A.R, @view X[:, i])
        out[i] = sum(abs2, z)
    end
    return out
end
PDMats.invquad(A::PSDMatrix, x::AbstractVector) = (z = A.R' \ x; z'z)
PDMats.invquad(A::PSDMatrix, X::AbstractMatrix) = diag(PSDMatrix(A.R' \ X))
PDMats.invquad!(out::AbstractArray, A::PSDMatrix, X::AbstractMatrix) = begin
    z = similar(X, size(X, 2))
    L = choleskify(A).L
    for i = 1:size(X, 2)
        ldiv!(z, L, @view X[:, i])
        out[i] = sum(abs2, z)
    end
    return out
end

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

export PSDMatrix
export add_cholesky
export add_qr
export triangularize_factor
export X_A_Xt
export X_A_Xt!

end
