using LinearAlgebra
abstract type AbstractPSDMatrix{T<:Real} <: AbstractMatrix{T} end
struct PSDMatrix{T<:Real} <: AbstractPSDMatrix{T}
    L::LowerTriangular{T,Matrix{T}}
    mat::Matrix{T}
end
PSDMatrix(mat::Matrix) = error("Instantiation from Matrix is not yet implemented")
PSDMatrix(L::LowerTriangular) = PSDMatrix(L, L * L')

Base.size(a::PSDMatrix) = size(a.mat)
Base.getindex(a::PSDMatrix, i::Int) = getindex(a.mat, i)
Base.getindex(a::PSDMatrix, I::Vararg{Int, N}) where {N} = getindex(a.mat, I...)
Base.copy(A::PSDMatrix) = PSDMatrix(copy(A.L), copy(A.mat))
Base.copy!(dst::PSDMatrix, src::PSDMatrix) = (dst.L .= src.L; dst.mat .= src.mat)

Base.Matrix(a::PSDMatrix) = copy(a.mat)
LinearAlgebra.diag(a::PSDMatrix) = diag(a.mat)

Base.inv(a::PSDMatrix) = PSDMatrix(LowerTriangular(qr(inv(a.L)).R'))
\(a::PSDMatrix, x::AbstractVecOrMat) = a.L' \ (a.L \ x)


function X_A_Xt(A::PSDMatrix, X::AbstractMatrix)
    Q, R = qr(A.L' * X')
    L = LowerTriangular(R')
    return PSDMatrix(L)
end

function +(A::PSDMatrix, B::PSDMatrix)
    Q, R = qr([A.L B.L]')
    L = LowerTriangular(R')
    return PSDMatrix(L)
end
