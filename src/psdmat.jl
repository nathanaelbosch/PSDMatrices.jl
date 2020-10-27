using LinearAlgebra
abstract type AbstractPSDMat{T<:Real} <: AbstractMatrix{T} end
struct PSDMat{T<:Real,LT<:LowerTriangular{T},MT<:AbstractMatrix{T}} <: AbstractPSDMat{T}
    L::LT
    mat::MT
end
PSDMat(mat::Matrix) = error("Instantiation from Matrix is not yet implemented")
PSDMat(L::LowerTriangular) = PSDMat(L, L * L')

Base.size(a::PSDMat) = size(a.mat)
Base.getindex(a::PSDMat, i::Int) = getindex(a.mat, i)
Base.getindex(a::PSDMat, I::Vararg{Int, N}) where {N} = getindex(a.mat, I...)
Base.copy(A::PSDMat) = PSDMat(copy(A.L), copy(A.mat))
Base.copy!(dst::PSDMat, src::PSDMat) = (dst.L .= src.L; dst.mat .= src.mat)

Base.Matrix(a::PSDMat) = copy(a.mat)
LinearAlgebra.diag(a::PSDMat) = diag(a.mat)

Base.inv(a::PSDMat) = PSDMat(LowerTriangular(qr(inv(a.L)).R'))
\(a::PSDMat, x::AbstractVecOrMat) = a.L' \ (a.L \ x)


function X_A_Xt(A::PSDMat, X::AbstractMatrix)
    Q, R = qr(A.L' * X')
    L = LowerTriangular(R')
    return PSDMat(L)
end

function +(A::PSDMat, B::PSDMat)
    Q, R = qr([A.L B.L]')
    L = LowerTriangular(R')
    return PSDMat(L)
end
