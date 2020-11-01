using LinearAlgebra
abstract type AbstractPSDMatrix{T<:Real} <: AbstractMatrix{T} end
struct PSDMatrix{T<:Real} <: AbstractPSDMatrix{T}
    L::LowerTriangular{T,Matrix{T}}
    mat::Matrix{T}
end
function PSDMatrix(mat::Matrix)
    if !(mat ≈ mat') error("Matrix not symmetric") end
    mat = Symmetric(mat)
    vals, U = eigen(mat)
    if any(vals .< 0) error("Matrix not positive semi definite") end
    D = Diagonal(vals)
    # A ≈ U * D * U'
    L, Q = lq(U * sqrt.(D))
    # A ≈ L*L'
    return PSDMatrix(LowerTriangular(L))
end
PSDMatrix(L::LowerTriangular) = PSDMatrix(L, L * L')

Base.size(a::PSDMatrix) = size(a.mat)
Base.getindex(a::PSDMatrix, i::Int) = getindex(a.mat, i)
Base.getindex(a::PSDMatrix, I::Vararg{Int, N}) where {N} = getindex(a.mat, I...)
Base.copy(A::PSDMatrix) = PSDMatrix(copy(A.L), copy(A.mat))
Base.copy!(dst::PSDMatrix, src::PSDMatrix) = (dst.L .= src.L; dst.mat .= src.mat)

Base.Matrix(a::PSDMatrix) = copy(a.mat)
LinearAlgebra.diag(a::PSDMatrix) = diag(a.mat)

inv(a::PSDMatrix) = (Li = inv(a.L); Li'Li)
\(a::PSDMatrix, x::AbstractVecOrMat) = a.L' \ (a.L \ x)


function X_A_Xt(A::PSDMatrix, X::AbstractMatrix)
    L, _ = lq(Matrix(X*A.L))
    L = LowerTriangular(L)
    return PSDMatrix(L)
end

function +(A::PSDMatrix, B::PSDMatrix)
    L, _ = lq([A.L B.L])
    L = LowerTriangular(L)
    return PSDMatrix(L)
end
