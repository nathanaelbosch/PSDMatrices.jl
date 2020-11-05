const LT = Union{Diagonal, LowerTriangular}

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
    Q, R = qr(sqrt.(D)*U')
    # A ≈ L*L'
    return PSDMatrix(LowerTriangular(collect(R')))
end
PSDMatrix(L::LowerTriangular) = (nonnegativediag!(L); PSDMatrix(L, L * L'))

Base.size(a::PSDMatrix) = size(a.mat)
Base.getindex(a::PSDMatrix, I::Vararg{Int, N}) where {N} = getindex(a.mat, I...)
Base.copy(A::PSDMatrix) = PSDMatrix(copy(A.L), copy(A.mat))
Base.copy!(dst::PSDMatrix, src::PSDMatrix) = (dst.L .= src.L; dst.mat .= src.mat)

Base.Matrix(a::PSDMatrix) = copy(a.mat)
diag(a::PSDMatrix) = diag(a.mat)

cholesky(A::PSDMatrix) = cholesky(A.mat)

inv(a::PSDMatrix) = (Li = inv(a.L); Li'Li)
\(a::PSDMatrix, x::AbstractVecOrMat) = a.L' \ (a.L \ x)

function *(c::T, A::PSDMatrix{S}) where {S<:Real, T<:Real}
    if c < 0
        return Matrix(A)*c
    else
        return PSDMatrix(A.L * sqrt(c))
    end
end
*(A::PSDMatrix{S}, c::T) where {S<:Real, T<:Real} = c*A

function X_A_Xt(A::PSDMatrix, X::AbstractMatrix)
    _, R = qr((X*A.L)')
    return PSDMatrix(LowerTriangular(collect(R')))
end
X_A_Xt(A::PSDMatrix, X::LT) = PSDMatrix(X*A.L)


function +(A::PSDMatrix, B::PSDMatrix)
    Q, R = qr([A.L B.L]')
    return PSDMatrix(LowerTriangular(collect(R')))
end


function nonnegativediag!(L::LowerTriangular)
    signs = signbit.(diag(L))
    if !any(signs) return end
    L.data .*= (1 .- 2 .* signs)'
end
