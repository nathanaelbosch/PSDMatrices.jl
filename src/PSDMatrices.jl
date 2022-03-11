
module PSDMatrices

import Base: \, /, size, inv, copy, ==
import LinearAlgebra: cholesky, det, logdet

struct SquareRootMatrix{T,Ltype} <: AbstractMatrix{T}
    L::Ltype
    SquareRootMatrix(L::AbstractMatrix{T}) where {T} = new{T,typeof(L)}(L)
end

include("psdmatrix.jl")


# Exports

export SquareRootMatrix
export todense
export X_A_Xt

end
