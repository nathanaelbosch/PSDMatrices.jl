module MarkovKernelsExt

using LinearAlgebra
import PSDMatrices
import MarkovKernels

MarkovKernels.stein(Σ::PSDMatrices.PSDMatrix, Φ::AbstractMatrix) = begin
    X, A = Φ, Σ
    return PSDMatrices.X_A_Xt(A, X)
end

end
