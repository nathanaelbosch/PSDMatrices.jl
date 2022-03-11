
# Custom functions

function todense(M::SquareRootMatrix)
    return M.L * M.L'
end


function X_A_Xt(A::SquareRootMatrix, X::AbstractMatrix)
    return SquareRootMatrix(X * A.L)
end


# Base overloads

size(M::SquareRootMatrix) = (size(M.L, 1), size(M.L, 1))
inv(M::SquareRootMatrix) = SquareRootMatrix(inv(M.L'))
\(A::SquareRootMatrix, B::AbstractVecOrMat) = A.L' \ (A.L \ B)
/(B::AbstractVecOrMat, A::SquareRootMatrix) = B / A.L' / A.L
copy(M::SquareRootMatrix) = SquareRootMatrix(copy(M.L))
==(M1::SquareRootMatrix, M2::SquareRootMatrix) = M1.L == M2.L  # todo: same as isequal()?!


function cholesky(M::SquareRootMatrix)
    QR = qr(M.L')
    R = triu(QR.factors)
    R_pos_diag = nonnegative_diagonal(R')
    return Cholesky(UpperTriangular(R))
end


# LinearAlgebra overloads

det(M::SquareRootMatrix) = det(M.L)^2
logdet(M::SquareRootMatrix) = 2 * logdet(M.L)


function nonnegative_diagonal(L)
    signs = signbit.(diag(L))
    if !any(signs)
        return
    end
    return L .*= (1 .- 2 .* signs)'
end