using PSDMatrices
using Test

using PSDMatrices: X_A_Xt
using LinearAlgebra

@testset "PSDMatrices.jl" begin

    d = 5
    L = LowerTriangular(randn(d,d))
    LLT = L*L'
    A = PSDMatrix(L)
    @test A == LLT
    @test all(A .== LLT)
    @test size(A) == size(LLT)
    @test A[:] == LLT[:]

    @test PSDMatrix(Matrix(A)) ≈ A

    @test copy(A) isa PSDMatrix
    B = PSDMatrix(LowerTriangular(rand(d,d)))
    copy!(B, A)
    @test B == A

    @test Matrix(A) == LLT
    @test typeof(Matrix(A)) === typeof(LLT)
    @test diag(A) == diag(LLT)

    @test inv(A) ≈ inv(LLT)
    x = rand(d)
    @test A\x ≈ LLT\x

    X = rand(d,d)
    @test X_A_Xt(A, X) isa PSDMatrix
    @test X_A_Xt(A, X) ≈ X*A*X'

    D = Diagonal(rand(d))
    @test X_A_Xt(A, D) isa PSDMatrix
    @test X_A_Xt(A, D) ≈ D*A*D'

    L2 = LowerTriangular(randn(d,d))
    B = PSDMatrix(L2)
    LLT2 = L2*L2'
    @test A+B isa PSDMatrix
    @test A+B ≈ LLT+LLT2

    @test cholesky(A) == cholesky(LLT)

    c = rand()
    @test c*A isa PSDMatrix
    @test 0*A isa PSDMatrix
    @test !(-1*A isa PSDMatrix)
    @test c*A == A*c
end
