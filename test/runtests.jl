using PSDMatrices
using Test

using PSDMatrices: X_A_Xt
using LinearAlgebra

@testset "PSDMatrices.jl" begin

    d = 5
    L = LowerTriangular(randn(d,d))
    LLT = L*L'
    A = PSDMatrix(L)
    B = PSDMatrix(LowerTriangular(rand(d,d)))

    @testset "Basics" begin
        @test A == LLT
        @test all(A .== LLT)
        @test size(A) == size(LLT)
        @test A[:] == LLT[:]

        @test PSDMatrix(Matrix(A)) ≈ A

        @test copy(A) isa PSDMatrix
        copy!(B, A)
        @test B == A

        @test Matrix(A) == LLT
        @test typeof(Matrix(A)) === typeof(LLT)
    end

    @testset "LinearAlgebra" begin
        @test diag(A) == diag(LLT)
        @test det(A) ≈ det(LLT)
        @test logdet(A) ≈ logdet(LLT)

        @test inv(A) ≈ inv(LLT)
        x = rand(d)
        @test A\x ≈ LLT\x

        @test cholesky(A) == cholesky(LLT)
    end

    @testset "X_A_Xt" begin
        X = rand(d,d)
        @test X_A_Xt(A, X) isa PSDMatrix
        @test X_A_Xt(A, X) ≈ X*A*X'

        D = Diagonal(rand(d))
        @test X_A_Xt(A, D) isa PSDMatrix
        @test X_A_Xt(A, D) ≈ D*A*D'
    end

    @testset "Arithmetic" begin
        L2 = LowerTriangular(randn(d,d))
        B = PSDMatrix(L2)
        LLT2 = L2*L2'
        @test A+B isa PSDMatrix
        @test A+B ≈ LLT+LLT2

        c = rand()
        @test c*A isa PSDMatrix
        @test 0*A isa PSDMatrix
        @test !(-1*A isa PSDMatrix)
        @test c*A == A*c
    end

    @testset "BigFloats" begin
        L = LowerTriangular(big.(randn(d,d)))
        @test PSDMatrix(L) isa PSDMatrix
    end
end
