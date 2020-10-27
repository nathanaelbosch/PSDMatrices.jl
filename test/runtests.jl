using PSDMats
using Test

@testset "PSDMats.jl" begin
    # Write your tests here.
    d = 5
    L = LowerTriangular(randn(d,d))
    LLT = L*L'
    A = PSDMat(L)
    @test A == LLT
    @test all(A .== LLT)
    @test size(A) == size(LLT)

    @test copy(A) isa PSDMat
    B = PSDMat(LowerTriangular(rand(d,d)))
    copy!(B, A)
    @test B == A

    @test Matrix(A) == LLT
    @test typeof(Matrix(A)) === typeof(LLT)
    @test diag(A) == diag(LLT)

    @test inv(A) ≈ inv(LLT)
    x = rand(d)
    @test A\x ≈ LLT\x

    X = rand(d,d)
    @test PSDMats.X_A_Xt(A, X) isa PSDMat
    @test PSDMats.X_A_Xt(A, X) ≈ X*A*X'

    L2 = LowerTriangular(randn(d,d))
    B = PSDMat(L2)
    LLT2 = L2*L2'
    @test A+B isa PSDMat
    @test A+B ≈ LLT+LLT2
end
