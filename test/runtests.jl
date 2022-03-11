using Test


using PSDMatrices
using LinearAlgebra



M1 = [1 1; 2 20]
M2 = [1.0 1.0; 2.0 20.0]
M3 = big.([1.0 1.0; 2.0 20.0])


@testset "$M" for M in (M1, M2, M3)

    S = SquareRootMatrix(M)

    @test eltype(S) == eltype(M)

    @testset "My exports" begin
        @test norm(todense(S) - M * M') == 0.0

        X = M
        received = todense(X_A_Xt(S, X))
        expected = X * todense(S) * X'
        @test received ≈ expected

    end

    @testset "Base" begin
        @test todense(inv(S)) ≈ inv(todense(S))
        @test size(S) == size(todense(S))
        @test S \ M ≈ todense(S) \ M
        @test M / S ≈ M / todense(S)

        @test S == S
        @test copy(S) == S
        @test !(copy(S) === S)

    end

    @testset "LinearAlgebra" begin

        @test det(S) ≈ det(todense(S))

        @test logdet(S) ≈ logdet(todense(S))

        cholesky_dense = cholesky(todense(S)).L
        cholesky_sparse = cholesky(S).L
        @test cholesky(S).U ≈ cholesky(todense(S)).U rtol = 1e-14

    end


end
nothing
