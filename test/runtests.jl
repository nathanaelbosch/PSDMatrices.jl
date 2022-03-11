using Test
using PSDMatrices
using LinearAlgebra


M1 = [1 1; 2 20]
M2 = [1.0 1.0; 2.0 20.0]
M3 = big.([1.0 1.0; 2.0 20.0])

@testset "$M" for M in (M1, M2, M3)

    S = PSDMatrix(M)

    @test eltype(S) == eltype(M)

    @testset "My exports" begin
        @test norm(todense(S) - M * M') == 0.0
        @test todense(X_A_Xt(S, M)) ≈ M * todense(S) * M'
        @test choleskify_factor(S).L ≈ cholesky(todense(S)).U'
        @test todense(add_cholesky(S, S)) ≈ todense(S) + todense(S)
        @test todense(add_qr(S, S)) ≈ todense(S) + todense(S)
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
    end


end
nothing
