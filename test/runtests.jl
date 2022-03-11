using Test
using PSDMatrices
using LinearAlgebra

Mbase = [1 1; 2 20]
eltypes = (Int64, Float64, BigFloat)

@testset "PSDMatrices.jl" begin
    @testset "eltype=$t" for t in eltypes
        M = t.(Mbase)
        S = PSDMatrix(M)

        @test eltype(S) == eltype(M) == t

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
end
nothing
