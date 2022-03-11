using Test
using PSDMatrices
using LinearAlgebra

M_square = [1 1; 2 20]
M_tall = [1 1; 2 20; 3 30]
M_wide = [1 1 1; 2 20 200]
eltypes = (Int64, Float64, BigFloat)

@testset "PSDMatrices.jl" begin
    @testset "eltype=$t" for t in eltypes
        @testset "General tests: shape=$(size(Mbase))" for Mbase in
                                                           (M_square, M_tall, M_wide)
            M = t.(Mbase)
            S = PSDMatrix(M)
            X = rand(size(S, 1), size(S, 2))

            @test eltype(S) == t
            @test norm(todense(S) - M' * M) == 0.0
            @test todense(X_A_Xt(S, X)) ≈ X * todense(S) * X'
            @test todense(X_A_Xt(A=S, X=X)) ≈ X * todense(S) * X'
            @test todense(add_qr(S, S)) ≈ todense(S) + todense(S)
            @test size(S) == size(todense(S))
            @test S == S
            @test copy(S) == S
            @test !(copy(S) === S)
        end

        @testset "Square and tall factors only: shape=$(size(Mbase))" for Mbase in
                                                                          (M_square, M_tall)
            M = t.(Mbase)
            S = PSDMatrix(M)
            X = rand(size(S, 1), size(S, 2))

            @test choleskify_factor(S).R ≈ cholesky(todense(S)).U
            @test todense(add_cholesky(S, S)) ≈ todense(S) + todense(S)
            @test S \ X ≈ todense(S) \ X
            @test X / S ≈ X / todense(S)
        end

        @testset "Square factors only: shape=$(size(Mbase))" for Mbase in (M_square,)
            M = t.(M_square)
            S = PSDMatrix(M)

            @test todense(inv(S)) ≈ inv(todense(S))
            @test det(S) ≈ det(todense(S))
            @test logdet(S) ≈ logdet(todense(S))
        end
    end
end
nothing
