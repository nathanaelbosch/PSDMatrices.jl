using Test
using PSDMatrices
using LinearAlgebra

M_square = [1 1; 2 20]
M_tall = [1 1; 2 20; 3 30]
M_wide = [1 1 1; 2 20 200]
eltypes = (Int64, Float64, BigFloat)

@testset "PSDMatrices.jl" begin
    @testset "eltype=$t | shape=$(size(Mbase))" for t in eltypes,
        Mbase in (M_square, M_tall, M_wide)

        M = t.(Mbase)
        S = PSDMatrix(M)
        X = rand(size(S, 1), size(S, 2))

        @testset "Base" begin
            @test eltype(S) == t
            @test size(S) == size(todense(S))
            @test S == S
            @test copy(S) == S
            @test !(copy(S) === S)
            if size(M, 1) == size(M, 2)
                @test todense(inv(S)) ≈ inv(todense(S))
            end
            @test show(S) == nothing
            @test show(stdout, MIME("text/plain"), S) == nothing
        end

        @testset "LinearAlgebra" begin
            if size(M, 1) == size(M, 2)
                @test det(S) ≈ det(todense(S))
                @test logdet(S) ≈ logdet(todense(S))
            else
                @test_throws Exception det(S) ≈ det(todense(S))  # TODO: raise better error here?! (#16)
                @test_throws Exception logdet(S) ≈ det(todense(S))  # TODO: raise better error here?! (#16)
            end
            if (size(M, 1) >= size(M, 2))
                @test S \ X ≈ todense(S) \ X
                @test X / S ≈ X / todense(S)
            end
        end

        @testset "Exports" begin
            @test norm(todense(S) - M' * M) == 0.0
            @test todense(X_A_Xt(S, X)) ≈ X * todense(S) * X'
            @test todense(X_A_Xt(A=S, X=X)) ≈ X * todense(S) * X'
            @test todense(add_qr(S, S)) ≈ todense(S) + todense(S)
            if (size(M, 1) >= size(M, 2))
                @test choleskify_factor(S).R ≈ cholesky(todense(S)).U
                @test todense(add_cholesky(S, S)) ≈ todense(S) + todense(S)
            end
        end
    end
end
nothing
