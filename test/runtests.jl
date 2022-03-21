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
            @test size(S) == size(Matrix(S))
            @test S == S
            @test copy(S) == S
            @test !(copy(S) === S)
            if size(M, 1) == size(M, 2)
                @test Matrix(inv(S)) ≈ inv(Matrix(S))
            end
            @test show(S) == nothing
            @test show(stdout, MIME("text/plain"), S) == nothing
        end

        @testset "LinearAlgebra" begin
            if size(M, 1) == size(M, 2)
                @test det(S) ≈ det(Matrix(S))
                @test logdet(S) ≈ logdet(Matrix(S))
            else
                @test_throws Exception det(S) ≈ det(Matrix(S))  # TODO: raise better error here?! (#16)
                @test_throws Exception logdet(S) ≈ det(Matrix(S))  # TODO: raise better error here?! (#16)
            end
            if (size(M, 1) >= size(M, 2))
                @test S \ X ≈ Matrix(S) \ X
                @test X / S ≈ X / Matrix(S)
            end
        end

        @testset "Exports" begin
            @test norm(Matrix(S) - M' * M) == 0.0
            @test Matrix(X_A_Xt(S, X)) ≈ X * Matrix(S) * X'
            @test Matrix(X_A_Xt(A=S, X=X)) ≈ X * Matrix(S) * X'
            @test begin
                product_eltype = typeof(one(eltype(X)) * one(eltype(S)))
                S2 = PSDMatrix(zeros(product_eltype, size(S.R)...))
                X_A_Xt!(S2, S, X)
                Matrix(S2) ≈ Matrix(X_A_Xt(S, X))
            end
            @test begin
                product_eltype = typeof(one(eltype(X)) * one(eltype(S)))
                S2 = PSDMatrix(zeros(product_eltype, size(S.R)...))
                X_A_Xt!(S2, A=S, X=X)
                Matrix(S2) ≈ Matrix(X_A_Xt(S, X))
            end
            @test Matrix(add_qr(S, S)) ≈ Matrix(S) + Matrix(S)
            if (size(M, 1) >= size(M, 2))
                @test choleskify_factor(S).R ≈ cholesky(Matrix(S)).U
                @test Matrix(add_cholesky(S, S)) ≈ Matrix(S) + Matrix(S)
            end
        end
    end
end
nothing
