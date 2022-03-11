using Test
using PSDMatrices
using LinearAlgebra

M_square = [1 1; 2 20]
M_tall = [1 1; 2 20; 3 30]
M_wide = [1 1 1; 2 20 200]
eltypes = (Int64, Float64, BigFloat)

not_square(M) = (size(M, 1) != size(M, 2))

@testset "PSDMatrices.jl" begin
    @testset "Base shape=$size(Mbase)" for Mbase in (M_square, M_tall)
        @testset "eltype=$t" for t in eltypes
            M = t.(Mbase)
            S = PSDMatrix(M)
            X = rand(size(S, 1), size(S, 2))

            @test eltype(S) == t

            @testset "My exports" begin
                @test norm(todense(S) - M' * M) == 0.0
                @test todense(X_A_Xt(S, X)) ≈ X * todense(S) * X' skip = not_square(M)
                # todo
                @test todense(X_A_Xt(A=S, X=X)) ≈ X * todense(S) * X' skip = not_square(M)  # todo
                @test choleskify_factor(S).R ≈ cholesky(todense(S)).U skip = not_square(M)  # todo
                @test todense(add_cholesky(S, S)) ≈ todense(S) + todense(S) skip =
                    not_square(M)  # todo
                @test todense(add_qr(S, S)) ≈ todense(S) + todense(S) skip = not_square(M)  # todo
            end

            @testset "Base" begin
                @test todense(inv(S)) ≈ inv(todense(S)) skip = not_square(M)
                @test size(S) == size(todense(S))
                @test S \ M ≈ todense(S) \ M skip = not_square(M)
                @test M / S ≈ M / todense(S) skip = not_square(M)
                @test S == S
                @test copy(S) == S
                @test !(copy(S) === S)
            end

            @testset "LinearAlgebra" begin
                @test det(S) ≈ det(todense(S)) skip = not_square(M)
                @test logdet(S) ≈ logdet(todense(S)) skip = not_square(M)
            end
        end
    end
end
nothing
