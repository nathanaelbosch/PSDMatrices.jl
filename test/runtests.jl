using Test
using PSDMatrices
using LinearAlgebra

M_square = [1 1; 2 20]
M_tall = [1 1; 2 20; 3 30]
M_wide = [1 1 1; 2 20 200]
eltypes = (Int64, Float64, BigFloat)

@testset "PSDMatrices.jl" begin
    @testset "Base shape=$size(Mbase)" for Mbase in (M_square, M_tall)
        @testset "eltype=$t" for t in eltypes
            M = t.(Mbase)
            S = PSDMatrix(M)

            @test eltype(S) == t

            @testset "My exports" begin
                @test norm(todense(S) - M * M') == 0.0
                @test todense(X_A_Xt(S, M)) ≈ M * todense(S) * M' skip =
                    (size(M, 1) != size(M, 2))  # todo
                @test todense(X_A_Xt(A=S, X=M)) ≈ M * todense(S) * M' skip =
                    (size(M, 1) != size(M, 2))  # todo
                @test choleskify_factor(S).L ≈ cholesky(todense(S)).U' skip =
                    (size(M, 1) != size(M, 2))  # todo
                @test todense(add_cholesky(S, S)) ≈ todense(S) + todense(S) skip =
                    (size(M, 1) != size(M, 2))  # todo
                @test todense(add_qr(S, S)) ≈ todense(S) + todense(S) skip =
                    (size(M, 1) != size(M, 2))  # todo
            end

            @testset "Base" begin
                @test todense(inv(S)) ≈ inv(todense(S)) skip = (size(M, 1) != size(M, 2))
                @test size(S) == size(todense(S))
                @test S \ M ≈ todense(S) \ M skip = (size(M, 1) != size(M, 2))
                @test M / S ≈ M / todense(S) skip = (size(M, 1) != size(M, 2))
                @test S == S
                @test copy(S) == S
                @test !(copy(S) === S)
            end

            @testset "LinearAlgebra" begin
                @test det(S) ≈ det(todense(S)) skip = (size(M, 1) != size(M, 2))
                @test logdet(S) ≈ logdet(todense(S)) skip = (size(M, 1) != size(M, 2))
            end
        end
    end
end
nothing
