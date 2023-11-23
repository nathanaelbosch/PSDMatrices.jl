using Test
using PSDMatrices
using LinearAlgebra
using Suppressor
using Aqua, JET
using PDMats

M_square = [1 1; 2 20]
M_tall = [1 1; 2 20; 3 30]
M_wide = [1 1 1; 2 20 200]
M_neg = -ones(1, 1)
# eltypes = (Int64, Float64, BigFloat)
eltypes = (Float64, )
# sizes = (M_square, M_tall, M_wide, M_neg)
sizes = (M_square,)

@testset "PSDMatrices.jl" begin
    @testset "eltype=$t | shape=$(size(Mbase))" for t in eltypes, Mbase in sizes

        M = t.(Mbase)
        S = PSDMatrix(M)
        SM = M'M
        X = rand(t, size(S, 1), size(S, 2))

        @testset "Base" begin
            @test eltype(S) == t
            @test size(S) == size(Matrix(S))
            @test S == S
            @test S ≈ S
            @test copy(S) == S
            @test !(copy(S) === S)
            @test typeof(similar(S)) == typeof(S)
            @test (S2 = similar(S); copy!(S2, S); S2 == S)
            if size(M, 1) == size(M, 2)
                @test Matrix(inv(S)) ≈ inv(Matrix(S))
            end
            @suppress_out @test isnothing(show(S))
            @suppress_out @test isnothing(show(stdout, MIME("text/plain"), S))
            @test iszero(S) == iszero(S.R)
        end

        @testset "LinearAlgebra" begin
            @test diag(S) ≈ diag(Matrix(S))
            if size(M, 1) == size(M, 2)
                @test det(S) ≈ det(Matrix(S))
                @test logdet(S) ≈ logdet(Matrix(S))
            else
                @test_throws MethodError det(S)
                @test_throws MethodError logdet(S)
            end
            if (size(M, 1) >= size(M, 2))
                @test S \ X ≈ Matrix(S) \ X
                @test X / S ≈ X / Matrix(S)

                v = rand(t, size(S, 2))
                @test transpose(v) / S ≈ transpose(v) / Matrix(S)
                @test adjoint(v) / S ≈ adjoint(v) / Matrix(S)
            end
        end

        @testset "Exports" begin
            @test norm(Matrix(S) - M' * M) == 0.0
            @test Matrix(X_A_Xt(S, X)) ≈ X * Matrix(S) * X'
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
                @test Matrix(add_cholesky(S, S)) ≈ Matrix(S) + Matrix(S)
                tri = triangularize_factor(S)
                @test tri.R isa UpperTriangular
                @test Matrix(tri) ≈ Matrix(S)
            end
        end

        if size(S.R, 1) == size(S.R, 2)
        @testset "PDMats.jl interface" begin
            i = 1
            x = rand(t, size(S, 2), size(S, 2))
            r = similar(x)
            c = rand(t)
            @test size(S) == size(SM)
            @test size(S, i) == size(SM, i)
            @test ndims(S) == ndims(SM)
            @test_nowarn eltype(S)
            @test_nowarn Matrix(S)
            @test diag(S) ≈ diag(SM)
            @test Matrix(inv(S)) ≈ inv(SM)
            @test_broken eigmax(S)
            @test_broken eigmin(S)
            @test logdet(S) ≈ logdet(SM)
            @test_broken S * x
            @test S \ x ≈ SM \ x
            @test_broken S * c
            @test_broken c * S
            @test_broken S + b
            # @test_nowarn pdadd(S, b, c)
            # @test_nowarn pdadd(m, S)
            # @test_nowarn pdadd(m, S, c)
            # @test_nowarn pdadd!(m, S)
            # @test_nowarn pdadd!(m, S, c)
            # @test_nowarn pdadd!(r, m, S)
            # @test_nowarn pdadd!(r, m, S, c)
            # @test_nowarn quad(S, x)
            # @test_nowarn quad!(copy(S), S, x)
            # @test_nowarn invquad(S, x)
            # @test_nowarn invquad!(r, S, x)
            @test Matrix(X_A_Xt(S, x')) ≈ x' * SM * x
            @test Matrix(Xt_A_X(S, x)) ≈ x' * SM * x
            @test Matrix(X_invA_Xt(S, x')) ≈ x' * inv(SM) * x
            @test Matrix(Xt_invA_X(S, x)) ≈ x' * inv(SM) * x
            @test whiten(S, x) ≈ whiten(SM, x)
            @test whiten!(S, copy(x)) ≈ whiten(SM, x)
            @test whiten!(r, S, x) ≈ whiten(SM, x)
            @test unwhiten(S, x) ≈ unwhiten(SM, x)
            @test unwhiten!(S, copy(x)) ≈ unwhiten(SM, x)
            @test unwhiten!(r, S, x) ≈ unwhiten(SM, x)
        end
        end
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(PSDMatrices)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(
            PSDMatrices;
            target_defined_modules=true,
        )
    end
end
nothing
