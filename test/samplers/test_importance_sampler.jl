using BAT
using Test
using Plots
using Sobol
using Random, Distributions, StatsBase, IntervalSets

likelihood = params -> begin
    l = logpdf.(Normal(0.0, 1.0), params.a)
    k = logpdf.(Normal(2.0, 3.0), params.b)
    return LogDVal(l)
end

prior = BAT.NamedTupleDist(
    a = -1.0..1.0,
    b = Normal(2.0, 2.0)
)

posterior = PosteriorDensity(likelihood, prior)

n_samples = 10^4
mean_truth = [0.5 * (minimum(prior.a) + maximum(prior.a)), prior.b.μ]
std_truth = [1 / sqrt(12) * (maximum(prior.a) - minimum(prior.a)), prior.b.σ]

@testset "importance_sampler" begin
    @testset "sobol_sampler" begin

        sobol = SobolSeq([-1.0, -6.53],[1.0, 10.53])
        p = vcat([[Sobol.next!(sobol)] for i in 1:n_samples]...)

        samples_Sobol = bat_sample(posterior, n_samples, SobolSampler()).result
        logvals = BAT.density_logval(posterior, (a = samples_Sobol.v.a[1], b = samples_Sobol.v.b[1]))
        @test length(samples_Sobol) == n_samples
        @test isapprox([samples_Sobol.v.a[1], samples_Sobol.v.b[1]], p[1]; rtol = 0.05)
        @test isapprox(mean(BAT.unshaped.(samples_Sobol.v), FrequencyWeights(samples_Sobol.weight)), mean_truth; rtol = 0.05)
        @test isapprox(std(BAT.unshaped.(samples_Sobol.v), FrequencyWeights(samples_Sobol.weight)), std_truth; rtol = 0.05)
        @test isapprox(BAT.estimate_finite_bounds(prior).vol.lo, [-1.0, -6.53]; rtol = 0.05)
        @test isapprox(BAT.estimate_finite_bounds(prior).vol.hi, [1.0, 10.53]; rtol = 0.05)
        @test exp(logvals) == samples_Sobol.weight[1]
    end
    @testset "grid_sampler" begin

        samples_Grid = bat_sample(posterior, n_samples, GridSampler()).result
        logvals = BAT.density_logval(posterior, (a = samples_Grid.v.a[1], b = samples_Grid.v.b[1]))
        @test length(samples_Grid) == n_samples
        @test isapprox([samples_Grid.v.a[1], samples_Grid.v.b[1]], [-1.0, -6.53]; rtol = 0.05)
        @test isapprox(mean(BAT.unshaped.(samples_Grid.v), FrequencyWeights(samples_Grid.weight)), mean_truth; rtol = 0.05)
        @test isapprox(std(BAT.unshaped.(samples_Grid.v), FrequencyWeights(samples_Grid.weight)), std_truth; rtol = 0.05)
        @test exp(logvals) == samples_Grid.weight[1]
    end
    @testset "prior_importance_sampler" begin

        samples_PriorImportance = bat_sample(posterior, n_samples, PriorImportanceSampler()).result
        logvals = BAT.density_logval(posterior, (a = samples_PriorImportance.v.a[1], b = samples_PriorImportance.v.b[1]))
        @test length(samples_PriorImportance) == n_samples
        @test isapprox(mean(BAT.unshaped.(samples_PriorImportance.v), FrequencyWeights(samples_PriorImportance.weight)), mean_truth; rtol = 0.05)
        @test isapprox(std(BAT.unshaped.(samples_PriorImportance.v), FrequencyWeights(samples_PriorImportance.weight)), std_truth; rtol = 0.05)
        @test isapprox([minimum(samples_PriorImportance.v.a), maximum(samples_PriorImportance.v.a)], [-1.0, 1.0]; rtol = 0.05)
    end
end
