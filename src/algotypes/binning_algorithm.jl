abstract type AbstractBinningAlgorithm end

struct FixedBinning <: AbstractBinningAlgorithm end
struct Sturges <: AbstractBinningAlgorithm end
struct Scott <: AbstractBinningAlgorithm end
struct Rice <: AbstractBinningAlgorithm end
struct FreedmanDiaconis <: AbstractBinningAlgorithm end
struct Wand <: AbstractBinningAlgorithm end
struct SqrtBinning <: AbstractBinningAlgorithm end


function function auto_binning(samples::DensitySampleVector; bins::AbstractBinningAlgorithm = FreedmanDiaconis())
    shape = varshape(samples)
    flat_samples = flatview(unshaped.(samples.v))
    n_params = size(flat_samples)[1]
    nt_samples = ntuple(i -> flat_samples[i,:], n_params)

    binningmode = binning_mode(bins)
    number_of_bins = _auto_binning_nbins(nt_samples, param; mode=binningmode)

end

binning_mode(ba::SqrtBinning) = :sqrt
binning_mode(ba::Sturges) = :sturges
binning_mode(ba::Scott) = :scott
binning_mode(ba::Rice) = :rice
binning_mode(ba::FreedmanDiaconis) = :fd
binning_mode(ba::Wand) = :wand

# also for prior



# From Plots.jl, original authors Oliver Schulz and Michael K. Borregaard
function _auto_binning_nbins(vs::NTuple{N,AbstractVector}, dim::Integer; mode::Symbol = :auto) where N
    max_bins = 10_000
    _cl(x) = min(ceil(Int, max(x, one(x))), max_bins)
    _iqr(v) = (q = quantile(v, 0.75) - quantile(v, 0.25); q > 0 ? q : oftype(q, 1))
    _span(v) = maximum(v) - minimum(v)

    n_samples = length(LinearIndices(first(vs)))

    # The nd estimator is the key to most automatic binning methods, and is modified for twodimensional histograms to include correlation
    nd = n_samples^(1/(2+N))
    nd = N == 2 ? min(n_samples^(1/(2+N)), nd / (1-cor(first(vs), last(vs))^2)^(3//8)) : nd # the >2-dimensional case does not have a nice solution to correlations

    v = vs[dim]

    if mode == :auto
        mode = :fd
    end

    if mode == :sqrt  # Square-root choice
        _cl(sqrt(n_samples))
    elseif mode == :sturges  # Sturges' formula
        _cl(log2(n_samples) + 1)
    elseif mode == :rice  # Rice Rule
        _cl(2 * nd)
    elseif mode == :scott  # Scott's normal reference rule
        _cl(_span(v) / (3.5 * std(v) / nd))
    elseif mode == :fd  # Freedman–Diaconis rule
        _cl(_span(v) / (2 * _iqr(v) / nd))
    elseif mode == :wand
        _cl(wand_edges(v))  # this makes this function not type stable, but the type instability does not propagate
    else
        error("Unknown auto-binning mode $mode")
    end
end
