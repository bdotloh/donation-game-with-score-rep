# Generate reference outputs from helper_funcs.jl for the Python engine tests.
#
#   julia tests/reference/make_reference.jl
#
# Writes stable_cases.tsv (stable score distributions, payoffs and rates for
# random one- and two-strategy populations) and ess_sweep.tsv (ESS found by
# find_ESS for many scoring norms), both with REPUTATION=:tokens.

using Random

# helper_funcs.jl only needs OrderedCollections for makeAllBB/makeAllstrategies,
# which are not used here.
const OrderedDict = Dict
const HERE = @__DIR__
src = read(joinpath(HERE, "helper_funcs.jl"), String)
include_string(Main, replace(src, "using OrderedCollections" => ""))

const HELP = ['0', 'g']
const STRATEGIES = [string(a, ',', b, '|', c, ',', d) for a in HELP for b in HELP for c in HELP for d in HELP]

# Valid :tokens rules: a player with score 0 cannot lose a score.
# Positions follow (Need's S, act, Can's S); CN: Can's S = 0 at 1,3,5,7, NC: Need's S = 0 at 1-4.
function all_codes(zero_positions)
    options = [i in zero_positions ? ['0', '+'] : ['-', '0', '+'] for i in 1:8]
    return [String(collect(c)) for c in Iterators.product(options...)][:]
end
const CN_CODES = sort(all_codes([1, 3, 5, 7]))
const NC_CODES = sort(all_codes([1, 2, 3, 4]))

interleave(w) = vcat(w[1], [x for v in w[2:end] for x in (0.0, v)])   # [w0, w-1, w1, w-2, w2, ...]
scores(w) = w[1:2:end]                                               # drop the (empty) negative bins
fmt(v) = join([repr(x) for x in v], " ")

function random_w(rng)
    n = rand(rng, 2:4)
    w = rand(rng, n)
    return w ./ sum(w)
end

function stable_cases(path; n_cases=400, seed=1)
    rng = MersenneTwister(seed)
    open(path, "w") do io
        println(io, join(["kind", "barcode", "strategy_a", "strategy_b", "rho_a", "noise", "maxn",
                          "w_init_a", "w_init_b", "w_a", "w_b", "fit_a", "fit_b",
                          "trans_a", "trans_b"], "\t"))
        for case in 1:n_cases
            barcode = "CmeetsN:" * rand(rng, CN_CODES) * " NmeetsC:" * (rand(rng) < 0.5 ? "00000000" : rand(rng, NC_CODES))
            bb = name2bigbrother(barcode)
            A, B = rand(rng, STRATEGIES), rand(rng, STRATEGIES)
            noise = rand(rng, [0.0, 0.0, 0.01, 0.1])
            maxn = rand(rng, [0, 1, 7, 100, 100, 1000])
            wA, wB = random_w(rng), random_w(rng)
            if case % 2 == 1
                w, fit, trans, _ = calc_stable_w_1strategies(name2strategy(A), bb; w_init=interleave(wA),
                                                            maxn=maxn, NOISE_VAL=noise)
                println(io, join(["one", barcode, A, "", "1.0", repr(noise), maxn, fmt(wA), "",
                                  fmt(scores(w)), "", repr(fit), "",
                                  fmt([trans["alpha"], trans["beta"], trans["gamma"]]), ""], "\t"))
            else
                rhoA = rand(rng, [0.01, 0.2, 0.5, 0.99])
                chA, chB = calc_stable_w_2strategies(name2strategy(A), name2strategy(B), bb; rhoA=rhoA,
                                                     wA_init=interleave(wA), wB_init=interleave(wB),
                                                     maxn=maxn, NOISE_VAL=noise)
                t(ch) = fmt([ch["trans"]["alpha"], ch["trans"]["beta"], ch["trans"]["gamma"]])
                println(io, join(["two", barcode, A, B, repr(rhoA), repr(noise), maxn, fmt(wA), fmt(wB),
                                  fmt(scores(chA["w"])), fmt(scores(chB["w"])), repr(chA["fit"]),
                                  repr(chB["fit"]), t(chA), t(chB)], "\t"))
            end
        end
    end
end

function ess_sweep(path; n_with_nc=300, seed=2)
    rng = MersenneTwister(seed)
    strategies = [name2strategy(s) for s in STRATEGIES]
    barcodes = ["CmeetsN:" * cn * " NmeetsC:00000000" for cn in CN_CODES]
    append!(barcodes, ["CmeetsN:" * rand(rng, CN_CODES) * " NmeetsC:" * rand(rng, NC_CODES) for _ in 1:n_with_nc])
    open(path, "w") do io
        println(io, join(["barcode", "noise", "ess"], "\t"))
        for noise in [0.0, 0.01]
            for barcode in barcodes
                ess = find_ESS(strategies, strategies, name2bigbrother(barcode); w_init=[0.5, 0.0, 0.5],
                               rhoA=0.99, threshold_fitness=1e-5, maxn=100, REPUTATION=:tokens,
                               NOISE_VAL=noise, BENEFIT=2, COST=1)
                println(io, join([barcode, repr(noise), join(ess, ";")], "\t"))
            end
        end
    end
end

stable_cases(joinpath(HERE, "stable_cases.tsv"))
ess_sweep(joinpath(HERE, "ess_sweep.tsv"))
