using JuMP
using SDDP: SDDP
using Gurobi: Gurobi
using CSV: CSV
using DataFrames: DataFrame


GRB_ENV = Gurobi.Env()

# ~~~~~ TE data/parameter calculation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# annuity := total * rate / (1 - (1 + rate)^(-lifetime)) * fraction
WACC = 0.07
CAPEX_THERMAL = 600_000 * (0.02 + WACC / (1 - (1 + WACC)^(-30))) * 168 / 8760
CAPEX_WIND = 1_150_000 * (0.02 + WACC / (1 - (1 + WACC)^(-30))) * 168 / 8760
CAPEX_PV = 380_000 * (0.02 + WACC / (1 - (1 + WACC)^(-20))) * 168 / 8760
CAPEX_STORAGE = 800_000 * (0.02 + WACC / (1 - (1 + WACC)^(-15))) * 168 / 8760
VOM_WIND = 5.0
VOM_PV = 2.5
VOM_THERMAL = 1.0 / 0.43 * (30.0 + 0.202 * 125.0)
VOM_SLACK = 500.0

include("actors/thermal.jl")
include("actors/res.jl")
include("actors/storage.jl")

# ~~~~~ Prepare data (time series) files ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
files = readdir("data/"; join = true)
α = [CSV.read(f, DataFrame; decimal = ',') for f in files if contains(f, "DAFTG")][1:2:end]
δ = [CSV.read(f, DataFrame; decimal = ',') for f in files if contains(f, "AL")][1:2:end]
inputs = Dict(
    :wind => [df[!, "Wind [MW]"] for df in α],
    :pv => [df[!, "Solar [MW]"] for df in α],
    :demand => [df[!, "Gesamtlast [MW]"] for df in δ],
)
limits = Dict(k => maximum(vcat(v...)) for (k, v) in inputs)
inputs = Dict(
    k => [el ./ limits[k] * (k == :demand ? 100.0 : 1.0) for el in v] for (k, v) in inputs
)

T = length(inputs[:demand][1])
data = [
    Dict(
        k => (
            α = (k in [:pv, :wind]) ? (inputs[k][1]) : nothing,
            λ = 100.0 .* ones(T),
            ub = inputs[:demand][1],
        ) for k in [:thermal, :wind, :pv, :storage]
    ),
]

# ~~~~~ Iterative solution procedure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for k = 1:200
    models = Dict(
        :thermal => make_thermal([el[:thermal] for el in data]),
        :wind => make_res(CAPEX_WIND, VOM_WIND, [el[:wind] for el in data]),
        :pv => make_res(CAPEX_PV, VOM_PV, [el[:pv] for el in data]),
    )
    invest = Dict()

    # ~~~~~ Train and evaluate investor models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (k, model) in models
        SDDP.train(
            model;
            print_level = 0,
            run_numerical_stability_report = false,
            stopping_rules = [SDDP.FirstStageStoppingRule(; atol = 1e-1, iterations = 5)],
        )

        invest[k] =
            SDDP.simulate(model, 1, [:invest]; skip_undefined_variables = true)[1][1][:invest]
    end

    # ~~~~~ Storage bid optimization using the deterministic equivalent model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    det_model_storage = SDDP.deterministic_equivalent(
        make_storage([el[:storage] for el in data]),
        () -> Gurobi.Optimizer(GRB_ENV),
    )
    set_silent(det_model_storage)
    optimize!(det_model_storage)
    invest[:storage] = value(variable_by_name(det_model_storage, "invest#1"))

    bids = []
    for t = 1:T
        gen = value(variable_by_name(det_model_storage, "generation[$(t)]_out#1"))
        avg_price = sum([data[i][:storage].λ[t] for i in eachindex(data)]) / length(data)

        if gen < 0
            # Consumption.
            push!(
                bids,
                (
                    vol = gen,
                    price = avg_price + reduced_cost(
                        variable_by_name(det_model_storage, "discharge[$(t)]#1"),
                    ),
                ),
            )
        elseif gen > 0
            # Generation.
            push!(
                bids,
                (
                    vol = gen,
                    price = avg_price - reduced_cost(
                        variable_by_name(det_model_storage, "charge[$(t)]#1"),
                    ),
                ),
            )
        else
            push!(bids, (vol = 0.0, price = 0.0))
        end
    end

    # ~~~~~ Sample drawing for market operation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n = rand(1:4)
    α = Dict(
        :thermal => ones(T),
        :wind => inputs[:wind][n],
        :pv => inputs[:pv][n],
        :storage => [
            invest[:storage] > 0 ? max(0.0, bid.vol) / invest[:storage] : 0.0 for
            bid in bids
        ],
    )
    δ = inputs[:demand][n] .+ [-min(0.0, bid.vol) for bid in bids]
    vom = Dict(
        :thermal => ones(T) .* VOM_THERMAL,
        :wind => ones(T) .* VOM_WIND,
        :pv => ones(T) .* VOM_PV,
        :storage => [bid.price for bid in bids],
        :slack => ones(T) .* VOM_SLACK,
    )

    # ~~~~~ Market clearing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    market = Model(() -> Gurobi.Optimizer(GRB_ENV))
    @variable(market, gen[g = collect(keys(vom)), t = 1:T] >= 0)
    @constraint(market, [g = collect(keys(α)), t = 1:T], gen[g, t] <= α[g][t] * invest[g])
    con_nb = @constraint(market, [t = 1:T], sum(gen[:, t]) == δ[t])
    @objective(market, Min, sum(vom[g][t] * gen[g, t] for g in keys(vom) for t = 1:T))
    set_silent(market)
    optimize!(market)
    λ = -shadow_price.(con_nb)

    # ~~~~~ Update data storage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vom[:storage] = λ
    ub = Dict(
        k => [
            sum(value(gen[j, t]) for j in keys(vom) if vom[k][t] <= vom[j][t]) for t = 1:T
        ] for k in keys(invest)
    )

    (k <= 1) && empty!(data)
    push!(
        data,
        Dict(
            k => (α = (k in [:pv, :wind]) ? α[k] : nothing, λ = λ, ub = ub[k]) for
            k in keys(invest)
        ),
    )

    println(
        "$(k)  |  $(sum(data[end][:thermal].λ)/T)  |  $(invest[:thermal])  |  $(invest[:pv])  |  $(invest[:wind])  |  $(invest[:storage])",
    )
end

# ~~~~~ Load out-of-sample test data for market simulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
files = readdir("data/"; join = true)
α = [CSV.read(f, DataFrame; decimal = ',') for f in files if contains(f, "DAFTG")][2:2:end]
δ = [CSV.read(f, DataFrame; decimal = ',') for f in files if contains(f, "AL")][2:2:end]
inputs = Dict(
    :wind => [df[!, "Wind [MW]"] for df in α],
    :pv => [df[!, "Solar [MW]"] for df in α],
    :demand => [df[!, "Gesamtlast [MW]"] for df in δ],
)
limits = Dict(k => maximum(vcat(v...)) for (k, v) in inputs)
inputs = Dict(
    k => [el ./ limits[k] * (k == :demand ? 100.0 : 1.0) for el in v] for (k, v) in inputs
)

# ~~~~~ Market simulation and some high-level KPIs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
wv = [0.0, 0.0]
for n = 1:4
    α = Dict(
        :thermal => ones(T),
        :wind => inputs[:wind][n],
        :pv => inputs[:pv][n],
        :storage => [
            invest[:storage] > 0 ? max(0.0, bid.vol) / invest[:storage] : 0.0 for
            bid in bids
        ],
    )
    δ = inputs[:demand][n] .+ [-min(0.0, bid.vol) for bid in bids]
    vom = Dict(
        :thermal => ones(T) .* VOM_THERMAL,
        :wind => ones(T) .* VOM_WIND,
        :pv => ones(T) .* VOM_PV,
        :storage => [bid.price for bid in bids],
        :slack => ones(T) .* VOM_SLACK,
    )

    market = Model(() -> Gurobi.Optimizer(GRB_ENV))

    @variable(market, gen[g = collect(keys(vom)), t = 1:T] >= 0)
    @constraint(market, [g = collect(keys(α)), t = 1:T], gen[g, t] <= α[g][t] * invest[g])

    con_nb = @constraint(market, [t = 1:T], sum(gen[:, t]) == δ[t])

    @objective(market, Min, sum(vom[g][t] * gen[g, t] for g in keys(vom) for t = 1:T))

    set_silent(market)
    optimize!(market)

    λ = -shadow_price.(con_nb)
    eens = value.(gen[:slack, :])

    println("λ (max.): $(maximum(λ))")
    println("λ (nom.): $(maximum([el for el in λ if el < VOM_SLACK]))")
    println("EENS (sum.): $(sum(eens))")
    println("EENS (max.): $(maximum(eens))")
    println("---------------------------------")

    wv[1] += sum(λ .* δ)
    wv[2] += sum(δ)
end
println(wv[1] / wv[2])
