using JuMP
using SDDP: SDDP
using Gurobi: Gurobi
using CSV: CSV
using DataFrames: DataFrame

# TODOs:
# - split into PV & Wind
# - get proper input (avail. and demand) time series data
# - implement "market simulation" afterwards, based on all avail.+demand drawings and perfect foresight (fixed capacities)
# - analyse resulting EENS and max. missing rated power
# - assume tendering a bit less than the max. missing rated power of conventional capacity
# - show (?) that that already fixes the problem (even if we did not know that it could even be enough)

GRB_ENV = Gurobi.Env()

# annuity := total * rate / (1 - (1 + rate)^(-lifetime)) * fraction
WACC = 0.07
CAPEX_THERMAL = 600_000 * (0.02 + WACC / (1 - (1 + WACC)^(-25))) * 24 / 8760
CAPEX_WIND = 1_150_000 * (0.02 + WACC / (1 - (1 + WACC)^(-30))) * 24 / 8760
CAPEX_PV = 380_000 * (0.02 + WACC / (1 - (1 + WACC)^(-15))) * 24 / 8760
CAPEX_STORAGE = 600_000 * (0.02 + WACC / (1 - (1 + WACC)^(-15))) * 24 / 8760
VOM_WIND = 4.0
VOM_PV = 1.0
VOM_THERMAL = 1.0 / 0.43 * (30.0 + 0.202 * 125.0)

include("actors/thermal.jl")
include("actors/res.jl")
include("actors/storage.jl")

dat_α = CSV.read("DAFTG_2025-03-02T23_00_00Z_2025-03-07T23_00_00Z_60M_de_2025-03-26T08_50_08Z.csv", DataFrame; decimal=',')
dat_δ = CSV.read("AL_2025-03-02T23_00_00Z_2025-03-07T23_00_00Z_60M_de_2025-03-26T09_07_51Z.csv", DataFrame; decimal=',')

αs_wind = [dat_α[(i*24 - 23):(i*24), "Wind [MW]"] ./ maximum(dat_α[!, "Wind [MW]"]) for i in 1:5]
αs_pv = [dat_α[(i*24 - 23):(i*24), "Solar [MW]"] ./ maximum(dat_α[!, "Solar [MW]"]) for i in 1:5]
δs = [dat_δ[(i*24 - 23):(i*24), "Gesamtlast [MW]"] ./ maximum(dat_δ[!, "Gesamtlast [MW]"]) .* 100 for i in 1:5]

data = [
    Dict(
        k => (
            α = (k in [:pv, :wind]) ? (k == :pv ? αs_pv[1] : αs_wind[1]) : nothing,
            λ = 100.0 .* ones(24),
            ub = δs[1],
        )
        for k in [:thermal, :wind, :pv, :storage]
    )
]

for k = 1:30
    models = Dict(
        :thermal => make_thermal([el[:thermal] for el in data]),
        :wind => make_res(CAPEX_WIND, VOM_WIND, [el[:wind] for el in data]),
        :pv => make_res(CAPEX_PV, VOM_PV, [el[:pv] for el in data]),
    )
    invest = Dict()

    for (k, model) in models
        SDDP.train(
            model;
            print_level = 0,
            run_numerical_stability_report = false,
            stopping_rules = [SDDP.FirstStageStoppingRule(; atol = 1e-1, iterations = 5)],
        )

        invest[k] = SDDP.simulate(model, 1, [:invest]; skip_undefined_variables = true)[1][1][:invest]
    end

    det_model_storage =
        SDDP.deterministic_equivalent(make_storage([el[:storage] for el in data]), () -> Gurobi.Optimizer(GRB_ENV))
    set_silent(det_model_storage)
    optimize!(det_model_storage)
    invest[:storage] = value(variable_by_name(det_model_storage, "invest#1"))

    bids = []
    for t = 1:24
        gen = value(variable_by_name(det_model_storage, "generation[$(t)]_out#1"))
        avg_price = sum([data[i][:storage].λ[t] for i in eachindex(data)]) / length(data)

        if gen < 0
            # Consumption.
            push!(
                bids,
                (vol=gen, price=avg_price + reduced_cost(variable_by_name(det_model_storage, "discharge[$(t)]#1"))),
            )
        elseif gen > 0
            # Generation.
            push!(
                bids,
                (vol=gen, price=avg_price - reduced_cost(variable_by_name(det_model_storage, "charge[$(t)]#1"))),
            )
        else
            push!(bids, (vol=0, price=0))
        end
    end

    n = rand(1:5)
    α = Dict(:thermal => ones(24), :wind => αs_wind[n], :pv => αs_pv[n])
    δ = δs[n]
    vom = Dict(:thermal => ones(24) .* VOM_THERMAL, :wind => ones(24) .* VOM_WIND, :pv => ones(24) .* VOM_PV, :storage => [bid.price for bid in bids], :slack => ones(24) .* 4000)

    market = Model(() -> Gurobi.Optimizer(GRB_ENV))
    @variable(market, award[g = vcat(collect(keys(invest)), :slack), t = 1:24] >= 0)
    @constraint(market, [g = collect(keys(α)), t = 1:24], award[g, t] <= α[g][t] * invest[g])
    @constraint(market, [t = 1:24], award[:storage, t] <= max(bids[t].vol, 0.0))
    con_nb = @constraint(market, [t = 1:24], sum(award[:, t]) == δ[t] + max(-bids[t].vol, 0.0))
    @objective(
        market,
        Min,
        sum(vom[g][t] * award[g, t] for g in keys(vom) for t in 1:24)
    )
    set_silent(market)
    optimize!(market)
    λ = -shadow_price.(con_nb)

    vom[:storage] = λ
    ub = Dict(k => [sum(value(award[j, t]) for j in keys(vom) if vom[k][t] <= vom[j][t]) for t = 1:24] for k in keys(invest))

    (k == 1) && empty!(data)
    push!(
        data,
        Dict(
            k => (
                α = (k in [:pv, :wind]) ? α[k] : nothing,
                λ = λ,
                ub = ub[k],
            )
            for k in keys(invest)
        )
    )

    println("$(k)  |  $(sum(data[end][:thermal].λ)/24)  |  $(invest[:thermal])  |  $(invest[:pv])  |  $(invest[:wind])  |  $(invest[:storage])")
end


for n in 1:5
    α = Dict(:thermal => ones(24), :wind => αs_wind[n], :pv => αs_pv[n])
    δ = δs[n]
    vom = Dict(:thermal => ones(24) .* VOM_THERMAL, :wind => ones(24) .* VOM_WIND, :pv => ones(24) .* VOM_PV, :slack => ones(24) .* 3000)

    market = Model(() -> Gurobi.Optimizer(GRB_ENV))

    @variable(market, gen[g = collect(keys(vom)), t = 1:24] >= 0)
    @constraint(market, [g = collect(keys(α)), t = 1:24], gen[g, t] <= α[g][t] * invest[g])

    con_nb = @constraint(market, [t = 1:24], sum(gen[:, t]) + bids[t].vol == δ[t])

    @objective(
        market,
        Min,
        sum(vom[g][t] * gen[g, t] for g in keys(vom) for t in 1:24)
    )

    set_silent(market)
    optimize!(market)

    λ = -shadow_price.(con_nb)

    eens = value.(gen[:slack, :])
    println("$(sum(eens) / 24)  |  $(maximum(eens))")
end

