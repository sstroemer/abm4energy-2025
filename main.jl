using JuMP
using SDDP: SDDP
using Gurobi: Gurobi

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
CAPEX_THERMAL = 600_000 * WACC / (1 - (1 + WACC)^(-25)) * 24 / 8760
CAPEX_RES = 1_000_000 * WACC / (1 - (1 + WACC)^(-25)) * 24 / 8760
CAPEX_STORAGE = 1_000_000 * WACC / (1 - (1 + WACC)^(-15)) * 24 / 8760
VOM_RES = 5.0
VOM_THERMAL = 1.0 / 0.45 * (30.0 + 0.202 * 125.0)

include("actors/thermal.jl")
include("actors/res.jl")
include("actors/storage.jl")


prices = [ones(24) .* 100.0]
αs = [0.95 .* rand(24) for _ = 1:5]
δs = [100 .+ rand(24) .* 300 for _ = 1:5]
ubs = [[(thermal = 1e3, res = 1e3) for _ = 1:24]]

for k = 1:50
    m_thermal = make_thermal(prices, ubs)
    m_res = make_res(αs, prices, ubs)

    SDDP.train(
        m_thermal;
        print_level = 0,
        run_numerical_stability_report = false,
        stopping_rules = [SDDP.FirstStageStoppingRule(; atol = 1e-1, iterations = 5)],
    )
    SDDP.train(
        m_res;
        print_level = 0,
        run_numerical_stability_report = false,
        stopping_rules = [SDDP.FirstStageStoppingRule(; atol = 1e-1, iterations = 5)],
    )

    sim_thermal = SDDP.simulate(m_thermal, 1, [:invest]; skip_undefined_variables = true)
    sim_res = SDDP.simulate(m_res, 1, [:invest]; skip_undefined_variables = true)

    inv_thermal = sim_thermal[1][1][:invest]
    inv_res = sim_res[1][1][:invest]

    det =
        SDDP.deterministic_equivalent(make_storage(prices), () -> Gurobi.Optimizer(GRB_ENV))
    set_silent(det)
    optimize!(det)

    bids = []
    for t = 1:24
        gen = value(variable_by_name(det, "generation[$(t)]_out#1"))
        avg_price = sum([prices[i][t] for i in eachindex(prices)]) / length(prices)

        if gen < 0
            # Consumption.
            push!(
                bids,
                (gen, avg_price + reduced_cost(variable_by_name(det, "discharge[$(t)]#1"))),
            )
        elseif gen > 0
            # Generation.
            push!(
                bids,
                (gen, avg_price - reduced_cost(variable_by_name(det, "charge[$(t)]#1"))),
            )
        else
            push!(bids, (0, 0))
        end
    end

    α = αs[rand(1:5)]
    δ = δs[rand(1:5)]

    m_market = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_silent(m_market)
    @variable(m_market, award[g = [:thermal, :res, :storage, :slack], t = 1:24] >= 0)
    @constraint(m_market, [t = 1:24], award[:thermal, t] <= inv_thermal)
    @constraint(m_market, [t = 1:24], award[:res, t] <= α[t] * inv_res)
    @constraint(m_market, [t = 1:24], award[:storage, t] <= max(bids[t][1], 0.0))
    @constraint(
        m_market,
        dc[t = 1:24],
        sum(award[g, t] for g in [:thermal, :res, :storage, :slack]) == δ[t]
    )
    @objective(
        m_market,
        Min,
        sum(
            1000.0 * award[:slack, t] +
            VOM_THERMAL * award[:thermal, t] +
            VOM_RES * award[:res, t] +
            bids[t][2] * award[:storage, t] for t = 1:24
        )
    )
    optimize!(m_market)
    λ = -shadow_price.(dc)

    ub = []
    for t = 1:24
        max_res_award =
            value(award[:res, t]) +
            value(award[:thermal, t]) +
            value(award[:slack, t]) +
            (bids[t][2] > VOM_RES && bids[t][1] > 0 ? value(award[:storage, t]) : 0.0)
        max_thermal_award =
            value(award[:thermal, t]) +
            value(award[:slack, t]) +
            (bids[t][2] > VOM_THERMAL && bids[t][1] > 0 ? value(award[:storage, t]) : 0.0)
        push!(ub, (thermal = max_thermal_award, res = max_res_award))
    end

    if k == 1
        prices[1] = λ
        ubs[1] = ub
    elseif (k <= 10) && (k % 10 == 0)
        prices = [sum(prices) ./ length(prices)]
        ubs = [[
            (
                thermal = sum(el[t].thermal for el in ubs) / length(ubs),
                res = sum(el[t].res for el in ubs) / length(ubs),
            ) for t = 1:24
        ]]
    else
        push!(prices, λ)
        push!(ubs, ub)
    end

    println("$(k)  |  $(sum(λ) / 24)  |  $(inv_thermal)  |  $(inv_res)")
end
