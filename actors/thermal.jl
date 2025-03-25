function make_thermal(prices, ubs)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        @variable(sp, p_max, SDDP.State, initial_value = 0.0)

        if stage == 1
            @variable(sp, 0 <= invest <= 500.0)
            @constraint(sp, p_max.out == p_max.in + invest)

            SDDP.@stageobjective(sp, 1000.0 * invest)
        else
            @constraint(sp, p_max.out == p_max.in)

            @variable(sp, generation[t = 1:24] >= 0)
            @constraint(sp, [t = 1:24], generation[t] <= p_max.in)

            Ω = [
                (price = prices[i], ub = [el.thermal for el in ubs[i]]) for
                i in eachindex(prices)
            ]
            SDDP.parameterize(sp, Ω) do ω
                @constraint(sp, [t = 1:24], generation[t] <= ω.ub[t])
                SDDP.@stageobjective(
                    sp,
                    sum(generation[t] * (VOM_THERMAL - ω.price[t]) for t = 1:24)
                )
                return nothing
            end
        end
    end

    return model
end
