function make_res(αs, prices, ubs)
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

            SDDP.@stageobjective(sp, 600.0 * invest)
        else
            @constraint(sp, p_max.out == p_max.in)

            @variable(sp, generation[t = 1:24] >= 0)

            Ω = [
                (price = prices[i], ub = [el.res for el in ubs[i]], α = αs[j]) for
                i in eachindex(prices) for j in eachindex(αs)
            ]
            SDDP.parameterize(sp, Ω) do ω
                @constraint(sp, [t = 1:24], generation[t] <= ω.α[t] * p_max.in)
                @constraint(sp, [t = 1:24], generation[t] <= ω.ub[t])
                SDDP.@stageobjective(
                    sp,
                    sum(generation[t] * (VOM_RES - ω.price[t]) for t = 1:24)
                )
                return nothing
            end
        end
    end

    return model
end
