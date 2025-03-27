function make_res(capex, vom, data)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        @variable(sp, p_max, SDDP.State, initial_value = 0.0)

        if stage == 1
            @variable(sp, 0.0 <= invest <= 100.0)
            @constraint(sp, p_max.out == p_max.in + invest)

            SDDP.@stageobjective(sp, capex * invest)
        else
            @constraint(sp, p_max.out == p_max.in)

            @variable(sp, generation[t = 1:24] >= 0)

            SDDP.parameterize(sp, data) do ω
                @constraint(sp, [t = 1:24], generation[t] <= ω.α[t] * p_max.in)
                @constraint(sp, [t = 1:24], generation[t] <= ω.ub[t])
                SDDP.@stageobjective(
                    sp,
                    sum(generation[t] * (vom - ω.λ[t]) for t = 1:24)
                )
                return nothing
            end
        end
    end

    return model
end
