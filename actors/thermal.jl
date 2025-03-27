function make_thermal(data)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        @variable(sp, p_max, SDDP.State, initial_value = 0.0)

        TENDER = 30.0

        if stage == 1
            @variable(sp, TENDER <= invest <= 150.0 + TENDER)
            @constraint(sp, p_max.out == p_max.in + invest)

            SDDP.@stageobjective(sp, CAPEX_THERMAL * (invest-TENDER))
        else
            @constraint(sp, p_max.out == p_max.in)

            @variable(sp, generation[t = 1:24] >= 0)
            @constraint(sp, [t = 1:24], generation[t] <= p_max.in)

            SDDP.parameterize(sp, data) do ω
                @constraint(sp, [t = 1:24], generation[t] <= ω.ub[t])
                SDDP.@stageobjective(
                    sp,
                    sum(generation[t] * (VOM_THERMAL - ω.λ[t]) for t = 1:24)
                )
                return nothing
            end
        end
    end

    return model
end
