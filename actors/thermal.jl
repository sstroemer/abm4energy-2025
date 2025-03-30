function make_thermal(data)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        T = length(data[1].λ)

        @variable(sp, p_max, SDDP.State, initial_value = 0.0)

        TENDER = 0 # 0.25116046660517

        if stage == 1
            @variable(sp, TENDER <= invest <= 150.0 + TENDER)
            @constraint(sp, p_max.out == p_max.in + invest)

            SDDP.@stageobjective(sp, CAPEX_THERMAL * (invest - TENDER))
        else
            @constraint(sp, p_max.out == p_max.in)
            @variable(sp, generation[t = 1:T] >= 0)
            @constraint(sp, [t = 1:T], generation[t] <= p_max.in)

            SDDP.parameterize(sp, data) do ω
                set_upper_bound.(generation, ω.ub)
                SDDP.@stageobjective(
                    sp,
                    sum(generation[t] * (VOM_THERMAL - ω.λ[t]) for t = 1:T)
                )
                return nothing
            end
        end
    end

    return model
end
