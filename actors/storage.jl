function make_storage(data)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        T = length(data[1].λ)

        @variable(sp, p_max, SDDP.State, initial_value = 0.0)
        @variable(sp, generation[t = 1:T], SDDP.State, initial_value = 0.0)

        if stage == 1
            @variable(sp, 0 <= invest <= 100.0)
            @constraint(sp, p_max.out == p_max.in + invest)

            @variable(sp, charge[t = 1:T] >= 0)
            @constraint(sp, [t = 1:T], charge[t] <= invest)
            @variable(sp, discharge[t = 1:T] >= 0)
            @constraint(sp, [t = 1:T], discharge[t] <= invest)

            @variable(sp, state[t = 1:T] >= 0)
            @constraint(sp, [t = 1:T], state[t] <= 8.0 * invest)
            @constraint(
                sp,
                [t = 1:23],
                state[t+1] == state[t] + charge[t] * 0.95 - discharge[t] / 0.95
            )
            @constraint(
                sp,
                state[1] == state[24] + charge[24] * 0.95 - discharge[24] / 0.95
            )

            @constraint(
                sp,
                [t = 1:T],
                generation[t].out == generation[t].in + discharge[t] - charge[t]
            )

            SDDP.@stageobjective(sp, CAPEX_STORAGE * invest)
        else
            @constraint(sp, p_max.out == p_max.in)
            @constraint(sp, [t = 1:T], generation[t].out == generation[t].in)

            @variable(sp, charge[t = 1:T] >= 0)
            @variable(sp, discharge[t = 1:T] >= 0)
            @constraint(sp, [t = 1:T], generation[t].in == discharge[t] - charge[t])

            SDDP.parameterize(sp, data) do ω
                set_upper_bound.(discharge, ω.ub)
                SDDP.@stageobjective(
                    sp,
                    sum((discharge[t] - charge[t]) * (-ω.λ[t]) for t = 1:T)
                )
                return nothing
            end
        end
    end

    return model
end
