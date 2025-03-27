function make_storage(data)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        @variable(sp, p_max, SDDP.State, initial_value = 0.0)
        @variable(sp, generation[t = 1:24], SDDP.State, initial_value = 0.0)
        @variable(sp, charge_s[t = 1:24], SDDP.State, initial_value = 0.0)
        @variable(sp, discharge_s[t = 1:24], SDDP.State, initial_value = 0.0)

        if stage == 1
            @variable(sp, 0 <= invest <= 50.0)
            @constraint(sp, p_max.out == p_max.in + invest)

            @variable(sp, charge[t = 1:24] >= 0)
            @constraint(sp, [t = 1:24], charge[t] <= invest)
            @variable(sp, discharge[t = 1:24] >= 0)
            @constraint(sp, [t = 1:24], discharge[t] <= invest)

            @variable(sp, state[t = 1:24] >= 0)
            @constraint(sp, [t = 1:24], state[t] <= 4.0 * invest)
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
                [t = 1:24],
                generation[t].out == generation[t].in + discharge[t] - charge[t]
            )
            @constraint(
                sp,
                [t = 1:24],
                charge_s[t].out == charge[t]
            )
            @constraint(
                sp,
                [t = 1:24],
                discharge_s[t].out == discharge[t]
            )

            SDDP.@stageobjective(sp, CAPEX_STORAGE * invest)
        else
            @constraint(sp, p_max.out == p_max.in)
            @constraint(sp, [t = 1:24], generation[t].out == generation[t].in)

            @variable(sp, buy[t = 1:24] >= 0)
            @variable(sp, sell[t = 1:24] >= 0)
            @variable(sp, actual_state[t = 1:24] >= 0)
            @constraint(sp, [t = 1:24], sell[t] <= discharge_s[t].in)
            @constraint(sp, [t = 1:24], buy[t] <= charge_s[t].in)
            @constraint(sp, [t = 1:24], actual_state[t] <= p_max.in)
            @constraint(
                sp,
                [t = 1:23],
                actual_state[t+1] == actual_state[t] + buy[t] * 0.95 - sell[t] / 0.95
            )
            @constraint(
                sp,
                actual_state[1] == actual_state[24] + buy[24] * 0.95 - sell[24] / 0.95
            )

            SDDP.parameterize(sp, data) do ω
                @constraint(sp, [t = 1:24], sell[t] <= ω.ub[t])
                SDDP.@stageobjective(sp, sum((sell[t] - buy[t]) * (-ω.λ[t]) for t = 1:24))
                return nothing
            end
        end
    end

    return model
end
