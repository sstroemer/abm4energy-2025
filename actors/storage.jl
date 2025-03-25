function make_storage(prices)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        @variable(sp, p_max, SDDP.State, initial_value = 0.0)
        @variable(sp, generation[t = 1:24], SDDP.State, initial_value = 0.0)

        if stage == 1
            @variable(sp, 0 <= invest <= 100.0)
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

            SDDP.@stageobjective(sp, 100.0 * invest)
        else
            @constraint(sp, p_max.out == p_max.in)
            @constraint(sp, [t = 1:24], generation[t].out == generation[t].in)

            SDDP.parameterize(sp, prices) do ω
                SDDP.@stageobjective(sp, sum(generation[t].in * (-ω[t]) for t = 1:24))
                return nothing
            end
        end
    end

    return model
end
