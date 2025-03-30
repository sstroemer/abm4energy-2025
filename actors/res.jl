function make_res(capex, vom, data)
    model = SDDP.LinearPolicyGraph(;
        stages = 2,
        lower_bound = -1e8,
        sense = :Min,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, stage
        T = length(data[1].λ)

        @variable(sp, p_max, SDDP.State, initial_value = 0.0)

        if stage == 1
            @variable(sp, 0.0 <= invest <= 500.0)
            @constraint(sp, p_max.out == p_max.in + invest)

            SDDP.@stageobjective(sp, capex * invest)
        else
            @constraint(sp, p_max.out == p_max.in)

            @variable(sp, generation[t = 1:T] >= 0)
            availability =
                @constraint(sp, [t = 1:T], generation[t] - 1.0 * p_max.in <= 0)

            SDDP.parameterize(sp, data) do ω
                set_normalized_coefficient.(availability, p_max.in, -ω.α)
                set_upper_bound.(generation, ω.ub)
                SDDP.@stageobjective(sp, sum(generation[t] * (vom - ω.λ[t]) for t = 1:T))
                return nothing
            end
        end
    end

    return model
end
