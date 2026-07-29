module AsyncMultiObjectiveBayesOpt

include("pareto.jl")
include("surrogate.jl")
include("scheduler.jl")

using .ParetoTools: dominates, nondominated_indices, hypervolume_2d,
                    inverted_generational_distance
using .Scheduler: async_mobo

export async_mobo, dominates, nondominated_indices, hypervolume_2d,
       inverted_generational_distance

end
