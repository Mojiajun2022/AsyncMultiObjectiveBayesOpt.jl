using AsyncMultiObjectiveBayesOpt
using Documenter

DocMeta.setdocmeta!(
    AsyncMultiObjectiveBayesOpt,
    :DocTestSetup,
    :(using AsyncMultiObjectiveBayesOpt);
    recursive=true,
)

makedocs(
    sitename="AsyncMultiObjectiveBayesOpt.jl",
    modules=[AsyncMultiObjectiveBayesOpt],
    repo="https://github.com/Mojiajun2022/AsyncMultiObjectiveBayesOpt.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(
        canonical="https://Mojiajun2022.github.io/AsyncMultiObjectiveBayesOpt.jl",
        edit_link="main",
        prettyurls=get(ENV, "CI", "false") == "true",
        repolink="https://github.com/Mojiajun2022/AsyncMultiObjectiveBayesOpt.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Tutorial" => "tutorial.md",
        "MPI and clusters" => "mpi.md",
        "Examples" => "examples.md",
        "Benchmarks" => "benchmarks.md",
        "API reference" => "api.md",
    ],
    checkdocs=:none,
)

deploydocs(
    repo="github.com/Mojiajun2022/AsyncMultiObjectiveBayesOpt.jl.git",
    devbranch="main",
)
