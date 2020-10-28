using PSDMatrices
using Documenter

makedocs(;
    modules=[PSDMatrices],
    authors="Nathanael Bosch <nathanael.bosch@gmail.com> and contributors",
    repo="https://github.com/nathanaelbosch/PSDMatrices.jl/blob/{commit}{path}#L{line}",
    sitename="PSDMatrices.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nathanaelbosch.github.io/PSDMatrices.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nathanaelbosch/PSDMatrices.jl",
)
