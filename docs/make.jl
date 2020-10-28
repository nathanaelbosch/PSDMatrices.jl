using PSDMats
using Documenter

makedocs(;
    modules=[PSDMats],
    authors="Nathanael Bosch <nathanael.bosch@gmail.com> and contributors",
    repo="https://github.com/nathanaelbosch/PSDMats.jl/blob/{commit}{path}#L{line}",
    sitename="PSDMats.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nathanaelbosch.github.io/PSDMats.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nathanaelbosch/PSDMats.jl",
)
