using PSDMatrices
using Documenter

makedocs(
    sitename="PSDMatrices.jl",
    modules=[PSDMatrices],
    format=Documenter.HTML(
        canonical="https://nathanaelbosch.github.io/PSDMatrices.jl/stable",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(repo="github.com/nathanaelbosch/PSDMatrices.jl", devbranch="main")
