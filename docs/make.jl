using Documenter
using NeoNEXUS

makedocs(;
    modules=[NeoNEXUS],
    sitename="NeoNEXUS.jl",
    format=Documenter.HTML(;
        canonical="https://ispirovjr.github.io/NeoNEXUS.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Workflow" => "workflow.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/ispirovjr/NeoNEXUS.jl",
    devbranch="main",
)
