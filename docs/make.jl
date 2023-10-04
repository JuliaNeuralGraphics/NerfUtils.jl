using NerfUtils
using Documenter

function main()
    ci = get(ENV, "CI", "") == "true"

    @info "Generating Documenter site"
    makedocs(;
        sitename="NerfUtils.jl",
        pages=[
            "Home" => "index.md",
            "Encoding" => "encoding.md",
            "MLP" => "mlp.md",
            "Rendering" => "rendering.md",
        ]
    )
    if ci
        @info "Deploying to GitHub"
        deploydocs(;
            repo="github.com/JuliaNeuralGraphics/NerfUtils.jl",
            push_preview=true)
    end
end

isinteractive() || main()
