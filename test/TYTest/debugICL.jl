include("../../src/WaterLily.jl")
using Plots
using JLD2

inside = WaterLily.inside




N = 130
f = zeros(N,N)

fileName = "JLD2/3DNewVortexBreak128VelVOF_3090.jld2"
ff = jldopen(fileName)
f = convert(Array{Float64},(1 .-ff["f"]))
labelStorage = Array{Int}(undef,size(f)...)

# f[inside(f)] = [
#     0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.1;
#     0.0 0.2 0.2 0.0 0.0 0.2 0.1 0.0;
#     0.0 0.2 0.2 0.2 0.2 0.0 0.1 0.1;
#     0.0 0.0 0.0 0.2 0.2 0.0 0.0 0.0;
#     0.0 0.0 0.0 0.1 0.1 0.0 0.0 0.0;
#     0.0 0.1 0.1 0.0 0.0 0.2 0.2 0.0;
#     0.0 0.1 0.1 0.0 0.0 0.2 0.2 0.0;
#     0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.1
# ]
# f[inside(f)] = [
#     0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.1;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.1
# ]
# f[inside(f)] = [
#     0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.1;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.0 0.0 0.1 0.2 0.2 0.1 0.0 0.0;
#     0.0 0.0 0.2 0.2 0.2 0.2 0.0 0.0;
#     0.0 0.0 0.2 0.2 0.2 0.2 0.0 0.0;
#     0.0 0.0 0.1 0.2 0.2 0.1 0.0 0.0;
#     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#     0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.1
# ]
# for I ∈ inside(f)
#     f[I] = rand()
#     f[I] = f[I]<0.5 ? 0.0 : f[I]
# end
# WaterLily.BCPer!(f)

α = zeros(N,N,N)
n̂ = zeros(N,N,N,3)

WaterLily.vof_reconstruct!(f,α,n̂,perdir=(1,2,3))

bInfo = WaterLily.BubblesInfo(labelStorage)
@time bb = WaterLily.ICCL_M!(bInfo,f,[0.0,0.0],n̂)

# Plot
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

xlist = 0.5:1:N-2.5
X,Y = meshgrid(0.5:1:N-2.5,0.5:1:N-2.5)

Plots.plot()
Plots.heatmap!(xlist,xlist,f[65,2:end-1,2:end-1]',c=:redblue,clim=(0.0,1.0))
# Plots.quiver!(Y,X,quiver=(n̂[inside(f),1]',n̂[inside(f),2]'),color=:black,linewidth=1.5)
Plots.plot!(aspect_ratio=:equal,xlimit=(0,N-2),ylimit=(0,N-2))
savefig("ICL_originalVOF.png")

Plots.plot()
Plots.heatmap!(xlist,xlist,bb.labelField[65,2:end-1,2:end-1]',c=:glasbey_hv_n256)
# Plots.quiver!(Y,X,quiver=(n̂[inside(f),1]',n̂[inside(f),2]'),color=:black,linewidth=1.5)
Plots.plot!(aspect_ratio=:equal,xlimit=(0,N-2),ylimit=(0,N-2))
savefig("ICL_AfterLabel.png")

@time WaterLily.ProjectBubbleInfoToVOFFiled!(bInfo)

Plots.plot()
Plots.heatmap!(xlist,xlist,bb.labelField[65,2:end-1,2:end-1]',c=:glasbey_hv_n256)
# Plots.quiver!(Y,X,quiver=(n̂[inside(f),1]',n̂[inside(f),2]'),color=:black,linewidth=1.5)
Plots.plot!(aspect_ratio=:equal,xlimit=(0,N-2),ylimit=(0,N-2))
savefig("ICL_AfterLabel_Check.png")

radia = [bubble.r for (_,bubble) ∈ bInfo.bubbleDict]/8

Plots.plot()
Plots.histogram!(log10.(radia), bins=-4.4:0.2:2.0)
savefig("ICL_BubbleHistogram.png")