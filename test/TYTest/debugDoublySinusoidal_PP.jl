# INCLUDE the WaterLily routine 
filePath = @__FILE__
workdir = dirname(filePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"
include(waterlilypath)
WaterLily = Main.WaterLily;


using Printf
using JLD2
using Plots
using Plots.PlotMeasures
using Statistics
using StatsBase
using WriteVTK
using StaticArrays
using LaTeXStrings
using Loess
ENV["GKSwstype"]="nul"

default()
Plots.scalefontsizes()
default(fontfamily="Palatino",linewidth=2, framestyle=:axes, label=nothing, grid=false, tick_dir=:out, size=(900,700),right_margin=5mm,left_margin=5mm,top_margin=5mm,bottom_margin=5mm)
Plots.scalefontsizes(2.1)


function main()
    inside = WaterLily.inside

    N = 128

    # CASE configuration
    computationID =  @sprintf("3DDoublySinusoidalN%d",N)
    println("You are now running: "*computationID); flush(stdout)

    JLDFile = jldopen("JLD2/"*computationID*"General.jld2")

    UScale = JLDFile["U"]
    LScale = JLDFile["L"]
    trueTime = JLDFile["trueTime"]; trueTime .*= UScale/LScale
    NTime = length(trueTime)

    xcen = ((2:N+1) .- 1.5)/N

    close(JLDFile)

    VOFStore = zeros(N+2,N+2,N+2)
    VelocityStore = zeros(N+2,N+2,N+2,3)

    for iTime ∈ 1:NTime
        JLDFile = jldopen("JLD2/"*computationID*"VelVOF_"*string(iTime-1)*".jld2")
        VelocityStore .= JLDFile["u"]/UScale
        VOFStore .= JLDFile["f"]
        vtk_grid("VTK/"*computationID*"VelVOF_"*string(iTime-1)*".vti", xcen, xcen, xcen) do vtk
            vtk["VOF"] = @views VOFStore[inside(VOFStore)]
            vtk["Vel"] = @views (VelocityStore[inside(VOFStore),1],VelocityStore[inside(VOFStore),2],VelocityStore[inside(VOFStore),3])
        end
        iTime%10==1 && @printf("%04d/%04d\n", iTime, NTime)
        close(JLDFile)
    end
end


main()



