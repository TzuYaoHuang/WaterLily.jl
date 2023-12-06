# INCLUDE the WaterLily routine 
filePath = @__FILE__
workdir = dirname(filePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"
include(waterlilypath)
WaterLily = Main.WaterLily;

include("Visulize2DgVOF.jl")


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

    N = 64

    # CASE configuration
    computationID =  @sprintf("2DSinusoidalN%d",N)
    println("You are now running: "*computationID); flush(stdout)

    JLDFile = jldopen("JLD2/"*computationID*"General.jld2")

    UScale = JLDFile["U"]
    LScale = JLDFile["L"]
    trueTime = JLDFile["trueTime"]; trueTime .*= UScale/LScale
    NTime = length(trueTime)

    xcen = ((1:N+2) .- 1.5)

    close(JLDFile)

    VOFStore = zeros(N+2,N+2)
    VelocityStore = zeros(N+2,N+2,2)
    alphaStore = VOFStore*0
    nHatStore = VelocityStore*0

    animGVOF = Plots.Animation()
    @time for iTime ∈ 1:Int(NTime÷2.5)
        JLDFile = jldopen("JLD2/"*computationID*"VelVOF_"*string(iTime-1)*".jld2")
        VelocityStore .= JLDFile["u"]/UScale
        VOFStore .= JLDFile["f"]
        println(minimum(VelocityStore))
        println(maximum(VelocityStore))
        # vtk_grid("VTK/"*computationID*"VelVOF_"*string(iTime-1)*".vti", xcen, xcen, xcen) do vtk
        #     vtk["VOF"] = @views VOFStore[inside(VOFStore)]
        #     vtk["Vel"] = @views (VelocityStore[inside(VOFStore),1],VelocityStore[inside(VOFStore),2],VelocityStore[inside(VOFStore),3])
        # end
        plt = Plots.plot(size=(1200,1000))
        # Plots.contourf!(plt,xcen,xcen,VOFStore',clim=(0,1),color=:Reds_3,levels=21,lw=0)
        # Plots.contourf!(plt,xcen,xcen,real.(√dropdims(sum(abs2,VelocityStore,dims=3),dims=3)'),clim=(0,2),color=:RdYlGn_3,levels=21,lw=0)
        Plots.contourf!(plt,xcen,xcen,(VelocityStore[:,:,1].^2 .+VelocityStore[:,:,2].^2)'.^0.5,clim=(0,1.5),color=:RdYlGn_3,levels=21,lw=0)
        generateNormalInterFromF!(VOFStore,alphaStore,nHatStore)
        plotAllgVOF!(plt,VOFStore,alphaStore,nHatStore)
        frame(animGVOF,plt)
        iTime%10==1 && @printf("%04d/%04d\n", iTime, NTime)
        close(JLDFile)
    end
    Plots.gif(animGVOF,computationID*"gVOF.gif",fps=50)
end


main()



