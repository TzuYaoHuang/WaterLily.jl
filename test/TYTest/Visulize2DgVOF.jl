using Plots
using Plots.PlotMeasures
using StaticArrays
ENV["GKSwstype"]="nul"

default()
Plots.scalefontsizes()
default(fontfamily="Palatino",linewidth=2, framestyle=:axes, label=nothing, grid=false, tick_dir=:out, size=(900,700),right_margin=5mm,left_margin=5mm,top_margin=5mm,bottom_margin=5mm)
Plots.scalefontsizes(2.1)

include("../../src/WaterLily.jl")
using .WaterLily

function return2Points(f,α,n̂;offset=[0 0])
    xy = @SMatrix [0 0; 1 0; 1 1; 0 1; 0 0]
    intercept = zeros(5)
    for i ∈ 1:4
        intercept[i] = xy[i,1]*n̂[1] + xy[i,2]*n̂[2]-α
    end
    intercept[end] = intercept[1]

    pt = fill(0.0,2,2).+offset
    ipt = 0

    for i∈1:4
        cur = intercept[i]
        nex = intercept[i+1]
        if cur*nex <= 0
            cur = abs(cur)
            nex = abs(nex)
            ipt += 1
            (ipt<=2) && (pt[ipt,:] .+= (nex*xy[i,:]+cur*xy[i+1,:])/(cur+nex))
        end
    end
    return ipt, pt, intercept
end

function plotgVOF!(plt, f,α,n̂;offset=[0 0],interPointSize=1, gridPointSize=4,interLineWidth=2,interColor=[:blue, :grey])
    xy = [0 0; 1 0; 1 1; 0 1] .+ offset
    # if f==0
    #     Plots.plot!(plt, xy[:,1], xy[:,2],
    #         seriestype=:path,
    #         linestyle=:auto,
    #         lw = 0,
    #         seriescolor = interColor[2],
    #         marker = :circle,
    #         markersize = gridPointSize,
    #         markercolor = interColor[2],
    #         markerstrokecolor = interColor[2], label=false
    #     )
    # elseif f==1
    #     Plots.plot!(plt, xy[:,1], xy[:,2],
    #         seriestype=:path,
    #         linestyle=:auto,
    #         lw = 0,
    #         seriescolor = interColor[1],
    #         marker = :circle,
    #         markersize = gridPointSize,
    #         markercolor = interColor[1],
    #         markerstrokecolor = interColor[1],label=false
    #     )
    if f>0 && f<1
        ipt, pt, intercept = return2Points(f,α,n̂;offset)
        Plots.plot!(plt, pt[:,1], pt[:,2],
            seriestype=:path,
            linestyle=:solid,
            lw = interLineWidth,
            seriescolor = interColor[1],
            marker = :circle,
            markersize = interPointSize,
            markercolor = interColor[1],
            markerstrokecolor = interColor[1],label=false
        )
        # for i∈1:4
        #     colorsym = intercept[i]<0 ? interColor[1] : interColor[2]
        #     Plots.plot!(plt, @SArray[xy[i,1]],@SArray[xy[i,2]],
        #         seriestype=:path,
        #         linestyle=:auto,
        #         lw = 0,
        #         seriescolor = colorsym,
        #         marker = :circle,
        #         markersize = gridPointSize,
        #         markercolor = colorsym,
        #         markerstrokecolor = colorsym,label=false
        #     )
        # end
    end
    Plots.plot!(aspect_ratio=:equal)
end

function plotAllgVOF!(plt, f,α,n̂)
    NN = size(f)
    N = NN[1]-2
    for I∈WaterLily.inside(f)
        plotgVOF!(plt, f[I],α[I],n̂[I,:];offset=[I.I[1]-2 I.I[2]-2])
    end
    for i∈0:N
        Plots.plot!(plt,[0,N],[i,i],lw=1,color=:grey,label="")
        Plots.plot!(plt,[i,i],[0,N],lw=1,color=:grey,label="")
    end
    Plots.plot!(xlimit=[0,N],ylimit=[0,N])
end

function generateNormalInterFromF!(f,α,n̂)
    WaterLily.vof_reconstruct!(f,α,n̂)
end

function main()
    N = 4
    Ng = N+2

    f = zeros(Ng,Ng)
    α = f*0
    n̂ = zeros(Ng,Ng,2)

    f[3:4,3:4] .= .1 
    WaterLily.BCPerNeu!(f)
    generateNormalInterFromF!(f,α,n̂)
    display(f)
    display(α)
    display(n̂[:,:,1])
    display(n̂[:,:,2])

    plt = Plots.plot()
    plotAllgVOF!(plt,f,α,n̂)
    Plots.savefig("testVOF.png")

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

