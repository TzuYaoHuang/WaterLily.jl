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
using CSV, Tables
ENV["GKSwstype"]="nul"

default()
Plots.scalefontsizes()
default(fontfamily="Palatino",linewidth=2, framestyle=:axes, label=nothing, grid=false, tick_dir=:out, size=(900,700),right_margin=5mm,left_margin=5mm,top_margin=5mm,bottom_margin=5mm)
Plots.scalefontsizes(2.1)

δ = WaterLily.δ
inside = WaterLily.inside

function calculateDiv!(divergence,velocity)
    WaterLily.@loop divergence[I] = WaterLily.div(I,velocity) over I∈inside(divergence)
end

function calculateVort!(vorticity,velocity)
    WaterLily.@loop vorticity[I] = WaterLily.curl(3,I,velocity) over I∈inside(vorticity)
end

function generateCoord(scalarArray;normalize=1)
    Ng = size(scalarArray)
    D = length(Ng)
    cenTuple = ntuple((i) -> ((1:Ng[i]) .- 1.5)/normalize,D)
    edgTuple = ntuple((i) -> ((1:Ng[i]) .- 2.0)/normalize,D)
    limTuple = ntuple((i) -> [0,Ng[i]-2]./normalize,D)

    return cenTuple,edgTuple,limTuple
end

function stagger2centerVel!(velCen,velStag)
    N,D = WaterLily.size_u(velStag)
    for i∈1:D
        WaterLily.@loop velCen[I,i] = (velStag[I,i]+velStag[I+δ(i,I),i])/2 over I∈inside(N)
    end
end

function calculateMagnitude!(velMag,velCen)
    N,D = WaterLily.size_u(velStag)
    WaterLily.@loop velMag[I] = √sum(abs2, @views velCen[I,:]) over I∈inside(N)
end

function calculateKE!(KE, vel, vof,λρ)
    WaterLily.@loop KE[I] = WaterLily.ρkeI(I,vel,vof,λρ) over I∈inside(vof)
    return sum(KE)
end

function plotContour!(plt,xc,yc,f;clim=(0,1),levels=[0.5],color=:Black,lw=2)
    clamp!(f,clim...)
    Plots.contour!(plt,xc,yc,f',levels=levels,color=color,lw=lw)
    return plt
end

function plotContourf!(plt,xc,yc,f;clim=(-5,5),levels=60,color=:seismic,lw=0)
    clamp!(f,clim...)
    Plots.contourf!(plt,xc,yc,f',clim=clim,levels=levels,color=color,lw=lw)
    return plt
end

function organizePlot!(plt,xlim,ylim)
    Plots.plot!(plt,xlimit=xlim,ylimit=ylim,aspect_ratio=:equal)
end


function puresim!(sim,duration,recostep,cID)
    timeCurrent = WaterLily.sim_time(sim)
    timeArray = (timeCurrent+recostep):recostep:duration

    numProject = 0
    trueTime = [WaterLily.time(sim)]

    oldPStorage = sim.flow.p*0

    iTime = 0
    jldsave("JLD2/"*cID*"VelVOF_"*string(iTime)*".jld2"; u=Array(sim.flow.u), f=Array(sim.inter.f), p=Array(sim.flow.p))
    ii = 0
    # @time for timeNow in timeArray
    # while WaterLily.time(sim) < timeArray[end]*sim.L/sim.U
    #     WaterLily.mom_step!(sim.flow,sim.pois,sim.inter,sim.body)
    for timeNow in timeArray
        WaterLily.sim_step!(sim,timeNow;remeasure=false)
        iTime += 1
        jldsave("JLD2/"*cID*"VelVOF_"*string(iTime)*".jld2"; u=Array(sim.flow.u), f=Array(sim.inter.f), p=Array(sim.flow.p))
        push!(trueTime,WaterLily.time(sim))
        # if trueTime[end]*sim.U/sim.L > 1.285 break end
        @printf("tU/L=%6.3f, ΔtU/L=%.8f\n",trueTime[end]*sim.U/sim.L,sim.flow.Δt[end]*sim.U/sim.L); flush(stdout)
        try
            jldsave(
                "JLD2/"*cID*"General.jld2"; 
                trueTime,
                U=sim.U,L=sim.L,rhoRatio=sim.inter.λρ,visRatio=sim.inter.λμ,
                resFin=sim.pois.res[numProject+1:end], resIni=sim.pois.res0[numProject+1:end],
                poisN=sim.pois.n[numProject+1:end],
                dts=sim.flow.Δt,Ng=size(sim.flow.p),ke=sim.inter.ke
            )
        catch
            jldsave(
                "JLD2/"*cID*"General.jld2"; 
                trueTime,
                U=sim.U,L=sim.L,rhoRatio=sim.inter.λρ,visRatio=sim.inter.λμ,
                dts=sim.flow.Δt,Ng=size(sim.flow.p),ke=sim.inter.ke
            )
        end
    end
end

function setupCompID(prefix,N,postfix="")
    cID =  @sprintf("%sN%d%s",prefix,N,postfix)
    println("You are now running: "*cID); flush(stdout)
    return cID
end

function getFunctionName(f::Function)
    return String(Symbol(f))
end

function determineMode(args)
    simulation = false
    postProcess= false
    modeFlag = args[1]
    if modeFlag=="run"
        simulation = true
        postProcess= false
    elseif modeFlag=="pp"
        simulation = false
        postProcess= true
    elseif modeFlag=="all"
        simulation = true
        postProcess= true
    end

    return simulation,postProcess
end

function readGeneralInfo(cID)
    JLDFile = jldopen("JLD2/"*cID*"General.jld2")

    UScale = JLDFile["U"]
    LScale = JLDFile["L"]
    trueTime = JLDFile["trueTime"]; trueTime .*= UScale/LScale
    NTime = length(trueTime)
    T = eltype(trueTime)
    timeLimit = [minimum(trueTime),maximum(trueTime)]
    dtTrueTime = trueTime[2:end] .- trueTime[1:end-1]
    dtTrueTime[dtTrueTime .<= 10eps(T)] .= median(dtTrueTime)
    dts = JLDFile["dts"]; dts .*= UScale/LScale
    λρ = JLDFile["rhoRatio"]
    λμ = JLDFile["visRatio"]
    allTime = cumsum(dts)[1:end-1]
    resIni = JLDFile["resIni"]
    resFin = JLDFile["resFin"]
    poisN = JLDFile["poisN"]
    ke = JLDFile["ke"]
    Ng = JLDFile["Ng"]
    Nv = (Ng...,length(Ng))

    close(JLDFile)

    return UScale, LScale, λρ, λμ, trueTime, NTime, dtTrueTime, dts, allTime, resIni, resFin, poisN, ke, Ng, Nv
end

function readData!(cID,iTime,VOFStore,VelStore,PreStore,UScale,LScale)
    JLDFile = jldopen("JLD2/"*cID*"VelVOF_"*string(iTime-1)*".jld2")
    VelStore .= JLDFile["u"]/UScale
    VOFStore .= JLDFile["f"]
    PreStore .= JLDFile["p"]/(0.5*UScale^2)
end

function plotPoisson(cID,allTime,dts,resIni,resFin,poisN)
    Plots.plot()
    Plots.plot!(allTime,dts[1:end-1])
    Plots.savefig(cID*"_delT.png")

    Plots.plot()
    Plots.plot!(allTime,resIni[1:2:end],yscale=:log10,label="First Stage")
    Plots.plot!(allTime,resIni[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
    Plots.savefig(cID*"_PoisResInitial.png")

    Plots.plot()
    Plots.plot!(allTime,resFin[1:2:end],yscale=:log10,label="First Stage")
    Plots.plot!(allTime,resFin[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
    Plots.savefig(cID*"_PoisResFinal.png")

    Plots.plot()
    Plots.plot!(allTime,poisN[1:2:end],label="First Stage")
    Plots.plot!(allTime,poisN[2:2:end],label="Second Stage")
    Plots.savefig(cID*"_PoisNum.png")
end

function ddt(f,dt)
    return (f[2:end]-f[1:end-1])./dt
end

function midp(t)
    return (t[2:end]+t[1:end-1])/2
end

function plotKEEvole(cID,t,dt,ke)
    Plots.plot()
    Plots.plot!(t,ke)
    Plots.savefig(cID*"_KE.png")

    dkedt = ddt(ke,dt)
    midt = midp(t)

    Plots.plot()
    Plots.plot!(midt,dkedt)
    Plots.savefig(cID*"_dKEdt.png")
    CSV.write(cID*"_dKEdt.txt",  Tables.table(hcat(midt,dt,dkedt)), writeheader=false)
end

function analyze4KE(cID,ke,t,dt)
    ke1Start = ke[1:7:end]
    ke1ConDi = ke[2:7:end]
    ke1PresP = ke[3:7:end]
    ke2Start = ke[4:7:end]
    ke2ConDi = ke[5:7:end]
    ke2PresP = ke[6:7:end]
    ke2Enddd = ke[7:7:end]

    δke1Con = (ke1ConDi .- ke1Start)./dt
    δke1Pre = (ke1PresP .- ke1ConDi)./dt
    δke1all = (ke2Start .- ke1Start)./dt
    δke2Con = (ke2ConDi .- ke2Start)./dt
    δke2Pre = (ke2PresP .- ke2ConDi)./dt
    δke2all = (ke2Enddd .- ke2Start)./dt

    Plots.plot()
    Plots.plot!(t,δke1Con, label="1-ConvDiff", color=:blue, linestyle=:solid)
    Plots.plot!(t,δke1Pre, label="1-PresProj", color=:red, linestyle=:solid)
    Plots.plot!(t,δke2Con, label="2-ConvDiff", color=:blue, linestyle=:dash)
    Plots.plot!(t,δke2Pre, label="2-PresProj", color=:red, linestyle=:dash)
    Plots.plot!(t,0*t, label="",color=:black)
    Plots.savefig(cID*"_KEDiffDetail.png")

    Plots.plot()
    Plots.plot!(t,δke1all, label="1-all", color=:blue, linestyle=:solid)
    Plots.plot!(t,δke2all, label="2-all", color=:blue, linestyle=:dash)
    Plots.plot!(t,0*t, label="",color=:black)
    Plots.savefig(cID*"_KEfromDiffAll.png")
end

function exportVelVOF(cID,iTime,vof,vel,cenTuple)
    D = length(cenTuple)
    insideI = WaterLily.inside(vof)
    vtk_grid("VTK/"*cID*"VelVOF_"*string(iTime-1)*".vti", ntuple((i)->cenTuple[i][2:end-1],D)...) do vtk
        vtk["VOF"] = @views vof[insideI]
        vtk["Vel"] = @views ntuple((i)->vel[insideI,i], D)
    end
end