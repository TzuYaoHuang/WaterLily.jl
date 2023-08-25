include("../../src/WaterLily.jl")
WaterLily = Main.WaterLily;
using Plots; gr()
using StaticArrays
using JLD
using Images
using FFTW
using Statistics
using Interpolations
using DelimitedFiles
using LinearAlgebra
using PyPlot
using GLMakie
GLMakie.activate!()
using PlotlyJS

inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

computationID = "DamBreakHeun"
N = 96
lx = ((1:N).-0.5)/N
ly = ((1:N).-0.5)/N

function hydroDynP!(sim)
    ρ = sim.inter.f[2:end-1,2:end-1,2:end-1]*(1-sim.inter.λρ).+sim.inter.λρ
    cumsumρ = cumsum(ρ,dims=3)
    sumρ = sum(ρ,dims=3)
    cumsumρ = sumρ .- cumsumρ 
    sim.flow.σ[2:end-1,2:end-1,2:end-1] .= ((sim.flow.p[2:end-1,2:end-1,2:end-1]+sim.flow.p[2:end-1,2:end-1,1:end-2])/2 + cumsumρ*sim.flow.g[3])/(0.5sim.U^2)
end

function flood(f::Array;shift=(0.,0.),cfill=:RdBu_11,clims=(),levels=10,kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],f',
        linewidth=0, levels=levels, color=cfill, clims = clims, 
        aspect_ratio=:equal; kv...)
    Plots.contour!(axes(f,1).+shift[1],axes(f,2).+shift[2],f',
        linewidth=2, levels=[0.5], color=:black)
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ[R]';levels,lines)
end

function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)
    t₀ = round(WaterLily.sim_time(sim))
    for I∈inside(sim.flow.σ)
        sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    end
    diver = [Statistics.sum(abs.(sim.flow.σ[R]))]
    mass = [Statistics.mean(sim.inter.f[R])]
    maxU = [maximum(sqrt.(Statistics.sum(sim.flow.u.^2,dims=4)))]
    @time anim = @animate for tᵢ in range(t₀,t₀+duration;step)
        try
            WaterLily.sim_step!(sim,tᵢ;remeasure)
        catch y
            println(y)
            return diver,mass,maxU
        end
        for I∈inside(sim.flow.σ)
            sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        end
        push!(diver,maximum(abs.(sim.inter.f[R].*sim.flow.σ[R])))
        push!(mass,Statistics.mean(sim.inter.f[R]))
        push!(maxU,maximum(sqrt.(Statistics.sum(sim.flow.u.^2,dims=4))))
        ff = sim.inter.f[R]
        hydroDynP!(sim)
        Plots.contourf(lx,ly.-0.5/N,clamp.(sim.flow.σ[4,2:end-1,2:end-1]',-1.5,1.5), aspect_ratio=:equal,color=:seismic,levels=60,xlimit=[0,1],ylimit=[0,1],linewidth=0,clim=(-1.5,1.5))
        Plots.contour!(lx,ly,sim.inter.f[4,2:end-1,2:end-1]', aspect_ratio=:equal,color=:Black,levels=[0.5],xlimit=[0,1],ylimit=[0,1],linewidth=2)
        plotbody && body_plot!(sim)
        verbose && println("tU/L=",round(tᵢ,digits=4),
            ", Δt=",sim.flow.Δt[end])
    end
    gif(anim, computationID*"_hydroDyn.gif", fps = 30)
    return diver,mass,maxU
end

function damBreak(LDomain;Re=1000,Fr=1,g=9.81)
    N = (8,LDomain,LDomain)
    H = LDomain/2
    LScale = H
    UScale = sqrt(g*LScale)*Fr
    ν = UScale*LScale/Re

    function interSDF(xx)
        x,y,z = @. xx-1.5
        if y<=H && z <= H
            return max(y-H,z-H)
        elseif y<=H && z > H
            return z-H
        elseif y>H && z <= H
            return y-H
        elseif y>H && z > H
            return sqrt((z-H)^2+(y-H)^2)
        end
    end

    return WaterLily.TwoPhaseSimulation(N, (0,0,0), LScale;U=UScale, Δt=0.01,grav=(0,0,-g), ν=ν, InterfaceSDF=interSDF, T=Float64)
end


sim = damBreak(N)

diver,mass,maxU = sim_gif!(sim,duration=15, step=0.01,clims=(0,1),plotbody=false,verbose=true,levels=0.0:0.05:1.0,remeasure=false,cfill=:RdBu,linewidth=2,xlimit=[0,32],ylimit=[0,32],shift=(-0.5,-0.5));

massrel = abs.((mass.-mass[1])/mass[1]).+1e-20
Plots.plot(massrel,yaxis=:log10,label="mass",color=:blue)
Plots.plot!(diver.+1e-20,yaxis=:log10,label="divergence",ylimit=[1e-10,1e0],color=:red)
Plots.savefig(computationID*"_MassDivergence.png")

Plots.plot(maxU)
Plots.savefig(computationID*"_MaxU.png")

aa = cumsum(sim.flow.Δt)[1:end-1]*sim.U/sim.L
Plots.plot(sim.flow.Δt,ylimit=(0,0.05))
Plots.savefig(computationID*"_delT.png")

Plots.plot(aa,sim.pois.res[1:2:end],yscale=:log10,ylimit=[1e-6,2e-3],label="First Stage")
Plots.plot!(aa,sim.pois.res[2:2:end],yscale=:log10,ylimit=[1e-6,2e-3],label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisRes.png")

Plots.plot(aa,sim.pois.n[1:2:end],label="First Stage")
Plots.plot!(aa,sim.pois.n[2:2:end],label="Second Stage")
Plots.savefig(computationID*"_PoisNum.png")


