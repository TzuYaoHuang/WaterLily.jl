filePath = @__FILE__
workdir = dirname(filePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"

include(waterlilypath)
WaterLily = Main.WaterLily;
using Plots; gr()
using StaticArrays
using JLD2
using Images
using FFTW
using Statistics
using Interpolations
using DelimitedFiles
using LinearAlgebra
using PyPlot
using GLMakie
# GLMakie.activate!()
using PlotlyJS
using Tables
using CSV

inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

N = 64
grav=1
computationID = "BubbleMove"*string(N)

lx = ((1:N).-0.5)/N
ly = ((1:N).-0.5)/N

function hydroDynP!(sim)
    ρ = sim.inter.f[2:end-1,2:end-1]*(1-sim.inter.λρ).+sim.inter.λρ
    cumsumρ = cumsum(ρ,dims=2)
    sumρ = sum(ρ,dims=2)
    cumsumρ = sumρ .- cumsumρ 
    sim.flow.σ[2:end-1,2:end-1] .= ((sim.flow.p[2:end-1,2:end-1]+sim.flow.p[2:end-1,1:end-2])/2 + cumsumρ*sim.flow.g[2])/(0.5sim.U^2)
end

function KE(u,f,λρ)
    N,n = WaterLily.size_u(u)
    ke = zeros(eltype(f),2)
    for i∈1:n
        for I ∈ WaterLily.inside_uWB(N,i)
            buf = u[I,i]^2
            fWater = WaterLily.ϕ(i,I,f)
            if (I[i] == 2) || (I[i] == N[i])
                ke[1] += 1*fWater*buf*0.5
                ke[2] += λρ*(1-fWater)*buf*0.5
            else
                ke[1] += 1*fWater*buf
                ke[2] += λρ*(1-fWater)*buf
            end
        end
    end
    return ke/2
end

function PE(f,λρ,g,gravdir)
    pe = zeros(eltype(f),2)
    for I ∈ WaterLily.inside(f)
        fWater = WaterLily.ϕ(0,I,f)
        gh = g*(I[gravdir]-0.5)
        pe[1] += 1*fWater*gh
        pe[2] += λρ*(1-fWater)*gh
    end
    return pe
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
    ke = KE(sim.flow.u,sim.inter.f,sim.inter.λρ)
    pe = PE(sim.inter.f,sim.inter.λρ,grav,2)
    diver = [Statistics.sum(abs.(sim.flow.σ[R]))]
    mass = [Statistics.mean(sim.inter.f[R])]
    maxU = [maximum(sqrt.(Statistics.sum(sim.flow.u.^2,dims=3)))]
    trueTime = [WaterLily.time(sim)]
    @time anim = @animate for tᵢ in range(t₀,t₀+duration;step)
    # @time for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ;remeasure)
        for I∈inside(sim.flow.σ)
            sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        end
        push!(diver,maximum(abs.(sim.inter.f[R].*sim.flow.σ[R])))
        push!(mass,Statistics.mean(sim.inter.f[R]))
        push!(maxU,maximum(sqrt.(Statistics.sum(sim.flow.u.^2,dims=3))))
        push!(trueTime,WaterLily.time(sim))
        ke = cat(ke,KE(sim.flow.u,sim.inter.f,sim.inter.λρ),dims=2)
        pe = cat(pe,PE(sim.inter.f,sim.inter.λρ,grav,2),dims=2)
        for I∈inside(sim.flow.σ)
            sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        end
        Plots.plot()
        Plots.contourf(lx,ly,clamp.(sim.flow.p[2:end-1,2:end-1]'/(grav*sim.L),-1,1), aspect_ratio=:equal,color=:seismic,levels=60,xlimit=[0,1],ylimit=[0,1],linewidth=0,clim=(-1,1))
        # Plots.contourf(lx.-0.5/N,ly.-0.5/N,clamp.(sim.flow.σ[2:end-1,2:end-1]',-50,50), aspect_ratio=:equal,color=:seismic,levels=60,xlimit=[0,1],ylimit=[0,1],linewidth=0,clim=(-50,50))
        Plots.contour!(lx,ly,sim.inter.f[2:end-1,2:end-1]', aspect_ratio=:equal,color=:Black,levels=[0.5],xlimit=[0,1],ylimit=[0,1],linewidth=2)
        plotbody && body_plot!(sim)
        verbose && println("tU/L=",round(tᵢ,digits=4),
            ", Δt=",sim.flow.Δt[end])
    end
    gif(anim, computationID*"_Velocity.gif", fps = 30)
    return diver,mass,maxU,trueTime,ke,pe
end

function MovingBubble(NN;Re=500,g=9.81)
    LDomain = (NN,NN)
    R = NN/4
    LScale = R
    UScale = sqrt(g*LScale)
    ν = UScale*LScale/Re

    function interSDF(xx)
        x,y = @. xx-1.5
        return ((x-NN/2)^2+(y-NN/2)^2)^0.5 - R
    end

    return WaterLily.TwoPhaseSimulation(LDomain, (0,0), LScale;U=UScale, Δt=0.01,grav=(0,-g), ν=ν, InterfaceSDF=interSDF, T=Float32,λμ=1e-3,λρ=1e-3,perdir=(1,),uλ=(i,x) -> ifelse(i==1,UScale,0))
end


sim = MovingBubble(N, Re=1000, g=grav)

# for ii∈1:10
# WaterLily.mom_step!(sim.flow, sim.pois, sim.inter, sim.body)
# end

diver,mass,maxU,trueTime,ke,pe = sim_gif!(sim,duration=8, step=0.01,clims=(0,1),plotbody=false,verbose=true,levels=0.0:0.05:1.0,remeasure=false,cfill=:RdBu,linewidth=2,xlimit=[0,32],ylimit=[0,32],shift=(-0.5,-0.5));

trueTime *= sim.U/sim.L
maxU /= sim.U
ke /= sim.U^2*sim.L^2
pe /= sim.U^2*sim.L^2

massrel = abs.((mass.-mass[1])/mass[1]).+1e-20
Plots.plot(trueTime,diver.+1e-20,yaxis=:log10,label="Velocity Divergence" ,color=:red)
Plots.plot!(trueTime,massrel,yaxis=:log10,label="Mass loss",color=:blue)
Plots.plot!(ylimit=[1e-10,1])
Plots.savefig(computationID*"_MassDivergence.png")

Plots.plot(trueTime,maxU)
Plots.savefig(computationID*"_MaxU.png")

Plots.plot()
Plots.plot!(trueTime,ke[1,:],label="K.E. Water",color=:blue,linestyle=:dash)
Plots.plot!(trueTime,pe[1,:],label="P.E. Water",color=:blue,linestyle=:dot)
Plots.plot!(trueTime,ke[1,:].+pe[1,:],label="T.E. Water",color=:blue,linestyle=:solid)
Plots.plot!(trueTime,ke[2,:],label="K.E. Air",color=:green,linestyle=:dash)
Plots.plot!(trueTime,pe[2,:],label="P.E. Air",color=:green,linestyle=:dot)
Plots.plot!(trueTime,ke[2,:].+pe[2,:],label="T.E. Air",color=:green,linestyle=:solid)
Plots.plot!(trueTime,ke[1,:].+pe[1,:].+ke[2,:].+pe[2,:],label="T.E. All",color=:black,linestyle=:solid)
Plots.savefig(computationID*"_Energy.png")


aa = cumsum(sim.flow.Δt)[1:end-1]*sim.U/sim.L

Plots.plot(aa,sim.pois.res0[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(aa,sim.pois.res0[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisRes.png")

Plots.plot(aa,sim.pois.res[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(aa,sim.pois.res[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisResFinal.png")

Plots.plot(aa,sim.pois.n[1:2:end],label="First Stage")
Plots.plot!(aa,sim.pois.n[2:2:end],label="Second Stage")
Plots.savefig(computationID*"_PoisNum.png")

Plots.plot(aa,sim.flow.Δt[1:end-1],ylimit=(0,0.05))
Plots.savefig(computationID*"_delT.png")

jldsave(computationID*"_VelVof.jld";sim.flow.u,sim.inter.f)


