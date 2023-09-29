filePath = @__FILE__
workdir = dirname(filePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"

include(waterlilypath)
WaterLily = Main.WaterLily;
using Plots; gr()
using StaticArrays
using JLD
using BenchmarkTools

ENV["GKSwstype"]="nul"

computationID = "2DTGV"

inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

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
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    contour!(sim.flow.σ[R]';levels,lines)
end

function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)
    t₀ = round(WaterLily.sim_time(sim))
    omegaInter = Float64[]
    @time anim = @animate for tᵢ in range(t₀,t₀+duration;step)
        # Plots.plot()
        WaterLily.sim_step!(sim,tᵢ;remeasure)
        for I∈inside(sim.flow.σ)
            sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        end
        WaterLily.BCPer!(sim.flow.σ)
        flood(sim.flow.σ[2:end-1,2:end-1]; kv...)
        # flood(sim.flow.p[2:end-1,2:end-1]/sim.U^2; kv...)
        # flood(sim.inter.f[2:end-1,2:end-1,4]; kv...)
        plotbody && body_plot!(sim)
        verbose && println("tU/L=",round(tᵢ,digits=4),
            ", Δt=",sim.flow.Δt[end])
	flush(stdout)
    end
    gif(anim, computationID*"_VOFPlot.gif", fps = 30)
    return omegaInter
end

function TGV(; L=64, Re=1e5, T=Float64, mem=Array)
    # Define vortex size, velocity, viscosity
    L = L; U = 1; ν = U*L/Re
    # Taylor-Green-Vortex initial velocity field
    function uλ(i,xyz)
        x,y = @. (xyz-1.5)*2π/L                # scaled coordinates
        i==1 && return -U*(sin(x)*cos(y)+0.5*sin(5x)*cos(5y)) # u_x
        i==2 && return  U*(cos(x)*sin(y)+0.5*cos(5x)*sin(5y)) # u_y
        return 0.                              # u_z
    end
    InterfaceSDF=(x) -> (x[1]-1.5+x[2]-1.5)-L
    InterfaceSDF=(x) -> (x[1]-1.5)-L/2
    # Initialize simulation
    return WaterLily.Simulation((L, L), (0, 0), L; U=U, Δt=0.01, uλ=uλ, ν=ν, T=T, mem=mem,perdir=(1,2))
    # return WaterLily.TwoPhaseSimulation((L, L), (0,0), L;U=U, uλ=uλ, Δt=0.01, ν=ν, T=T, mem=mem,perdir=(0,),λν=1.0,λρ=1.0, InterfaceSDF=InterfaceSDF)
end

sim = TGV(;Re=10000,L=64)
# sim = TGVHalfPer(;Re=10000)

omegaInter = sim_gif!(sim,duration=10, step=0.01,clims=(-10,10),plotbody=false,verbose=true,levels=41);


aa = cumsum(sim.flow.Δt)[1:end-1]*sim.U/sim.L

Plots.plot(aa,sim.pois.res[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(aa,sim.pois.res[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
# Plots.plot(aa,sim.pois.res[1:1:end],yscale=:log10,label="First Stage")
Plots.savefig(computationID*"_PoisRes.png")

Plots.plot(aa,sim.pois.n[1:2:end],label="First Stage")
Plots.plot!(aa,sim.pois.n[2:2:end],label="Second Stage")
# Plots.plot(aa,sim.pois.n[1:1:end],label="First Stage")
Plots.savefig(computationID*"_PoisNum.png")

Plots.plot(aa,sim.flow.Δt[1:end-1])
Plots.savefig(computationID*"_delT.png")
