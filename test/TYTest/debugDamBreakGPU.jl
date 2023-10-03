filePath = @__FILE__
workdir = dirname(filePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"

include(waterlilypath)
WaterLily = Main.WaterLily;
# using WaterLily
using StaticArrays
using JLD2
using Cthulhu

inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

N = 16
computationID = "DamBreakHeun"*string(N)

function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)
    t₀ = round(WaterLily.sim_time(sim))
    @time for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ;remeasure)
        verbose && println("tU/L=",round(tᵢ,digits=4),
            ", Δt=",sim.flow.Δt[end])
    end
end

function damBreak(NN;Re=493.954,g=9.81, T=Float32, mem=Array)
    LDomain = (NN,NN)
    H = NN/4
    LScale = H
    UScale = sqrt(2g*LScale)
    ν = UScale*LScale/Re

    function interSDF_(xx)
        y,z = @. xx-1.5
        if y<=H && z <= 2H
            return max(y-H,z-2H)
        elseif y<=H && z > 2H
            return z-2H
        elseif y>H && z <= 2H
            return y-H
        elseif y>H && z > 2H
            return sqrt((z-2H)^2+(y-H)^2)
        end
    end

    function interSDF(xx)
        y,z = @. xx-1.5
        return y-H
    end

    return WaterLily.TwoPhaseSimulation(LDomain, (0,0), LScale;U=UScale, Δt=0.01,grav=(0,-g), ν=ν, InterfaceSDF=interSDF, T=T, mem=mem,λν=1e-3,λρ=1e-3)
end

function TGV(NN; Re=1e5, T=Float32, mem=Array)
    # Define vortex size, velocity, viscosity
    L = NN; U = 1; ν = U*L/Re
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
    return WaterLily.Simulation((L, L), (0, 0), L; U=U, Δt=0.01, uλ=uλ, ν=ν, T=T, mem=mem,perdir=(1,2))#
    # return WaterLily.TwoPhaseSimulation((L, L), (0,0), L;U=U, uλ=uλ, Δt=0.01, ν=ν, T=T, mem=mem,perdir=(0,),λν=1.0,λρ=1.0, InterfaceSDF=InterfaceSDF)
end

using CUDA: CUDA
@assert CUDA.functional()
println("CUDA is working? ", CUDA.functional())

sim = damBreak(N,mem=CUDA.CuArray)
# sim = damBreak(N,mem=Array)
# sim = TGV(N,mem=CUDA.CuArray)

sim_gif!(sim,duration=10, step=0.01,clims=(0,1),plotbody=false,verbose=true,levels=0.0:0.05:1.0,remeasure=false,cfill=:RdBu,linewidth=2,xlimit=[0,32],ylimit=[0,32],shift=(-0.5,-0.5));

jldsave(computationID*"_Vel.jld2";sim.flow.u|>Array)
jldsave(computationID*"_Vof.jld2";sim.inter.f|>Array)


