# INCLUDE the WaterLily routine 
filePath = @__FILE__
workdir = dirname(filePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"
include(waterlilypath)
WaterLily = Main.WaterLily;

using Printf
using JLD2

# DEFINE some useful functions
inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

N = 128

# CASE configuration
computationID =  @sprintf("3DDoublySinusoidalN%d",N)
println("You are now running: "*computationID); flush(stdout)

function calculateDiv!(flow)
    for I∈inside(flow.σ)
        flow.σ[I] = WaterLily.div(I,flow.u)
    end
end

function puresim!(sim,duration,recostep)
    timeCurrent = WaterLily.sim_time(sim)
    timeArray = (timeCurrent+recostep):recostep:duration

    numProject = 0
    trueTime = [WaterLily.time(sim)]

    oldPStorage = sim.flow.p*0

    iTime = 0
    jldsave("JLD2/"*computationID*"VelVOF_"*string(iTime)*".jld2"; u=Array(sim.flow.u), f=Array(sim.inter.f))
    ii = 0
    @time for timeNow in timeArray
        WaterLily.sim_step!(sim,timeNow;remeasure=false)
        iTime += 1
        jldsave("JLD2/"*computationID*"VelVOF_"*string(iTime)*".jld2"; u=Array(sim.flow.u), f=Array(sim.inter.f))
        push!(trueTime,WaterLily.time(sim))
        @printf("tU/L=%6.3f, ΔtU/L=%.8f\n",trueTime[end]*sim.U/sim.L,sim.flow.Δt[end]*sim.U/sim.L); flush(stdout)
        jldsave(
            "JLD2/"*computationID*"General.jld2"; 
            trueTime,
            U=sim.U,L=sim.L,rhoRatio=sim.inter.λρ,visRatio=sim.inter.λμ,
            resFin=sim.pois.res[numProject+1:end], resIni=sim.pois.res0[numProject+1:end],
            poisN=sim.pois.n[numProject+1:end],
            dts=sim.flow.Δt
        )
    end
end


function DS(N; Re=4000, T=Float32, mem=Array)
    NN = (N,N,N)
    λ = N
    g = 1
    U = √(g*λ)
    ν = U*λ/Re

    # Interface function
    function Inter(xyz)
        x,y,z = @. (xyz-1.5-N/2)
        return z - 0.25λ*cos(2π*x/λ)*cos(2π*y/λ)
    end

    return WaterLily.TwoPhaseSimulation(
        NN, (0, 0, 0), λ;
        U=U, Δt=0.001, ν=ν, InterfaceSDF=Inter, T=T, λμ=1e-2, λρ=1e-3, mem=mem, grav=(0,0,-g)
    )
end

dur = 10
stp = 0.01

sim = DS(N,T=Float32,Re=10000)

puresim!(sim,dur,stp)





