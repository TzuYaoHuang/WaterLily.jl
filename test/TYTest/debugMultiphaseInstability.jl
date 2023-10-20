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

# CASE configuration
N = 96
q = 1.0
disturb = 0.05
computationID =  @sprintf("3DNewVortexBreak%d_q%.2f_dis%.2f",N,q,disturb)
println("You are now running: "*computationID); flush(stdout)

function calculateDiv!(flow)
    for I∈inside(flow.σ)
        flow.σ[I] = WaterLily.div(I,flow.u)
    end
end

function puresim!(sim,duration,recostep)
    timeCurrent = WaterLily.sim_time(sim)
    timeArray = (timeCurrent+recostep):recostep:duration

    numProject = 3
    for i ∈ 1:numProject
        WaterLily.project!(sim.flow,sim.pois)
        WaterLily.BCVecPerNeu!(sim.flow.u;Dirichlet=true, A=sim.flow.U, perdir=sim.flow.perdir)
        println("Projected the initial velocity field to the divergence free space. ($i/$numProject)"); flush(stdout)
    end

    trueTime = [WaterLily.time(sim)]

    iTime = 0
    jldsave("JLD2/"*computationID*"VelVOF_"*string(iTime)*".jld2"; u=Array(sim.flow.u), f=Array(sim.inter.f))
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


function multiphaseQVortex(NN; Re=4000, T=Float32, mem=Array)
    # Define vortex size, velocity, viscosity
    Lp = NN  # The grid of WaterLily is always Δx=Δy=Δz=1, so the length of domain is exactly the number of grids.
    delta0 = 1/16*Lp*0.892  # vortex core radius as 1/16 of the domain such that λ = 8dᵥ
    dᵥ = 2/0.892 # normalize Dv
    λ = 8dᵥ
    U = 1.0
    ν = delta0*U/Re

    # q-Vortex initial velocity field
    function uλ(i,xyz)
        # nomalized coordinate with basic length δ0
        x,y,z = @. (xyz-1.5-Lp/2)/delta0

        # converted to cylindrical coordinate
        r = sqrt(x^2+y^2) + 1e-10
        theta = atan(y,x)
        ξ = r/(4.001dᵥ)
        compactRSupport = max(0, (ξ^2-2)/(2*(ξ^2-1))*exp(ξ^2/(ξ^2-1)))

        # velocities in cylindrical coordinate
        uTheta  = q/r*U*(1-exp(-r^2))*compactRSupport
        uRadial = disturb*U/r*(1-exp(-r^2))/0.63817*sin(2*pi/λ*z)*compactRSupport
        uAxial  =     U*(1-exp(-r^2))  # wake-like axial flow

        zConnect = 1.0

        # return the velocity component in cartisian coordiante
        i==1 && return  uTheta*-sin(theta)+uRadial*cos(theta) # u_x
        i==2 && return  uTheta* cos(theta)+uRadial*sin(theta) # u_y
        i==3 && return  uAxial*zConnect    # u_z
        return 0
    end

    # Interface function
    function Inter(xyz)
        x,y,z = @. (xyz-1.5-Lp/2)
        return 0.5*dᵥ*delta0-√(x^2+y^2)
    end

    # Initialize simulation
    # (Lp, Lp, Lp) is the domain size, 
    # (0, 0, 0) specified the boundary BC. due to sepcial treatment in WaterLily, it is still slip instead of no-slip BC
    # delta0, U together define the length and velocity scales to normalize the simulation time
    return WaterLily.TwoPhaseSimulation(
        (Lp, Lp, Lp), (0, 0, 0), dᵥ*delta0;
        U=U, Δt=0.01, ν=ν, InterfaceSDF=Inter, T=T,λμ=1e-2,λρ=1e-3,perdir=(1,2,3),uλ=uλ,mem=mem
    )
end

dur = 200
stp = 0.1

sim = multiphaseQVortex(N)

puresim!(sim,dur,stp)





