module WaterLily

include("util.jl")
export L₂,BC!,BCPer!,BCPerVec!,@inside,inside,δ,apply!,loc

using Reexport
@reexport using KernelAbstractions: @kernel,@index,get_backend

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult!,residual!,Jacobi!,smooth!,increment!,pcg!,pureSolver!

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult!,Vcycle!,residual!,restrict!,prolongate!

include("Flow.jl")
export Flow,mom_step!

include("Body.jl")
export AbstractBody,measure_sdf!

include("AutoBody.jl")
export AutoBody,measure,sdf,+,-

include("Metrics.jl")

include("Multiphase.jl")
export cVOF,mom_step!

include("ComponentLabelling.jl")

abstract type AbstractSimulation end

"""
    Simulation(dims::NTuple, u_BC::NTuple, L::Number;
               U=norm2(u_BC), Δt=0.25, ν=0., ϵ=1,
               uλ::Function=(i,x)->u_BC[i],
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `u_BC`: Simulation domain velocity boundary conditions, `u_BC[i]=uᵢ, i=eachindex(dims)`.
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `ϵ`: BDIM kernel width.
  - `uλ`: Function to generate the initial velocity field.
  - `body`: Immersed geometry.
  - `T`: Array element type.
  - `mem`: memory location. `Array` and `CuArray` run on CPU and CUDA backends, respectively.

See files in `examples` folder for examples.
"""
struct Simulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::NTuple{N}, u_BC::NTuple{N}, L::Number;
                        Δt=0.25, ν=0., U=√sum(abs2,u_BC), ϵ=1, perdir=(0,),
                        uλ::Function=(i,x)->u_BC[i],
                        body::AbstractBody=NoBody(),T=Float32,mem=Array) where N
        flow = Flow(dims,u_BC;uλ,Δt,ν,T,f=mem,perdir=perdir)
        measure!(flow,body;ϵ,perdir=perdir)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir=perdir))
    end
end

struct TwoPhaseSimulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    inter:: cVOF
    body :: AbstractBody
    pois :: AbstractPoisson
    function TwoPhaseSimulation(
                        dims::NTuple{N}, u_BC::NTuple{N}, L::Number;
                        Δt=0.25, ν=0.,λμ=0.0180989244,λρ=0.001206, U=√sum(abs2,u_BC), ϵ=1, 
                        perdir=(0,), dirdir=(0,), grav=ntuple(i->0,N),
                        uλ::Function=(i,x)->u_BC[i], 
                        InterfaceSDF::Function=(x) -> -5-x[1],
                        body::AbstractBody=NoBody(),T=Float32,mem=Array) where N
        flow = Flow(dims,u_BC;uλ,Δt,ν,T,f=mem,perdir=perdir,g=grav)
        inter= cVOF(dims,flow.f,flow.σ; arr=mem, InterfaceSDF=InterfaceSDF, T=T, perdir=flow.perdir, dirdir=dirdir,λμ=λμ,λρ=λρ)
        measure!(flow,body;ϵ,perdir=perdir)
        # calculateL!(flow,inter)
        new(U,L,ϵ,flow,inter,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir=perdir))
    end
end


time(sim::AbstractSimulation) = sum(sim.flow.Δt[1:end-1])
"""
    sim_time(sim::AbstractSimulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::AbstractSimulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::AbstractSimulation,t_end;remeasure=true,verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step. 
Can be set to `false` for static geometries to speed up simulation.
"""
function sim_step!(sim::Simulation,t_end;verbose=false,remeasure=true)
    t = time(sim)
    while t < t_end*sim.L/sim.U
        remeasure && measure!(sim,t)
        mom_step!(sim.flow,sim.pois) # evolve Flow
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function sim_step!(sim::TwoPhaseSimulation,t_end;verbose=false,remeasure=true,smoothStep=Inf,oldPStorage=zeros(1))
    t = time(sim)
    while t < t_end*sim.L/sim.U
        remeasure && measure!(sim,t)
        mom_step!(sim.flow,sim.pois,sim.inter,sim.body) # evolve Flow
        (length(sim.flow.Δt)%smoothStep==0) && SmoothVelocity!(sim.flow,sim.pois,sim.inter,sim.body,oldPStorage)
        t += sim.flow.Δt[end]
        verbose && println("tU/L=",round(t*sim.U/sim.L,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

"""
    measure!(sim::AbstractSimulation,t=time(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::Simulation,t=time(sim))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ,perdir=sim.flow.perdir)
    update!(sim.pois)
end

export Simulation,TwoPhaseSimulation,sim_step!,sim_time,measure!,@inside,inside
end # module
