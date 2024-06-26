module WaterLily

include("util.jl")
export L₂,BC!,@inside,inside,δ,apply!,loc

using Reexport
@reexport using KernelAbstractions: @kernel,@index,get_backend

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult!

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult!

include("Flow.jl")
export Flow,mom_step!

include("Body.jl")
export AbstractBody,measure_sdf!

include("AutoBody.jl")
export AutoBody,Bodies,measure,sdf,+,-

include("Metrics.jl")

include("Multiphase.jl")
export cVOF,mom_step!

include("ComponentLabelling.jl")

abstract type AbstractSimulation end

"""
    Simulation(dims::NTuple, u_BC::Union{NTuple,Function}, L::Number;
               U=norm2(u_BC), Δt=0.25, ν=0., ϵ=1, perdir=(1,)
               uλ::nothing, g=nothing, exitBC=false,
               body::AbstractBody=NoBody(),
               T=Float32, mem=Array)

Constructor for a WaterLily.jl simulation:

  - `dims`: Simulation domain dimensions.
  - `u_BC`: Simulation domain velocity boundary conditions, either a
            tuple `u_BC[i]=uᵢ, i=eachindex(dims)`, or a time-varying function `f(i,t)`
  - `L`: Simulation length scale.
  - `U`: Simulation velocity scale.
  - `Δt`: Initial time step.
  - `ν`: Scaled viscosity (`Re=UL/ν`).
  - `g`: Domain acceleration, `g(i,t)=duᵢ/dt`
  - `ϵ`: BDIM kernel width.
  - `perdir`: Domain periodic boundary condition in the `(i,)` direction.
  - `exitBC`: Convective exit boundary condition in the `i=1` direction.
  - `uλ`: Function to generate the initial velocity field.
  - `body`: Immersed geometry.
  - `T`: Array element type.
  - `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or AMD devices, respectively.

See files in `examples` folder for examples.
"""
struct Simulation <: AbstractSimulation
    U :: Number # velocity scale
    L :: Number # length scale
    ϵ :: Number # kernel width
    flow :: Flow
    body :: AbstractBody
    pois :: AbstractPoisson
    function Simulation(dims::NTuple{N}, u_BC, L::Number;
                        Δt=0.25, ν=0., g=nothing, U=nothing, ϵ=1, perdir=(0,),
                        uλ=nothing, exitBC=false, body::AbstractBody=NoBody(),
                        T=Float32, mem=Array) where N
        @assert !(isa(u_BC,Function) && isa(uλ,Function)) "`u_BC` and `uλ` cannot be both specified as Function"
        @assert !(isnothing(U) && isa(u_BC,Function)) "`U` must be specified if `u_BC` is a Function"
        isa(u_BC,Function) && @assert all(typeof.(ntuple(i->u_BC(i,T(0)),N)).==T) "`u_BC` is not type stable"
        uλ = isnothing(uλ) ? ifelse(isa(u_BC,Function),(i,x)->u_BC(i,0.),(i,x)->u_BC[i]) : uλ
        U = isnothing(U) ? √sum(abs2,u_BC) : U # default if not specified
        flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,body,MultiLevelPoisson(flow.p,flow.μ₀,flow.σ;perdir))
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
                        Δt=0.25, ν=0.,λμ=1e-2,λρ=1e-3,η=0, U=√sum(abs2,u_BC), ϵ=1, 
                        perdir=(0,), dirdir=(0,), g=nothing,
                        uλ::Function=(i,x)->u_BC[i], exitBC=false,
                        InterfaceSDF::Function=(x) -> -5-x[1],
                        body::AbstractBody=NoBody(),T=Float32,mem=Array) where N
        flow = Flow(dims,u_BC;uλ,Δt,ν,g,T,f=mem,perdir,exitBC)
        inter= cVOF(dims; arr=mem,InterfaceSDF,T,perdir,dirdir,λμ,λρ,η)
        measure!(flow,body;ϵ)
        new(U,L,ϵ,flow,inter,body,Poisson(flow.p,flow.μ₀,flow.σ;perdir))
    end
end

time(sim::AbstractSimulation) = time(sim.flow)
"""
    sim_time(sim::AbstractSimulation)

Return the current dimensionless time of the simulation `tU/L`
where `t=sum(Δt)`, and `U`,`L` are the simulation velocity and length
scales.
"""
sim_time(sim::AbstractSimulation) = time(sim)*sim.U/sim.L

"""
    sim_step!(sim::Simulation,t_end=sim(time)+Δt;max_steps=typemax(Int),remeasure=true,verbose=false)

Integrate the simulation `sim` up to dimensionless time `t_end`.
If `remeasure=true`, the body is remeasured at every time step.
Can be set to `false` for static geometries to speed up simulation.
"""
function sim_step!(sim::Simulation,t_end;remeasure=true,max_steps=typemax(Int),verbose=false)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure)
        verbose && println("tU/L=",round(sim_time(sim),digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end
function sim_step!(sim::Simulation;remeasure=true)
    remeasure && measure!(sim)
    mom_step!(sim.flow,sim.pois)
end
function sim_step!(sim::TwoPhaseSimulation,t_end;verbose=false,remeasure=true,smoothStep=Inf,oldPStorage=zeros(1),ω=1)
    t = time(sim)
    while t < t_end*sim.L/sim.U
        remeasure && measure!(sim,t)
        mom_step!(sim.flow,sim.pois,sim.inter,sim.body) # evolve Flow
        (length(sim.flow.Δt)%smoothStep==0) && smoothVelocity!(sim.flow,sim.pois,sim.inter,sim.body,oldPStorage;ω)
        t += sim.flow.Δt[end]
        verbose && @printf(
            "    tU/L=%6.3f, ΔtU/L=%.10f, poisN=%4d, r0=%10.6e, re=%10.6e\n",
            t*sim.U/sim.L,sim.flow.Δt[end]*sim.U/sim.L,sim.pois.n[end],sim.pois.res0[end],sim.pois.res[end]
        ); flush(stdout)
    end
end

"""
    measure!(sim::AbstractSimulation,t=timeNext(sim))

Measure a dynamic `body` to update the `flow` and `pois` coefficients.
"""
function measure!(sim::Simulation,t=timeNext(sim.flow))
    measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    update!(sim.pois)
end
function measure!(sim::TwoPhaseSimulation,t=time(sim))
    measure!(sim.flow,sim.pois,sim.inter,sim.body,t)
end

export Simulation,sim_step!,sim_time,measure!

# default WriteVTK functions
function vtkWriter end
function write! end
function default_attrib end
function pvd_collection end
# export
export vtkWriter, write!, default_attrib

# default ReadVTK functions
function restart_sim! end
# export
export restart_sim!

# Backward compatibility for extensions
if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/WaterLilyAMDGPUExt.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/WaterLilyCUDAExt.jl")
        @require WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192" include("../ext/WaterLilyWriteVTKExt.jl")
        @require ReadVTK = "dc215faf-f008-4882-a9f7-a79a826fadc3" include("../ext/WaterLilyReadVTKExt.jl")
    end
end

end # module
