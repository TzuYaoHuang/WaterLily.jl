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
using Tables
using CSV
using Printf

inside(a...) = WaterLily.inside(a...)
δ(a...) = WaterLily.δ(a...)

N = 128
computationID = "DamBreakHeun"*string(N)
grav=9.81

xcen = ((0:N+1).-0.5)/N
xedg = ((0:N+1).-1.0)/N

function keI(I::CartesianIndex{n},u::AbstractArray{T}) where {n,T}
    ke = zero(eltype(u))
    for i∈1:n
        ke += (u[I,i]^2+u[I+δ(i,I),i]^2)/2
    end
    return 0.5ke
end

function ρE(fun,u,f,λρ)
    ρe = zeros(eltype(f),2)
    for I ∈ inside(f)
        buf = fun(I,u)
        fWater = f[I]
        ρe[1] += 1*fWater*buf
        ρe[2] += λρ*(1-fWater)*buf
    end
    return ρe
end
KE(u,f,ρ) = ρE(keI,u,f,ρ)

function PE(f,λρ,g,id)
    pe = zeros(eltype(f),2)
    for I ∈ inside(f)
        fWater = f[I]
        pe[1] += 1*fWater*g*(I.I[id]-1.5)
        pe[2] += λρ*(1-fWater)*g*(I.I[id]-1.5)
    end
    return pe
end

function sim_gif!(sim;duration=1,step=0.1,verbose=true)
    R = inside(sim.flow.p)
    t₀ = round(WaterLily.sim_time(sim))

    xMidList = (1:N).-0.5

    function getFrontLocation(f)
        locTemp = xMidList[1]
        for i∈1:N-1
            cur = f[i+1,2]-0.5
            nex = 0.5-f[i+2,2]
            locTemp = xMidList[i+1]
            if cur*nex >= 0
                locTemp = (nex*xMidList[i]+cur*xMidList[i+1])/(nex+cur)
                break
            end
        end
        return locTemp
    end

    uu = zeros(size(sim.flow.u))
    pp = zeros(size(sim.inter.f))
    uP = pp*0
    ff = pp*0
    ss = ff*0
    oldPStorage = pp*0

    copyto!(uu, sim.flow.u); copyto!(pp, sim.flow.p); copyto!(ss, sim.flow.σ); copyto!(ff, sim.inter.f)
    for I∈inside(ss)
        ss[I] = WaterLily.div(I,uu)
    end
    ke = KE(uu,ff,sim.inter.λρ)
    pe = PE(ff,sim.inter.λρ,grav,2)
    diver = [Statistics.sum(abs.(ss[R]))]
    mass = [Statistics.mean(ff[R])]
    maxU = [maximum(sqrt.(Statistics.sum(uu.^2,dims=4)))]
    loc = [getFrontLocation(ff)]
    trueTime = [WaterLily.time(sim)]
    
    ii=0
    anim = Animation()
    @time for tᵢ in range(t₀,t₀+duration;step)
    # @time for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ,remeasure=false)
        (ii%1 == 0) && WaterLily.SmoothVelocity!(sim.flow,sim.pois,sim.inter,sim.body,oldPStorage)
        ii+=1

        copyto!(uu, sim.flow.u); copyto!(pp, sim.flow.p); copyto!(ss, sim.flow.σ); copyto!(ff, sim.inter.fᶠ)
        for I∈inside(ss)
            ss[I] = WaterLily.div(I,uu)
        end
        push!(diver,Statistics.sum(abs.(ss[R])))
        push!(mass,Statistics.mean(ff[R]))
        push!(loc,getFrontLocation(ff))
        push!(trueTime,WaterLily.time(sim))
        ke = cat(ke,KE(uu,ff,sim.inter.λρ),dims=2)
        pe = cat(pe,PE(ff,sim.inter.λρ,grav,2),dims=2)

        for I∈R
            ss[I] = WaterLily.curl(3,I,uu)*sim.L/sim.U
        end
        WaterLily.BCPer!(ss)
        for I∈R
            uP[I] = √(2keI(I,uu)/sim.U^2)
        end
        WaterLily.BCPer!(uP)
        push!(maxU,maximum(uP)*sim.U)

        Plots.plot()
        # Plots.contourf!(xcen,xcen,clamp.(pp'/sim.U^2*2,0,1),color=:roma,levels=60,linewidth=0,clim=(0,1))
        Plots.contourf!(xcen,xcen,clamp.(uP',0,1.5),color=:GnBu,levels=60,linewidth=0,clim=(0,1.5))
        # Plots.contourf!(xcen,xcen,clamp.(ss',-50,50),color=:seismic,levels=60,linewidth=0,clim=(-50,50))
        Plots.contour!(xcen,xcen,ff',color=:Black,levels=[0.5],linewidth=2)
        frame(anim,Plots.plot!(aspect_ratio=:equal,xlimit=[0,1],ylimit=[0,1]))
        verbose && @printf("tU/L=%6.3f, ΔtU/L=%.8f\n",trueTime[end]*sim.U/sim.L,sim.flow.Δt[end]*sim.U/sim.L)
    end
    gif(anim, computationID*"_hydroDyn.gif", fps = 10)
    return diver,mass,maxU,loc,trueTime,ke,pe
end

function damBreak(NN;Re=493.954,g=grav,arr=Array)
    LDomain = (NN,NN)
    H = NN/4
    LScale = H
    UScale = sqrt(2g*LScale)
    ν = UScale*LScale/Re

    function interSDF(xx)
        x = xx .-1.5 .- SA[H,2H]
        return √sum((xi) -> max(0,xi)^2,x) + min(0, maximum(x))
    end

    return WaterLily.TwoPhaseSimulation(LDomain, (0,0), LScale;U=UScale, Δt=0.01,grav=(0,-g), ν=ν, InterfaceSDF=interSDF, T=Float32,λμ=1e-2,λρ=1e-3,mem=arr)
end

# using CUDA: CUDA
# @assert CUDA.functional()
# println("CUDA is working? ", CUDA.functional())

sim = damBreak(N,arr=Array)

diver,mass,maxU,loc,trueTime,ke,pe = sim_gif!(sim,duration=15,step=0.05,verbose=true);

trueTime *= sim.U/sim.L
maxU /= sim.U
loc /= sim.L
ke /= sim.U^2*sim.L^2
pe /= sim.U^2*sim.L^2

MartinData = CSV.File("DamBreakFront_SunTao.csv") |> Tables.matrix

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
Plots.plot!(trueTime,ke[1,:].+pe[1,:].+ke[2,:].+pe[2,:],label="T.E. All",color=:black,linestyle=:solid,legend=:bottomleft)
Plots.savefig(computationID*"_Energy.png")


Plots.plot(trueTime,loc,color=:black,linewidth=1.25,label="Simulation")
Plots.scatter!(MartinData[:,1],MartinData[:,2],color=:black,label="Sun and Tao (2010)")
Plots.plot!(xlimit=[0,3],ylimit=[1,4],aspect_ratio=:equal)
Plots.savefig(computationID*"_FrontLocation.png")

aa = cumsum(sim.flow.Δt)[1:end-1]*sim.U/sim.L

Plots.plot(aa,sim.pois.res0[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(aa,sim.pois.res0[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisRes0.png")

Plots.plot(aa,sim.pois.res[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(aa,sim.pois.res[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisRes.png")

Plots.plot(aa,sim.pois.n[1:2:end],label="First Stage")
Plots.plot!(aa,sim.pois.n[2:2:end],label="Second Stage")
Plots.savefig(computationID*"_PoisNum.png")

Plots.plot(aa,sim.flow.Δt[1:end-1],ylimit=(0,0.05))
Plots.savefig(computationID*"_delT.png")

jldsave(computationID*"_VelVof.jld";sim.flow.u,sim.inter.f)


