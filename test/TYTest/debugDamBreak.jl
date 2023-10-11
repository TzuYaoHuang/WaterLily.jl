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
computationID = "DamBreakHeun"*string(N)
grav=9.81

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
    # for i∈1:n
    #     for I ∈ WaterLily.inside_uWB(N,i)
    #         buf = u[I,i]^2
    #         fWater = WaterLily.ϕ(i,I,f)
    #         if (I[i] == 2) || (I[i] == N[i])
    #             ke[1] += 1*fWater*buf*0.5
    #             ke[2] += λρ*(1-fWater)*buf*0.5
    #         else
    #             ke[1] += 1*fWater*buf
    #             ke[2] += λρ*(1-fWater)*buf
    #         end
    #     end
    # end
    for I ∈ inside(f)
        for i ∈ 1:n
            buf = (u[I,i]+u[I+δ(i,I),i])^2*0.25
            fWater = f[I]
            ke[1] += 1*fWater*buf
            ke[2] += λρ*(1-fWater)*buf
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
    ke = KE(sim.flow.u,sim.inter.f,sim.inter.λρ)
    pe = PE(sim.inter.f,sim.inter.λρ,grav,2)
    diver = [Statistics.sum(abs.(sim.flow.σ[R]))]
    mass = [Statistics.mean(sim.inter.f[R])]
    maxU = [maximum(sqrt.(Statistics.sum(sim.flow.u.^2,dims=4)))]
    loc = [getFrontLocation(sim.inter.f)]
    trueTime = [WaterLily.time(sim)]
    
    @time anim = @animate for tᵢ in range(t₀,t₀+duration;step)
    # @time for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ;remeasure)
        for I∈inside(sim.flow.σ)
            sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        end
        push!(diver,maximum(abs.(sim.inter.f[R].*sim.flow.σ[R])))
        push!(mass,Statistics.mean(sim.inter.f[R]))
        push!(maxU,maximum(sqrt.(Statistics.sum(sim.flow.u.^2,dims=4))))
        push!(loc,getFrontLocation(sim.inter.f))
        push!(trueTime,WaterLily.time(sim))
        ke = cat(ke,KE(sim.flow.u,sim.inter.f,sim.inter.λρ),dims=2)
        pe = cat(pe,PE(sim.inter.f,sim.inter.λρ,grav,2),dims=2)
        Plots.contourf(lx,ly,clamp.(sim.flow.p[2:end-1,2:end-1]'/sim.U^2*2,0,1), aspect_ratio=:equal,color=:dense,levels=60,xlimit=[0,1],ylimit=[0,1],linewidth=0,clim=(0,1))
        Plots.contour!(lx,ly,sim.inter.f[2:end-1,2:end-1]', aspect_ratio=:equal,color=:Black,levels=[0.5],xlimit=[0,1],ylimit=[0,1],linewidth=2)
        plotbody && body_plot!(sim)
        verbose && @printf("tU/L=%6.3f, ΔtU/L=%.8f\n",trueTime[end]*sim.U/sim.L,sim.flow.Δt[end]*sim.U/sim.L)
    end
    gif(anim, computationID*"_hydroDyn.gif", fps = 30)
    return diver,mass,maxU,loc,trueTime,ke,pe
end

function damBreak(NN;Re=493.954,g=grav)
    LDomain = (NN,NN)
    H = NN/4
    LScale = H
    UScale = sqrt(2g*LScale)
    ν = UScale*LScale/Re

    function interSDF(xx)
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

    return WaterLily.TwoPhaseSimulation(LDomain, (0,0), LScale;U=UScale, Δt=0.01,grav=(0,-g), ν=ν, InterfaceSDF=interSDF, T=Float64,λμ=1e-3,λρ=1e-3)
end


sim = damBreak(N)

diver,mass,maxU,loc,trueTime,ke,pe = sim_gif!(sim,duration=10, step=0.01,clims=(0,1),plotbody=false,verbose=true,levels=0.0:0.05:1.0,remeasure=false,cfill=:RdBu,linewidth=2,xlimit=[0,32],ylimit=[0,32],shift=(-0.5,-0.5));

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
Plots.plot!(trueTime,ke[1,:].+pe[1,:].+ke[2,:].+pe[2,:],label="T.E. All",color=:black,linestyle=:solid)
Plots.savefig(computationID*"_Energy.png")


Plots.plot(trueTime,loc,color=:black,linewidth=1.25,label="Simulation")
Plots.scatter!(MartinData[:,1],MartinData[:,2],color=:black,label="Sun and Tao (2010)")
Plots.plot!(xlimit=[0,3],ylimit=[1,4],aspect_ratio=:equal)
Plots.savefig(computationID*"_FrontLocation.png")

aa = cumsum(sim.flow.Δt)[1:end-1]*sim.U/sim.L

Plots.plot(aa,sim.pois.res0[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(aa,sim.pois.res0[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisRes.png")

Plots.plot(aa,sim.pois.n[1:2:end],label="First Stage")
Plots.plot!(aa,sim.pois.n[2:2:end],label="Second Stage")
Plots.savefig(computationID*"_PoisNum.png")

Plots.plot(aa,sim.flow.Δt[1:end-1],ylimit=(0,0.05))
Plots.savefig(computationID*"_delT.png")

jldsave(computationID*"_VelVof.jld";sim.flow.u,sim.inter.f)


