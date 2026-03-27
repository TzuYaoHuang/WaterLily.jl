using WaterLily,StaticArrays
import WaterLily: fsum, fSV, sim_time,inside_u

ke(I::CartesianIndex{m},u,U=fSV(zero,m)) where m = 0.25fsum(m) do i
    abs2(u[I,i])+abs2(u[I+δ(i,I),i])-2abs2(U[i])
end

function circle(center;ϵ=1,Re=100,n=3*2^6,m=2^7,U=1,T=typeof(center))
    radius = m÷8
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body,T,ϵ)
end
function circle_ke(x::T;kwargs...) where T
    sim = circle(x;kwargs...)
    # for iTime∈1:10
    #     sim.flow.Δt[end]=T(0.2)
    #     mom_step!(sim.flow, sim.pois)
    # end
    # sim_step!(sim,0.1)
    # println(sim_time(sim))
    # println(sim.flow.Δt)
    # sum(I->WaterLily.ke(I,sim.flow.u),inside(sim.flow.p))
    # sum(I->ke(I,sim.flow.u),inside(sim.flow.p))
    # sum(I->ke(I,sim.flow.u,SA[1,0]),inside(sim.flow.p))
    # sum(I->WaterLily.ke(I,sim.flow.u,SA[1,0]),inside(sim.flow.p))
    # sim.flow.u[83,66,2] # within the BDIM-ϵ region
    # sim_time(sim)
    # sum(Ii->1.0-sim.flow.μ₀[Ii],inside_u(size(sim.flow.p)))/2
    # sum(I->sim.pois.levels[1].D[I],inside(sim.flow.p))
    sum(I->sim.pois.levels[1].iD[I],inside(sim.flow.p))
    # sum(sim.flow.u[2:end-1,1:end-1,2])
end

using Plots,TypedTables 
data = map(range(64,66,200)) do x
    (x=x,double=circle_ke(Float64(x)),single=circle_ke(Float32(x)))
end |> Table
data2 = map(range(64,66,200)) do x
    (x=x,double=circle_ke(Float64(x),ϵ=2),single=circle_ke(Float32(x),ϵ=2))
end |> Table

title="Sum of iD"

plot()
plot!(data.x,data.double,label="Float64, ϵ=1",c=:darkblue)
plot!(data.x,data.single,label="Float32, ϵ=1",ls=:dash,c=:royalblue)
plot!(data2.x,data2.double,label="Float64, ϵ=2",c=:darkred)
plot!(data2.x,data2.single,label="Float32, ϵ=2",ls=:dash,c=:orange)
plot!(title=title)

plot()
plot!(data.x,-data.double,label="Float64, ϵ=1",c=:darkblue)
plot!(data.x,-data.single,label="Float32, ϵ=1",ls=:dash,c=:royalblue)
plot!(data2.x,-data2.double,label="Float64, ϵ=2",c=:darkred)
plot!(data2.x,-data2.single,label="Float32, ϵ=2",ls=:dash,c=:orange)
plot!(yscale=:log10)
plot!(title=title*" (log negative)")

plot()
plot!(data.x,data.double.-data.double[1],label="Float64, ϵ=1",c=:darkblue)
plot!(data.x,data.single.-data.double[1],label="Float32, ϵ=1",ls=:dash,c=:royalblue)
plot!(data2.x,data2.double.-data2.double[1],label="Float64, ϵ=2",c=:darkred)
plot!(data2.x,data2.single.-data2.double[1],label="Float32, ϵ=2",ls=:dash,c=:orange)
plot!(title=title*" w/ value @ center=64 as baseline")

plot()
plot!(data.x[2:end],data.double[2:end]-data.double[1:end-1],label="Float64, ϵ=1",c=:darkblue)
plot!(data.x[2:end],data.single[2:end]-data.single[1:end-1],label="Float32, ϵ=1",ls=:dash,c=:royalblue)
plot!(data2.x[2:end],data2.double[2:end]-data2.double[1:end-1],label="Float64, ϵ=2",c=:darkred)
plot!(data2.x[2:end],data2.single[2:end]-data2.single[1:end-1],label="Float32, ϵ=2",ls=:dash,c=:orange)
plot!(title="Difference of "*title)

plot()
plot!(data.x[2:end],(data.double[2:end]-data.double[1:end-1])*√eps(Float64),label="Float64, ϵ=1",c=:darkblue)
plot!(data.x[2:end],(data.single[2:end]-data.single[1:end-1])*√eps(Float32),label="Float32, ϵ=1",ls=:dash,c=:royalblue)
plot!(data2.x[2:end],(data2.double[2:end]-data2.double[1:end-1])*√eps(Float64),label="Float64, ϵ=2",c=:darkred)
plot!(data2.x[2:end],(data2.single[2:end]-data2.single[1:end-1])*√eps(Float32),label="Float32, ϵ=2",ls=:dash,c=:orange)
plot!(title="Difference of "*title* " scale with √eps")
