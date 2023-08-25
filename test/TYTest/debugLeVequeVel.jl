include("../../src/WaterLily.jl")
WaterLily = Main.WaterLily;
using Plots; gr()
using StaticArrays
using JLD
using Images
using FFTW
using Statistics
using Interpolations
using DelimitedFiles
using PyPlot
using GLMakie
GLMakie.activate!()
using GRUtils

Lp = 128
N = (Lp, Lp, Lp)

inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())
@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]
@fastmath @inline function divv(I::CartesianIndex{m},u) where {m}
    init=zero(eltype(u))
    for i in 1:m
     init += @inbounds ∂(i,I,u)
    end
    return init
end

function project!(a,b)
    dt = a.Δt[end]
    for I∈inside(b.z)
        b.z[I] = divv(I,a.u) # divergence source term
    end
    rr=WaterLily.solver!(b,tol=1e-3,itmx=1000,log=true)
    for i ∈ 1:3  # apply pressure solution b.x
        for I∈inside(b.z) 
            a.u[I,i] -= b.L[I,i]*∂(i,I,b.x)
        end
    end
    return rr
end

sim = WaterLily.Simulation(N, (0, 0, 0), Lp; U=1, ν=0.0,T=Float32,mem=Array,perdir=(1,2,3), Δt=0.00001)

xList = reshape((1:N[1]+2).-2,(N[1]+2,1,1))/N[1]
yList = reshape((1:N[2]+2).-2,(1,N[2]+2,1))/N[2]
zList = reshape((1:N[3]+2).-2,(1,1,N[3]+2))/N[3]


function leVeque!(u,x,y,z)
    u[:,:,:,1] = @. 2*sin( pi*x)^2*sin(2pi*y)  *sin(2pi*z)  *N[1]
    u[:,:,:,2] = @.  -sin(2pi*x)  *sin( pi*y)^2*sin(2pi*z)  *N[2]
    u[:,:,:,3] = @.  -sin(2pi*x)  *sin(2pi*y)  *sin( pi*z)^2*N[3]
    # u[:,:,:,1] .= 2
    # u[:,:,:,2] .= 1
    # u[:,:,:,3] .= 1
end

leVeque!(sim.flow.u,xList,yList,zList);

rr=project!(sim.flow,sim.pois)
WaterLily.BCVecPerNeu!(sim.flow.u;Dirichlet=true, A=sim.flow.U, perdir=sim.flow.perdir)
save("aLeVequeubase.jld","data",sim.flow.u)

u⁰ = sim.flow.u
Statistics.mean(abs.(
    u⁰[3:end, 2:end-1, 2:end-1,1]-u⁰[2:end-1, 2:end-1, 2:end-1,1]+
    u⁰[2:end-1, 3:end, 2:end-1,2]-u⁰[2:end-1, 2:end-1, 2:end-1,2]+
    u⁰[2:end-1, 2:end-1, 3:end,3]-u⁰[2:end-1, 2:end-1, 2:end-1,3]
))/N[1]


u⁰Mag = sqrt.(sum(u⁰.^2,dims=[4]))[:,:,:,1];

Plots.plot(rr,yscale=:log10)

figures = Vector{GRUtils.Figure}(undef,181)
for d ∈ 0:180
    figures[d+1] = GRUtils.isosurface(u⁰Mag*(sind(d+50)*0.5+1),N[1])
end
GRUtils.videofile(figures, "a.mp4")

# GRUtils.isosurface(u⁰Mag*(sind(90)+1),N[1])

# # Make a plot with example data
# x = LinRange(0, 800, 100)
# y = sind.(x)
# GRUtils.plot(x,y)
# # Make a video sliding over the X axis
# GRUtils.video("mp4") do
#   for d = 0:10:440
#     GRUtils.xlim(d, d+360)
#     GRUtils.draw(GRUtils.gcf())
#   end
# end

# sind(90)+1

# uu = copy(sim.flow.u);
# leVeque!(sim.flow.u,xList,yList,zList);
# uuMag = sqrt.(sum(uu.^2,dims=[4]))[:,:,:,1];

# GLMakie.contour(uuMag,levels=[N[1]],alpha=1,isorange=N[1]/10)


# uDiffMag = sqrt.(sum((sim.flow.u .- uu).^2,dims=[4]))[:,:,:,1];

# maximum(uDiffMag)
