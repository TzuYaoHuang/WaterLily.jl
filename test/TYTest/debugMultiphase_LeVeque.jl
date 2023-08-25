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
using LinearAlgebra
using PyPlot
using GLMakie
GLMakie.activate!()

inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

computationID = "cVOFLeVeque"

Lp = 64
N = (Lp,Lp,Lp)
D = length(N)
Nd = ((N.+2)...,D)

u⁰ = zeros(Nd)
u  = copy(u⁰)


ins = WaterLily.cVOF(
    N, zeros(Nd), zeros(N.+2); 
    InterfaceSDF = (x) -> sqrt.(
        (x[1]-0.35*N[1]-1.5).^2 + (x[2]-0.35*N[2]-1.5).^2 + (x[3]-0.35*N[3]-1.5).^2
    )-N[1]*0.15, 
    perdir=(1,2,3)
)

xList = reshape((1:N[1]+2).-2,(N[1]+2,1,1))/N[1]
yList = reshape((1:N[2]+2).-2,(1,N[2]+2,1))/N[2]
zList = reshape((1:N[3]+2).-2,(1,1,N[3]+2))/N[3]



T = 3

function leVeque!(u,x,y,z,t)
    half = 0.5/N[1]
    u[:,:,:,1] = @. 2*sin( pi*x)^2*sin(2pi*(y.+half))  *sin(2pi*(z.+half))  *cos(pi*t/T)*N[1]
    u[:,:,:,2] = @.  -sin(2pi*(x.+half))  *sin( pi*y)^2*sin(2pi*(z.+half))  *cos(pi*t/T)*N[2]
    u[:,:,:,3] = @.  -sin(2pi*(x.+half))  *sin(2pi*(y.+half))  *sin( pi*z)^2*cos(pi*t/T)*N[3]
end


massConserv = []
push!(massConserv,Statistics.mean(ins.f[2:end-1,2:end-1,2:end-1]))

dt = 0.0005
t = Array(0:dt:T)


leVeque!(u⁰,xList,yList,zList,t[1])


lx,ly,lz = ((1:N[1]).-0.5)/N[1],((1:N[2]).-0.5)/N[2],((1:N[2]).-0.5)/N[2]


function Insidef!(f,dat)
    copyto!(dat,f[inside(f)])
end

dat = ins.f[inside(ins.f)] |> Array;
obs = Insidef!(ins.f,dat) |> Observable;
fig, ax, lineplot = GLMakie.contour(obs,levels=[0.5],alpha=1,isorange=0.2)

record(fig, computationID*"_"*"fIso.mp4", 2:size(t)[1]; framerate=100) do i
    push!(massConserv,Statistics.mean(ins.f[2:end-1,2:end-1,2:end-1]))
    leVeque!(u,xList,yList,zList,t[i])
    WaterLily.freeint_update!(t[i]-t[i-1], ins.f, ins.fᶠ, ins.n̂, ins.α, u⁰, u, ins.c̄, perdir=ins.perdir, dirdir=ins.dirdir)
    u⁰ .= u
    obs[] = Insidef!(ins.f,dat)
    if i%10==0
        println(i)
    end
end

Plots.plot()
Plots.plot((massConserv .-massConserv[1])/massConserv[1])
Plots.savefig(computationID*"_masserror.png")
