JLDFilePath = @__FILE__
workdir = dirname(JLDFilePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"
include(waterlilypath)
using .WaterLily

using Printf
using JLD2
using Plots
using Plots.PlotMeasures
using Statistics
using StatsBase
using WriteVTK
using StaticArrays
using LaTeXStrings
using Loess
ENV["GKSwstype"]="nul"

default()
Plots.scalefontsizes()
default(fontfamily="Palatino",linewidth=2, framestyle=:axes, label=nothing, grid=false, tick_dir=:out, size=(900,700),right_margin=5mm,left_margin=5mm,top_margin=5mm,bottom_margin=5mm)
Plots.scalefontsizes(2.1)

# DEFINE some useful functions
animAlpha(i,numFiles;y0 = 0.01) = 4*(1-y0)*(i/numFiles-0.5)^3 + (1+y0)/2
animAlpha(i,numFiles;y0 = 0.1) = (1-y0)*(i/numFiles)^2+y0
inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

# SUPPORTED functions
function CalculateDivergence!(storage,u,R)
    WaterLily.@loop storage[I] = WaterLily.div(I,u) over I∈R
end

function CalculateMeanScalar(f;func=(x)->x,R=inside(f))
    s = 0.0
    R = inside(f)
    count = length(R)
    for I ∈ R
        s += func(f[I])
    end
    return s/count
end

function keI(I::CartesianIndex{n},u::AbstractArray{T}) where {n,T}
    ke = zero(eltype(u))
    for i∈1:n
        ke += (u[I,i]^2+u[I+δ(i,I),i]^2)/2
    end
    return 0.5ke
end

function ωeI(I::CartesianIndex{n},ω::AbstractArray{T}) where {n,T}
    ωe = zero(eltype(ω))
    for i∈1:n
        for II ∈ I:(I+oneunit(I)-δ(i,I))
            ωe += ω[II,i]^2/4
        end
    end
    return 0.5ωe
end

function SeI(I::CartesianIndex{n},u::AbstractArray{T}) where {n,T}
    J = @SMatrix [WaterLily.∂(i,j,I,u) for i ∈ 1:3, j ∈ 1:3]
    S = 0.5(J+J')
    return sum(S.^2)
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

function E(fun,u,f,λρ)
    e = zero(eltype(f))
    for I ∈ inside(f)
        e += fun(I,u)
    end
    return e
end

function ComputeVorticity!(vortVec, u, R)
    WaterLily.@loop vortVec[I,:] = WaterLily.ω(I,u) over I∈R
    WaterLily.BCPerVec!(vortVec)
end

function Insidef!(f,dat)
    copyto!(dat,f[inside(f)])
end

function StaggerToCollocateVel!(u, uInside)
    uInside .= 0.0
    uInside[:,:,:,1] = 0.5*(u[2:end-1,2:end-1,2:end-1,1]+u[3:end,2:end-1,2:end-1,1])
    uInside[:,:,:,2] = 0.5*(u[2:end-1,2:end-1,2:end-1,2]+u[2:end-1,3:end,2:end-1,2])
    uInside[:,:,:,3] = 0.5*(u[2:end-1,2:end-1,2:end-1,3]+u[2:end-1,2:end-1,3:end,3])
end

function ToCylindricalVel!(uInside, uCyl, rMat, CosThetaMat, SinThetaMat)
    uCyl .= 0
    u,v,w = uInside[:,:,:,1],uInside[:,:,:,2],uInside[:,:,:,3]
    uCyl[:,:,:,1] = u .* CosThetaMat .+ v .* SinThetaMat
    uCyl[:,:,:,2] =-u .* SinThetaMat .+ v .* CosThetaMat
    uCyl[:,:,:,3] = w*1.0
end

function ComputeMeanU!(uCyl, uMeanRadial, uMeanAzimuthal, uMeanAxial, rMat, rGaussianMat)
    uCylFlat = Statistics.mean(uCyl,dims=3)
    gridSize = size(uCylFlat)
    uMeanRadial .= 0
    uMeanAzimuthal .= 0
    uMeanAxial .= 0
    sumR = similar(uMeanAxial)*0
    for I in CartesianIndices(gridSize[1:3])
        loc = Int(round(rMat[I]))
        if loc < length(uMeanAxial)
            uMeanRadial[loc] += uCylFlat[I,1]*rGaussianMat[I]
            uMeanAzimuthal[loc] += uCylFlat[I,2]*rGaussianMat[I]
            uMeanAxial[loc] += uCylFlat[I,3]*rGaussianMat[I]
            sumR[loc] += rGaussianMat[I]
        end
    end
    uMeanRadial ./= sumR
    uMeanAzimuthal ./= sumR
    uMeanAxial ./= sumR
end

# CASE configuration
N = 192
q = 1.00
disturb = 0.1
VOFdisturb = 0.0
m = 0
Axialq = 1.0
computationID =  @sprintf("3DVBSmooth_N%d_m%d_q%.2f_qA%.2f_Urdis%.2f_VOFdis%.2f",N,m,q,Axialq,disturb,VOFdisturb)
println("You are now processing: "*computationID); flush(stdout)

# READ the configuration
JLDFile = jldopen("JLD2/"*computationID*"General.jld2")

UScale = JLDFile["U"]
LScale = JLDFile["L"]
LDomain = N
try
    global λρ = JLDFile["rhoRatio"]; global λμ = JLDFile["visRatio"]
catch err
    global λρ = 1e-3; global λμ = 1e-2
end

trueTime = JLDFile["trueTime"]; trueTime .*= UScale/LScale
T = eltype(trueTime)
timeLimit = [minimum(trueTime),maximum(trueTime)]
dtTrueTime = trueTime[2:end] .- trueTime[1:end-1]
dtTrueTime[dtTrueTime .<= 10eps(T)] .= median(dtTrueTime)
dts = JLDFile["dts"]; dts .*= UScale/LScale

resIni = JLDFile["resIni"]
resFin = JLDFile["resFin"]
poisN = JLDFile["poisN"]

close(JLDFile)

# DERIVED configuration
NTime = length(trueTime)
ReportFreq = max(NTime÷50,1)
xcen = ((0:N+1).-0.5.-N/2)/LScale
xedg = ((0:N+1).-1.0.-N/2)/LScale

# DECLARE necessary variables
# storage
VelocityStore = zeros(N+2,N+2,N+2,3)
VOFStore = zeros(N+2,N+2,N+2)
DivergenceStore = zeros(N+2,N+2,N+2)
VorticityStore = zeros(N+2,N+2,N+2,3)
λ2Store = zeros(N+2,N+2,N+2)

VelocityAtCollocated = zeros(N,N,N,3)
VelocityCylatCollocated = zeros(N,N,N,3)
xCoord = reshape(Array((1:N).-0.5 .-N/2), N, 1, 1)
yCoord = reshape(Array((1:N).-0.5 .-N/2), 1, N, 1)
rMat = sqrt.(xCoord.^2 .+ yCoord.^2)
thetaMat = atan.(yCoord,xCoord)
CosThetaMat = cos.(thetaMat)
SinThetaMat = sin.(thetaMat)
cylGridSize = ntuple(i -> i==1 ? N÷2-2 : N, 3)
rList = (1:cylGridSize[1]) .- 0.5
thetaList = (1:cylGridSize[2])/cylGridSize[2]*2*pi
zList = (1:cylGridSize[3]) .- 0.5
uMeanRadial = zeros(cylGridSize[1])
uMeanAzimuthal = zeros(cylGridSize[1])
uMeanAxial = zeros(cylGridSize[1])
rGaussianMat = exp.(-0.2*(rMat .- Int.(round.(rMat))).^2)


# global
insidef = inside(VOFStore)
avgVOF = zeros(NTime)
avgDiv = zeros(NTime)
ke = zeros(NTime,2)
ωe = zeros(NTime)
Se = zeros(NTime)
uMeanRadialList = zeros(NTime,cylGridSize[1])
uMeanAzimuthalList = zeros(NTime,cylGridSize[1])
uMeanAxialList = zeros(NTime,cylGridSize[1])


# BubblesInfo
labelStorage = zeros(Int32,(N+2,N+2,N+2))
bInfo = WaterLily.BubblesInfo(labelStorage)
bubbleR = Array{Vector{Float64},1}(undef,NTime)
θs = [0.0,0.0]
normalStorage = zeros((N+2,N+2,N+2,3))
inteceStorage = zeros((N+2,N+2,N+2))

# ANIMATION
frameRate = 80

animXSlice = Animation()
animZSlice = Animation()

@time for iTime ∈ 1:NTime
    # Read in the file
    JLDFile = jldopen("JLD2/"*computationID*"VelVOF_"*string(iTime-1)*".jld2")
    VelocityStore .= JLDFile["u"]; VelocityStore ./= UScale
    λ2Store .= JLDFile["f"]
    WaterLily.vof_smooth!(2,λ2Store,VOFStore,DivergenceStore,perdir=(1,2,3))

    # Post-process the data
    CalculateDivergence!(DivergenceStore,VelocityStore,insidef)
    ComputeVorticity!(VorticityStore, VelocityStore, insidef); VorticityStore .*= LScale
    avgVOF[iTime] = CalculateMeanScalar(VOFStore,R=insidef)
    avgDiv[iTime] = CalculateMeanScalar(DivergenceStore,func=abs,R=insidef)
    ke[iTime,:] = ρE(keI,VelocityStore,VOFStore,λρ)
    ωe[iTime] =  E(ωeI,VorticityStore,VOFStore,λρ)
    Se[iTime] =  E(SeI,VelocityStore,VOFStore,λρ)
    

    if true #&& ((iTime-1)%4==0)
        WaterLily.@loop λ2Store[I] = WaterLily.λ₂(I,VelocityStore)*LScale^2 over I∈inside(λ2Store)
        WaterLily.BCPer!(λ2Store)
        vtk_grid("VTK/"*computationID*"VelVOF_"*string(iTime-1)*".vti", xcen, xcen, xcen) do vtk
            vtk["VOF"] = VOFStore
            # vtk["Vel"] = @views (VelocityStore[inside(VOFStore),1],VelocityStore[inside(VOFStore),2],VelocityStore[inside(VOFStore),3])
            vtk["l2"] = λ2Store
        end
    end

    if false
        StaggerToCollocateVel!(VelocityStore, VelocityAtCollocated)
        ToCylindricalVel!(VelocityAtCollocated, VelocityCylatCollocated, rMat, CosThetaMat, SinThetaMat)
        ComputeMeanU!(VelocityCylatCollocated, uMeanRadial, uMeanAzimuthal, uMeanAxial, rMat, rGaussianMat)

        uMeanAxialList[iTime,:] = uMeanAxial
        uMeanAzimuthalList[iTime,:] = uMeanAzimuthal
        uMeanRadialList[iTime,:] = uMeanRadial
    end

    if false
        useNormConnect=false
        useNormConnect && WaterLily.vof_reconstruct!(VOFStore,inteceStorage,normalStorage;perdir=(1,2,3),dirdir=(0,))
        WaterLily.InitilizeBubbleInfo!(bInfo)
        WaterLily.ICCL_M!(bInfo,1 .- VOFStore,θs,normalStorage,useNormConnect=useNormConnect)
        bubbleR[iTime] = [bubble.r for (label,bubble) ∈ bInfo.bubbleDict]
    end
    
    midSlice = N÷2+1

    # Plot X slice
    Plots.plot(size=(800,700))
    VorticitySlice = (VorticityStore[midSlice,:,:,1] + VorticityStore[midSlice+1,:,:,1])/2
    VOFSlice = (VOFStore[midSlice,:,:]+VOFStore[midSlice+1,:,:])/2
    Plots.contourf!(xedg,xedg,clamp.(VorticitySlice',-10,10), aspect_ratio=:equal,color=:seismic,levels=60,xlimit=[-4,4],ylimit=[-4,4],linewidth=0,clim=(-10,10))
    Plots.contour!(xcen,xcen,VOFSlice', aspect_ratio=:equal,color=:Black,levels=[0.5],xlimit=[-4,4],ylimit=[-4,4],linewidth=2)
    Plots.plot!(xlabel=L"y",ylabel=L"z",colorbar_title=L"\omega_x")
    frame(animXSlice,Plots.plot!())

    # Plot Z slice
    Plots.plot(size=(800,700))
    Plots.contourf!(xedg,xedg,clamp.(VorticityStore[:,:,midSlice,3]',-10,10), aspect_ratio=:equal,color=:seismic,levels=60,xlimit=[-4,4],ylimit=[-4,4],linewidth=0,clim=(-10,10))
    Plots.contour!(xcen,xcen,VOFStore[:,:,midSlice]', aspect_ratio=:equal,color=:Black,levels=[0.5],xlimit=[-4,4],ylimit=[-4,4],linewidth=2)
    Plots.plot!(xlabel=L"y",ylabel=L"z",colorbar_title=L"\omega_z")
    frame(animZSlice,Plots.plot!())

    close(JLDFile)

    if mod(iTime,ReportFreq) == 1
        @printf("%6.2f%% (%5d/%5d) of files have been processed.\n", iTime/NTime*100, iTime, NTime); flush(stdout)
    end
end
gif(animXSlice, computationID*"_"*"xSlice.gif", fps=frameRate)
gif(animZSlice, computationID*"_"*"zSlice.gif", fps=frameRate)


# GLOBAL Plot
allTime = cumsum(dts)[1:end-1]
midTrueTime = (trueTime[2:end]+trueTime[1:end-1])/2
ke ./= LDomain^3
ωe ./= LDomain^3
Se ./= LDomain^3/LScale^2
keT = ke[:,1] .+ ke[:,2]
ωeT = ωe#[:,1] .+ ωe[:,2]
SeT = Se#[:,1] .+ Se[:,2]
dωeTdt = (ωeT[2:end]-ωeT[1:end-1])./dtTrueTime; dωeTdt[1] = dωeTdt[2]
dkeTdt = (keT[2:end]-keT[1:end-1])./dtTrueTime; dkeTdt[1] = dkeTdt[2]
dSeTdt = (SeT[2:end]-SeT[1:end-1])./dtTrueTime; dSeTdt[1] = dSeTdt[2]
ωeTMid = (ωeT[2:end]+ωeT[1:end-1])/2
SeTMid = (SeT[2:end]+SeT[1:end-1])/2
effectiveRe = -ωeTMid./dkeTdt
effectiveReS = -SeTMid./dkeTdt
quantileRe = quantile(effectiveRe)
quantileReS = quantile(effectiveReS)
RelVOFFirstStep = abs.((avgVOF.-avgVOF[1])/avgVOF[1]).+1e-20
RelVOFPrevStep = abs.((avgVOF[2:end].-avgVOF[1:end-1])/avgVOF[1]).+1e-20

jldsave("JLD2/"*computationID*"_EnstrophyAndEnergy.jld2";trueTime, ens=ωeT, s=SeT, ener=keT)

model = loess(midTrueTime, dωeTdt, span=0.1)
dωeTdtSmooth = predict(model, midTrueTime)

# Divergence and mass conservation
Plots.plot()
Plots.plot!(trueTime,avgDiv.+1e-20,label=L"\mathbf{\nabla\cdot u}" ,color=:red)
Plots.plot!(trueTime,RelVOFFirstStep,label="Accumulated Mass loss",color=:blue)
Plots.plot!(midTrueTime,RelVOFPrevStep,label="Mass loss (w.r.t. prev)",color=:blue)
Plots.plot!(xlimit=timeLimit, ylimit=[1e-10,1],yaxis=:log10)
Plots.plot!(xlabel=L"t")
Plots.savefig(computationID*"_MassDivergence.png")

# Energy
Plots.plot()
Plots.plot!(trueTime,keT,color=:black,linestyle=:solid)
Plots.plot!(xlimit=timeLimit,xlabel=L"t",ylabel=L"\textrm{Energy } (E)")
Plots.savefig(computationID*"_Energy.png")

# Enstrophy
m = Plots.plot()
p = Plots.twinx(m)
Plots.plot!(p,trueTime,ωeT,legend=false,color=:grey,linestyle=:solid)
Plots.plot!(m,midTrueTime,dωeTdtSmooth,legend=false,color=:blue,linestyle=:solid)
Plots.hline!(m,[0],legend=false,color=:blue,linestyle=:dash,linewidth=1.2)
Plots.plot!(m,xlimit=timeLimit,ylabel=L"\mathrm{d} \mathcal{E}/\mathrm{d} t")
Plots.plot!(p,ylabel=L"\textrm{Enstrophy } (\mathcal{E})",xlabel="\n"*L"t")
Plots.plot!(m,y_foreground_color_axis=:blue,y_foreground_color_text=:blue,y_foreground_color_border=:blue)
Plots.plot!(p,y_foreground_color_axis=:grey,y_foreground_color_text=:grey,y_foreground_color_border=:grey)
Plots.savefig(computationID*"_DiffEnstrophy.png")

Plots.plot()
Plots.plot!(midTrueTime,effectiveRe,color=:black,linestyle=:solid)
Plots.hline!(quantileRe[3:4],color=:blue,linestyle=:dash)
Plots.hline!(quantileRe[2:2],color=:deepskyblue4,linestyle=:dash)
Plots.plot!(title=@sprintf("Median Re: %.2f, 3rd quantile Re: %.2f", quantileRe[3], quantileRe[4]),titlefontsize=24,legend=false, ylimit=(0,5000))
Plots.plot!(xlimit=timeLimit,xlabel=L"t",ylabel=L"\mathrm{Re}_\mathrm{eff} = Ud_\mathrm{v}/\nu_\mathrm{eff}")
Plots.savefig(computationID*"_EffectiveRe.png")

Plots.plot()
Plots.plot!(midTrueTime,effectiveReS,color=:black,linestyle=:solid)
Plots.hline!(quantileReS[3:4],color=:blue,linestyle=:dash)
Plots.hline!(quantileReS[2:2],color=:deepskyblue4,linestyle=:dash)
Plots.plot!(title=@sprintf("Median Re: %.2f, 3rd quantile Re: %.2f", quantileReS[3], quantileReS[4]),titlefontsize=24,legend=false, ylimit=(0,5000))
Plots.plot!(xlimit=timeLimit,xlabel=L"t",ylabel=L"\mathrm{Re}_\mathrm{eff} = Ud_\mathrm{v}/\nu_\mathrm{eff}")
Plots.savefig(computationID*"_EffectiveReS.png")

m = Plots.plot()
Plots.plot!(m,midTrueTime,dωeTdtSmooth,legend=false,color=:blue,linestyle=:solid)
Plots.hline!(m,[0],legend=false,color=:blue,linestyle=:dash,linewidth=1.2)
Plots.plot!(m,xlimit=timeLimit,xlabel=L"t",ylabel=L"\mathrm{d} \mathcal{E}/\mathrm{d} t")
# Plots.plot!(m,y_foreground_color_axis=:blue,y_foreground_color_text=:blue,y_foreground_color_border=:blue)
Plots.savefig(computationID*"_DiffEnstrophyOnly.png")

# Check Poisson solver
Plots.plot()
Plots.plot!(allTime,resIni[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(allTime,resIni[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisResInitial.png")

Plots.plot()
Plots.plot!(allTime,resFin[1:2:end],yscale=:log10,label="First Stage")
Plots.plot!(allTime,resFin[2:2:end],yscale=:log10,label="Second Stage",legend=:bottomleft)
Plots.savefig(computationID*"_PoisResFinal.png")

Plots.plot()
Plots.plot!(allTime,poisN[1:2:end],label="First Stage")
Plots.plot!(allTime,poisN[2:2:end],label="Second Stage")
Plots.savefig(computationID*"_PoisNum.png")

Plots.plot()
Plots.plot!(allTime,dts[1:end-1])
Plots.savefig(computationID*"_delT.png")

aMeanUAzi = Animation()
Plots.plot()
for iTime ∈ 1:NTime
    Plots.plot()
    Plots.plot!(rList/LScale, color=:blue, (@view uMeanAzimuthalList[iTime,:]),ylim=(0,max(0.8*q,0.2)),legend=false)
    frame(aMeanUAzi,Plots.plot!())
end
gif(aMeanUAzi,computationID*"_meanUAzi.gif", fps=frameRate)

aBubbleDistribution = Animation() 
for iTime ∈ 1:NTime
    Plots.plot()
    Plots.histogram!(log10.(bubbleR[iTime]/LScale),bins=-6.9:0.1:0.5,color=:gray)
    Plots.vline!([log10(1/LScale)],color=:blue,linewidth=2,linestyle=:dash,label=false)
    Plots.plot!(ylimit=(0,400))
    frame(aBubbleDistribution,Plots.plot!(legend=false))
end
gif(aBubbleDistribution,computationID*"_bubbleDistri.gif",fps=frameRate)
