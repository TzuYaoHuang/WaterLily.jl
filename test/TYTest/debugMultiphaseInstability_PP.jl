JLDFilePath = @__FILE__
workdir = dirname(JLDFilePath)
waterlilypath = dirname(dirname(workdir))*"/src/WaterLily.jl"
include(waterlilypath)
WaterLily = Main.WaterLily;

using Printf
using JLD2
using Plots
using Statistics
using StatsBase
using GLMakie
GLMakie.activate!()

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
    for I ∈ R
        storage[I] = WaterLily.div(I,u)
    end
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

function KE(u,f,λρ)
    N,n = WaterLily.size_u(u)
    ke = zeros(eltype(f),2)
    for i∈1:n
        for I ∈ WaterLily.inside_uWB(N,i)
            buf = u[I,i]^2
            fWater = WaterLily.ϕ(i,I,f)
            if (I[i] == 2) || (I[i] == N[i])
                ke[1] += 1*fWater*buf*0.5
                ke[2] += λρ*(1-fWater)*buf*0.5
            else
                ke[1] += 1*fWater*buf
                ke[2] += λρ*(1-fWater)*buf
            end
        end
    end
    return ke/2
end

function ComputeVorticity!(vortVec, u, R)
    for I∈R 
        vortVec[I,:] = WaterLily.ω(I,u)
    end
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
N = 128
q = 1.0
computationID = "3DNewVortexBreak"*string(N)
N = 96
q = 1.0
disturb = 0.02
computationID =  @sprintf("3DNewVortexBreak%d_q%.2f_dis%.2f",N,q,disturb)
println("You are now processing: "*computationID)

# READ the configuration
JLDFile = jldopen("JLD2/"*computationID*"General.jld2")

UScale = JLDFile["U"]
LScale = JLDFile["L"]
try
    global λρ = JLDFile["rhoRatio"]; global λμ = JLDFile["visRatio"]
catch err
    global λρ = 1e-3; global λμ = 1e-2
end

trueTime = JLDFile["trueTime"]; trueTime .*= UScale/LScale
dtTrueTime = trueTime[2:end] .- trueTime[1:end-1]
dts = JLDFile["dts"]; dts .*= UScale/LScale

resIni = JLDFile["resIni"]
resFin = JLDFile["resFin"]
poisN = JLDFile["poisN"]

close(JLDFile)

# DERIVED configuration
NTime = length(trueTime)
ReportFreq = NTime÷50
xcen = ((1:N).-0.5.-N/2)/LScale
xedg = ((1:N).-1.0.-N/2)/LScale

# DECLARE necessary variables
# storage
VelocityStore = zeros(N+2,N+2,N+2,3)
VOFStore = zeros(N+2,N+2,N+2)
DivergenceStore = zeros(N+2,N+2,N+2)
VorticityStore = zeros(N+2,N+2,N+2,3)

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
ωe = zeros(NTime,2)
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
frameRate = 50

animXSlice = Animation()
animZSlice = Animation()
dat = VOFStore[inside(VOFStore)] |> Array;
obs = Insidef!(VOFStore,dat) |> Observable;
fig, ax, lineplot = GLMakie.contour(obs,levels=[0.5],alpha=1,isorange=0.3)

@time record(fig, computationID*"_"*"fIso.mp4", 1:NTime; framerate=frameRate) do iTime
    # Read in the file
    JLDFile = jldopen("JLD2/"*computationID*"VelVOF_"*string(iTime-1)*".jld2")
    VelocityStore .= JLDFile["u"]; VelocityStore ./= UScale
    VOFStore .= JLDFile["f"]

    # Post-process the data
    CalculateDivergence!(DivergenceStore,VelocityStore,insidef)
    ComputeVorticity!(VorticityStore, VelocityStore, insidef); VorticityStore .*= LScale
    avgVOF[iTime] = CalculateMeanScalar(VOFStore,R=insidef)
    avgDiv[iTime] = CalculateMeanScalar(DivergenceStore,func=abs,R=insidef)
    ke[iTime,:] = KE(VelocityStore,VOFStore,λρ)
    ωe[iTime,:] = KE(VorticityStore,VOFStore,λρ)

    if false
        StaggerToCollocateVel!(VelocityStore, VelocityAtCollocated)
        ToCylindricalVel!(VelocityAtCollocated, VelocityCylatCollocated, rMat, CosThetaMat, SinThetaMat)
        ComputeMeanU!(VelocityCylatCollocated, uMeanRadial, uMeanAzimuthal, uMeanAxial, rMat, rGaussianMat)

        uMeanAxialList[iTime,:] = uMeanAxial
        uMeanAzimuthalList[iTime,:] = uMeanAzimuthal
        uMeanRadialList[iTime,:] = uMeanRadial
    end

    if false
        WaterLily.vof_reconstruct!(VOFStore,inteceStorage,normalStorage;perdir=(1,2,3),dirdir=(0,))
        WaterLily.InitilizeBubbleInfo!(bInfo)
        WaterLily.ICCL_M!(bInfo,1 .- VOFStore,θs,normalStorage)
        bubbleR[iTime] = [bubble.r for (label,bubble) ∈ bInfo.bubbleDict]
    end

    
    obs[] = Insidef!(VOFStore,dat)

    
    midSlice = N÷2+1

    # Plot X slice
    Plots.plot()
    Plots.contourf!(xedg,xedg,clamp.(VorticityStore[midSlice,2:end-1,2:end-1,1]',-10,10), aspect_ratio=:equal,color=:seismic,levels=60,xlimit=[-4,4],ylimit=[-4,4],linewidth=0,clim=(-10,10))
    Plots.contour!(xcen,xcen,VOFStore[midSlice,2:end-1,2:end-1]', aspect_ratio=:equal,color=:Black,levels=[0.5],xlimit=[-4,4],ylimit=[-4,4],linewidth=2)
    frame(animXSlice,Plots.plot!())

    # Plot Z slice
    Plots.plot()
    Plots.contourf!(xedg,xedg,clamp.(VorticityStore[2:end-1,2:end-1,midSlice,3]',-10,10), aspect_ratio=:equal,color=:seismic,levels=60,xlimit=[-4,4],ylimit=[-4,4],linewidth=0,clim=(-10,10))
    Plots.contour!(xcen,xcen,VOFStore[2:end-1,2:end-1,midSlice]', aspect_ratio=:equal,color=:Black,levels=[0.5],xlimit=[-4,4],ylimit=[-4,4],linewidth=2)
    frame(animZSlice,Plots.plot!())

    close(JLDFile)

    if mod(iTime,ReportFreq) == 1
        @printf("%6.2f%% (%5d/%5d) of files being processed.\n", iTime/NTime*100, iTime, NTime)
    end
end
gif(animXSlice, computationID*"_"*"xSlice.gif", fps=frameRate)
gif(animZSlice, computationID*"_"*"zSlice.gif", fps=frameRate)


# GLOBAL Plot
allTime = cumsum(dts)[1:end-1]
midTrueTime = (trueTime[2:end]+trueTime[1:end-1])/2
ke ./= LScale^3
ωe ./= LScale^3
keT = ke[:,1] .+ ke[:,2]
ωeT = ωe[:,1] .+ ωe[:,2]
dωeTdt = (ωeT[2:end]-ωeT[1:end-1])./dtTrueTime; dωeTdt[1] = dωeTdt[2]
dkeTdt = (keT[2:end]-keT[1:end-1])./dtTrueTime; dkeTdt[1] = dkeTdt[2]
ωeTMid = (ωeT[2:end]+ωeT[1:end-1])/2
effectiveRe = -2*ωeTMid./dkeTdt
quantileRe = quantile(effectiveRe)
RelVOF = abs.((avgVOF.-avgVOF[1])/avgVOF[1]).+1e-20

# Divergence and mass conservation
Plots.plot()
Plots.plot!(trueTime,avgDiv.+1e-20,label="Velocity Divergence" ,color=:red)
Plots.plot!(trueTime,RelVOF,label="Mass loss",color=:blue)
Plots.plot!(ylimit=[1e-10,1],yaxis=:log10)
Plots.savefig(computationID*"_MassDivergence.png")

# Energy
Plots.plot()
Plots.plot!(trueTime,keT,label="K.E. All",color=:black,linestyle=:solid)
# Plots.plot!(trueTime,ke[:,1],label="K.E. Water",color=:blue,linestyle=:dot)
# Plots.plot!(trueTime,ke[:,2],label="K.E. Air",color=:green,linestyle=:dash)
Plots.savefig(computationID*"_Energy.png")

# Enstrophy
Plots.plot()
Plots.plot!(trueTime,ωeT,legend=false,color=:black,linestyle=:solid)
Plots.savefig(computationID*"_Enstrophy.png")

Plots.plot()
Plots.plot!(midTrueTime,dωeTdt,legend=false,color=:black,linestyle=:solid)
Plots.savefig(computationID*"_DiffEnstrophy.png")

Plots.plot()
Plots.plot!(midTrueTime,effectiveRe,color=:black,linestyle=:solid)
Plots.hline!(quantileRe[3:4],color=:blue,linestyle=:dash)
Plots.hline!(quantileRe[2:2],color=:deepskyblue4,linestyle=:dash)
Plots.plot!(title=@sprintf("Median Re: %.2f, 3rd quantile Re: %.2f", quantileRe[3], quantileRe[4]),legend=false)
Plots.savefig(computationID*"_EffectiveRe.png")

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
    Plots.histogram!(log10.(bubbleR[iTime]),bins=-4.4:0.1:2.0,color=:gray)
    Plots.plot!(ylimit=(0,300))
    frame(aBubbleDistribution,Plots.plot!())
end
gif(aBubbleDistribution,computationID*"_bubbleDistri.gif",fps=frameRate)
