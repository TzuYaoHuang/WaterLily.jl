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

computationID = "ThreeDVortex_256"

inside(a::AbstractArray) = CartesianIndices(map(ax->first(ax)+1:last(ax)-1,axes(a)))
@inline CI(a...) = CartesianIndex(a...)
"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())

function InterpolOmega(xOrig,omega)
    x = xOrig .+ 2.0
    floorx = floor.(x)
    I = CartesianIndex(Int.(floorx)...)
    residual = x.-floorx
    N = length(I)
    s = 0.0
    for i ∈ N
        s += omega[I]*(1-residual[i]) + omega[I+δ(i,I)]*residual[i]
    end
    return s/N
end

function ComputeVorticity!(u, vorticityMag)
    for I∈inside(vorticityMag) 
        vorticityMag[I] = WaterLily.ω_mag(I,u)
    end
end

function ComputeEntrophySpectrum!(vorticityMag, vortSpectrum)
    vortSpectrum .= 0.0
    gridSize = size(vorticityMag)
    ω̂ = fft(vorticityMag)/prod(gridSize);
    lx = fftfreq(gridSize[1],gridSize[1])
    ly = fftfreq(gridSize[2],gridSize[2])
    lz = fftfreq(gridSize[3],gridSize[3])
    for I in CartesianIndices(gridSize)
        l = sqrt(lx[I[1]]^2+ly[I[2]]^2+lz[I[3]]^2)
        (l<0.5) && continue
        vortSpectrum[Int(round(l))] += abs2(ω̂[I])
    end
    vortSpectrum .+=1e-8
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
        uMeanRadial[loc] += uCylFlat[I,1]*rGaussianMat[I]
        uMeanAzimuthal[loc] += uCylFlat[I,2]*rGaussianMat[I]
        uMeanAxial[loc] += uCylFlat[I,3]*rGaussianMat[I]
        sumR[loc] += rGaussianMat[I]
    end
    uMeanRadial ./= sumR
    uMeanAzimuthal ./= sumR
    uMeanAxial ./= sumR
end

function ProjectCylMeanToCartesian!(uMeanFromCyl, uMeanRadial, uMeanAzimuthal, uMeanAxial, rMat, rGaussianMat)
    r = 1 : (size(uMeanRadial)[1])
    uRadialInterpolator = linear_interpolation(r, uMeanRadial, extrapolation_bc=Interpolations.Line())
    uAzimuthalInterpolator = linear_interpolation(r, uMeanAzimuthal, extrapolation_bc=Interpolations.Line())
    uAxialInterpolator = linear_interpolation(r, uMeanAxial, extrapolation_bc=Interpolations.Line())
    uMeanFromCyl[:,:,1,1] = uRadialInterpolator.(rMat)
    uMeanFromCyl[:,:,1,2] = uAzimuthalInterpolator.(rMat)
    uMeanFromCyl[:,:,1,3] = uAxialInterpolator.(rMat)
end

function ComputeFluc!(uMeanFromCyl, uCyl, uFluc)
    uFluc .= uCyl.-uMeanFromCyl
end

function ComputePlanarTKE!(uFluc)
    return dropdims(Statistics.mean(sum(uFluc.^2,dims=4),dims=3),dims=(3,4))*0.5
end

using GLMakie
GLMakie.activate!()
function makie_video!(makie_plot,sim,dat,obs_update!;remeasure=false,name="file.mp4",duration=1,step=0.1,framerate=30,compression=20)
    # Set up viz data and figure
    obs = obs_update!(dat,sim) |> Observable;
    fig, _, _ = makie_plot(obs)
    
    # Run simulation and update figure data
    t₀ = round(WaterLily.sim_time(sim))
    t = range(t₀,t₀+duration;step)
    iterations = size(t)[1]
    i = 1
    save("JLDs/"*computationID*"_"*string(i)*".jld","data",sim.flow.u)
    record(fig, name, t; framerate, compression) do tᵢ
        WaterLily.sim_step!(sim,tᵢ;remeasure)
        obs[] = obs_update!(dat,sim)
        i += 1
        ((i%10) == 1) && save("JLDs/"*computationID*"_"*string(i)*".jld","data",sim.flow.u)
        println("simulation ",round(Int,(tᵢ-t₀)/duration*100),"% complete")
    end
    return fig
end

using Meshing, GeometryBasics
function body_mesh(sim,t=0)
    a = sim.flow.σ; R = inside(a)
    WaterLily.measure_sdf!(a,sim.body,t)
    normal_mesh(GeometryBasics.Mesh(a[R]|>Array,MarchingCubes(),origin=Vec(0,0,0),widths=size(R)))
end;
function flow_λ₂!(dat,sim)
    a = sim.flow.σ
    for I∈inside(sim.flow.σ)
        a[I] = max(0,log10(-min(-1e-6,WaterLily.λ₂(I,sim.flow.u)*(sim.L/sim.U)^2))+.25)
    end
    copyto!(dat,a[inside(a)])                  # copy to CPU
end
function flow_λ₂(sim)
    dat = sim.flow.σ[inside(sim.flow.σ)] |> Array
    flow_λ₂!(dat,sim)
    dat
end


function qVortex(; pow=8, Re=4000, T=Float32, mem=Array)
    # Define vortex size, velocity, viscosity
    Lp = 2^pow  # The grid of WaterLily is always Δx=Δy=Δz=1, so the length of domain is exactly the number of grids.
    delta0 = 1/16*Lp*0.892  # vortex core radius as 1/16 of the domain such that λ = 8dᵥ
    dᵥ = 2/0.892 # normalize Dv
    λ = 8dᵥ
    U = 4.0
    q = 1.0
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
        uRadial = 0.02U/r*(1-exp(-r^2))/0.63817*sin(2*pi/λ*z)*compactRSupport
        uAxial  =     U*(1-exp(-r^2))  # wake-like axial flow

        zConnect = 1.0

        # return the velocity component in cartisian coordiante
        i==1 && return  uTheta*-sin(theta)+uRadial*cos(theta) # u_x
        i==2 && return  uTheta* cos(theta)+uRadial*sin(theta) # u_y
        i==3 && return  uAxial*zConnect    # u_z
        return 0
    end

    # Initialize simulation
    # (Lp, Lp, Lp) is the domain size, 
    # (0, 0, 0) specified the boundary BC. due to sepcial treatment in WaterLily, it is still slip instead of no-slip BC
    # delta0, U together define the length and velocity scales to normalize the simulation time
    return WaterLily.Simulation((Lp, Lp, Lp), (0, 0, 0), delta0; U, uλ, ν, T, mem)
end

pow = 8

sim = qVortex(pow=pow)
#sim.flow.u .+= uvw;

# Create a video using Makie
dat = sim.flow.σ[inside(sim.flow.σ)] |> Array; # CPU buffer array
function λ₂!(dat,sim)  # compute log10(-λ₂), where λ₂ has already been normalized by the prescribed length and time scales
    a = sim.flow.σ
    for I∈inside(sim.flow.σ)
        a[I] = log10(max(1e-6,-WaterLily.λ₂(I,sim.flow.u)*sim.L/sim.U))
    end
    copyto!(dat,a[inside(a)])                  # copy to CPU
end

dur, step = 200, 0.1

@time makie_video!(sim,dat,λ₂!,name=computationID*"_"*"qVortex.mp4",duration=200) do obs
    # plot the iso-surface of normalized λ₂ = 10⁻³, 10⁻², 10⁻¹, 10⁰
    GLMakie.contour(obs,levels=[-3,-2,-1,0],alpha=0.1,isorange=0.5)
end

