using ForwardDiff
using Printf
using JLD2
using Combinatorics
using Statistics: mean
using StaticArrays
using Interpolations
using Random


"""
    cVOF{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}}

Composite type for 2D or 3D two-phase advection scheme.

The heavy fluid is advected using operator-split conservative volume-of-fluid method proposed by [Weymouth & Yue (2010)](https://doi.org/10.1016/j.jcp.2009.12.018).
This guarentees mass conservation and preserves sharp interface across fluids.
The primary variable is the volume fraction of the heavy fluid, the cell-averaged color function, `f`. 
We use it to reconstruct sharp interface.
"""
struct cVOF{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}}
    f :: Sf  # cell-averaged color function (volume fraction field, VOF)
    f⁰:: Sf  # cell-averaged color function. Need to store it because Heun Correction step
    fᶠ:: Sf  # place to store flux or smoothed VOF
    fᵐ:: Sf  # place to store VOF value at a face due to the surface tension step
    n̂ :: Vf  # normal vector of the surfaces in cell
    α :: Sf  # intercept of intercace in cell: norm ⋅ x = α
    c̄ :: AbstractArray{Int8}  # color function at the cell center
    perdir :: NTuple  # periodic directions
    dirdir :: NTuple  # Dirichlet directions
    λρ :: T   # ratio of density (air/water)
    λμ :: T   # ratio of dynamic viscosity (air/water)
    η  :: T   # the surface tension
    ke ::Vector{T}
    keN::Vector{T}
    f1 :: Sf
    function cVOF(N::NTuple{D}; arr=Array, InterfaceSDF::Function=(x) -> 5-x[1], T=Float64, perdir=(0,), dirdir=(0,),λμ=1e-2,λρ=1e-3,η=0) where D
        Ng = N .+ 2  # scalar field size
        Nd = (Ng..., D)  # vector field size
        f = ones(T, Ng) |> arr
        fᶠ = copy(f)
        fᵐ = copy(f)
        n̂ = zeros(T, Nd)
        α = zeros(T, Ng)
        c̄ = zeros(Int8, Ng) |> arr
        applyVOF!(f,α,n̂,InterfaceSDF)
        BCVOF!(f,α,n̂;perdir=perdir,dirdir=dirdir)
        f⁰ = copy(f)
        smoothVOF!(0, f, fᶠ, α;perdir=perdir)
        f1 = ones(T, Ng) |> arr
        new{D,T,typeof(f),typeof(n̂)}(f,f⁰,fᶠ,fᵐ,n̂,α,c̄,perdir,dirdir,λρ,λμ,η,[],[],f1)
    end
end


calke = false

"""
    mom_step!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
"""
@fastmath function mom_step!(a::Flow{D},b::AbstractPoisson,c::cVOF,d::AbstractBody) where D
    a.u⁰ .= a.u;
    smoothStep = 2

    # predictor u → u'
    U = BCTuple(a.U,time(a),D)
    advect!(a,c,c.f,a.u⁰,a.u); measure!(a,d,c;t=0,ϵ=1)
    smoothVOF!(smoothStep, c.f⁰, c.fᶠ, c.α;perdir=c.perdir)
    calke && calke!(a.σ,a.u,c.fᶠ,c.f1,c.λρ,c.ke,c.keN)
    a.u .= 0
    ConvDiffSurf!(a.f,a.u⁰,a.σ,c.f⁰,c.fᶠ,c.fᵐ,c.α,c.n̂,c.λμ,c.λρ,a.ν,c.η,perdir=a.perdir)
    accelerate!(a.f,time(a),a.g,a.U)
    BDIM!(a); BC!(a.u,U,a.exitBC,a.perdir)
    calke && calke!(a.σ,a.u,c.f,c.f1,c.λρ,c.ke,c.keN)
    calculateL!(a,c); update!(b)
    project!(a,b); BC!(a.u,U,a.exitBC,a.perdir)
    calke && calke!(a.σ,a.u,c.f,c.f1,c.λρ,c.ke,c.keN)

    # corrector u → u¹
    U = BCTuple(a.U,timeNext(a),D)
    advect!(a,c,c.f⁰,a.u⁰,a.u); measure!(a,d,c;t=0,ϵ=1)
    smoothVOF!(smoothStep, c.f, c.fᶠ, c.α;perdir=c.perdir)
    ConvDiffSurf!(a.f,a.u,a.σ,c.f,c.fᶠ,c.fᵐ,c.α,c.n̂,c.λμ,c.λρ,a.ν,c.η,perdir=a.perdir)
    accelerate!(a.f,timeNext(a),a.g,a.U)
    BDIM!(a); scale_u!(a,0.5); BC!(a.u,U,a.exitBC,a.perdir)
    calke && calke!(a.σ,a.u,c.fᶠ,c.f1,c.λρ,c.ke,c.keN)
    calculateL!(a,c); update!(b)
    project!(a,b,0.5); BC!(a.u,U,a.exitBC,a.perdir)
    calke && calke!(a.σ,a.u,c.fᶠ,c.f1,c.λρ,c.ke,c.keN)
    c.f .= c.f⁰

    push!(a.Δt,min(CFL(a,c),1.1a.Δt[end]))
end

function ConvDiffSurf!(r,u,Φ,f,fᶠ,fbuffer,α,n̂,λμ,λρ,ν,η;perdir=(0,))
    r .= 0.
    N,n = size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        lowBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N,Val{tagper}())
        # inner cells
        @loop (
            Φ[I] = -calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u));
            r[I,i] += Φ[I];
        ) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N,Val{tagper}())
    end

    η≠0 && surfTen!(r,f,fbuffer,α,n̂,η;perdir)

    for i ∈ 1:n
        @loop r[I,i] /= calculateρ(i,I,fᶠ,λρ) over I ∈ inside(Φ)
    end

    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        lowBoundaryConv!(r,u,Φ,ν,i,j,N,Val{tagper}())
        # inner cells
        @loop (
            Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u));
            r[I,i] += Φ[I];
        ) over I ∈ inside_u(N,j)
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundaryConv!(r,u,Φ,ν,i,j,N,Val{tagper}())
    end
end

# Neumann BC Building block
lowBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N,::Val{false}) = @loop r[I,i] += -calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u)) over I ∈ slice(N,2,j,2)
lowBoundaryConv!(r,u,Φ,ν,i,j,N,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u)) over I ∈ slice(N,2,j,2)
upperBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N,::Val{false}) = @loop r[I-δ(j,I),i] += calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u)) over I ∈ slice(N,N[j],j,2)
upperBoundaryConv!(r,u,Φ,ν,i,j,N,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u)) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N,::Val{true}) = @loop (
    Φ[I] = -calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u));
    r[I,i] += Φ[I]
) over I ∈ slice(N,2,j,2)
lowBoundaryConv!(r,u,Φ,ν,i,j,N,::Val{true}) = @loop (
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u));
    r[I,i] += Φ[I]
) over I ∈ slice(N,2,j,2)
upperBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)
upperBoundaryConv!(r,u,Φ,ν,i,j,N,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)

function measure!(a::Flow,body::NoBody,c::cVOF;t=0,ϵ=1) a.μ₀ .= 1 end
function measure!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody,t=0)
    measure!(a,d,c;t=0,ϵ=1)
    calculateL!(a,c)
    update!(b)
end

@fastmath @inline function MaxTotalflux(I::CartesianIndex{d},u) where {d}
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(abs(u[I,i]),abs(u[I+δ(i,I),i])))
    end
    return s
end

function CFL(a::Flow{D},c::cVOF;Δt_max=10) where D
    @inside a.σ[I] = flux_out(I,a.u)
    fluxLimit = inv(maximum(@views a.σ[inside(a.σ)])+5*a.ν*max(1,c.λμ/c.λρ))
    @inside a.σ[I] = MaxTotalflux(I,a.u)
    cVOFLimit = 0.5*inv(maximum(@views a.σ[inside(a.σ)]))
    surfTenLimit = sqrt((1+c.λρ)/(8π*c.η)) # 8 from kelli's code
    gravLimit = isnothing(a.g) ? Δt_max : 1/√sum((i)->a.g(i,time(a))^2,1:D)
    0.8min(Δt_max,fluxLimit,cVOFLimit,surfTenLimit,gravLimit)
    # 0.03/maximum(abs,a.u)
end

# function myCFL(a::Flow,c::cVOF)
#     @inside a.σ[I] = flux_out(I,a.u)
#     tCFL = inv(maximum(@views a.σ[inside(a.σ)]))
#     tFr = inv(√sum((i)->i^2,a.g))
#     tRe = 3/14*inv(a.ν*max(1,c.λμ/c.λρ))^2
#     @inside a.σ[I] = MaxTotalflux(I,a.u)
#     tCOF = 0.5*inv(maximum(@views a.σ[inside(a.σ)]))
#     0.2min(10.,tCFL,tFr,tRe,tCOF)
# end



"""
    advect!(a::Flow{n}, c::cVOF, f=c.f, u¹=a.u⁰, u²=a.u)

This is the spirit of the operator-split cVOF calculation.
It calculates the volume fraction after one fluxing.
Volume fraction field `f` is being fluxed with the averaged of two velocity -- `u¹` and `u²`.
"""
advect!(a::Flow{D}, c::cVOF, f=c.f, u¹=a.u⁰, u²=a.u) where {D} = updateVOF!(
    a.Δt[end], f, c.fᶠ, c.n̂, c.α, u¹, u², c.c̄; perdir=a.perdir,dirdir=c.dirdir
)
function updateVOF!(
        δt, f::AbstractArray{T,D}, fᶠ::AbstractArray{T,D}, 
        n̂::AbstractArray{T,Dv}, α::AbstractArray{T,D}, 
        u::AbstractArray{T,Dv}, u⁰::AbstractArray{T,Dv}, c̄; perdir=(0,),dirdir=(0,)
    ) where {T,D,Dv}
    tol = 10eps(eltype(f))

    # Gil Strang splitting: see https://www.asc.tuwien.ac.at/~winfried/splitting/
    # if D==2
    #     opOrder = @SArray[2,1,2]
    #     opCoeff = @SArray[0.5,1.0,0.5]
    # elseif D==3
    #     opOrder = @SArray[3,2,1,2,3]
    #     opCoeff = @SArray[0.5,0.5,1.0,0.5,0.5]
    # end

    # Go to the quasi-Strang scheme:
    # I alterate the order of direction split to avoid bias.
    # TODO: this array allocation take too much time but D is not known during the compilation time so static array not possible
    opOrder = shuffle(1:D)
    opCoeff = ones(T,D)

    # calculate for dilation term
    @loop c̄[I] = f[I] <= 0.5 ? 0 : 1 over I ∈ CartesianIndices(f)
    for iOp ∈ CartesianIndices(opOrder)
        fᶠ .= 0
        d = opOrder[iOp]
        uMulp = opCoeff[iOp]*δt
        reconstructInterface!(f,α,n̂,perdir=perdir,dirdir=dirdir)
        getVOFFaceFlux!(d,f,α,n̂,u,u⁰,uMulp,fᶠ)
        @loop (
            f[I] += -∂(d,I+δ(d,I),fᶠ) + c̄[I]*(∂(d,I,u)+∂(d,I,u⁰))*0.5uMulp
        ) over I ∈ inside(f)

        # report errors if overfill or overempty
        maxf, maxid = findmax(f)
        minf, minid = findmin(f)
        if maxf-1 > tol
            du⁰,du = abs(div(maxid,u⁰)),abs(div(maxid,u))
            @printf("|∇⋅u⁰| = %+13.8f, |∇⋅u| = %+13.8f\n",du⁰,du)
            errorMsg = "max VOF @ $(maxid.I) ∉ [0,1] @ iOp=$iOp which is direction $d, Δf = $(maxf-1)"
            (du⁰+du > 10) && error(errorMsg)
            try
                error(errorMsg)
            catch e
                Base.printstyled("ERROR: "; color=:red, bold=true)
                Base.showerror(stdout, e, Base.catch_backtrace()); println()
            end
        end
        if minf < -tol
            du⁰,du = abs(div(minid,u⁰)),abs(div(minid,u))
            @printf("|∇⋅u⁰| = %+13.8f, |∇⋅u| = %+13.8f\n",du⁰,du)
            errorMsg = "min VOF @ $(minid.I) ∉ [0,1] @ iOp=$iOp which is direction $d, Δf = $(-minf)"
            (du⁰+du > 10) && error(errorMsg)
            try
                error(errorMsg)
            catch e
                Base.printstyled("ERROR: "; color=:red, bold=true)
                Base.showerror(stdout, e, Base.catch_backtrace()); println()
            end
        end

        # cleanup Wisp
        cleanWisp!(f,tol)
        BCVOF!(f,α,n̂,perdir=perdir,dirdir=dirdir)
    end
end

"""
    reconstructInterface!(f,α,n̂)

Reconstruct interface from volume fraction field, involving normal calculation and then the intercept.
Normal reconstruction follows the central difference algorithm 
proposed by [Pilliod & Puckett (2004)](https://doi.org/10.1016/j.jcp.2003.12.023) and further
modified by [Weymouth & Yue (2010)](https://doi.org/10.1016/j.jcp.2009.12.018).
"""
function reconstructInterface!(f,α,n̂;perdir=(0,),dirdir=(0,))
    N,n = size_u(n̂)
    @loop reconstructInterface!(f,α,n̂,N,I,perdir=perdir,dirdir=dirdir) over I ∈ inside(f)
    BCVOF!(f,α,n̂,perdir=perdir,dirdir=dirdir)
end
function reconstructInterface!(f::AbstractArray{T,n},α::AbstractArray{T,n},n̂::AbstractArray{T,nv},N,I;perdir=(0,),dirdir=(0,)) where {T,n,nv}
    fc = f[I]
    if (fc==0.0 || fc==1.0)
        for i∈1:n n̂[I,i] = 0 end
    else
        getInterfaceNormal_WY!(f,n̂,N,I;perdir,dirdir)
        α[I] = getIntercept(n̂,I, fc)
    end
end



"""
    getInterfaceNormal_WY!(f,n̂,N,I)

Normal reconstructure scheme from [Weymouth & Yue (2010)](https://doi.org/10.1016/j.jcp.2009.12.018). It's 3x3 compact height function with some checks.
"""
function getInterfaceNormal_WY!(f::AbstractArray{T,n},n̂,N,I;perdir=(0,),dirdir=(0,)) where {T,n}
    getInterfaceNormal_CD!(f,n̂,I)
    dc = myargmax(n,n̂,I)
    for d ∈ 1:n
        if (d == dc)
            n̂[I,d] = copysign(1.0,n̂[I,d])
        else
            hu = get3CellHeight(I+δ(d,I), f, dc)
            hc = get3CellHeight(I       , f, dc)
            hd = get3CellHeight(I-δ(d,I), f, dc)
            n̂[I,d] = -(hu-hd)*0.5
            if d ∉ dirdir && d ∉ perdir
                if I[d] == N[d]-1
                    n̂[I,d] = -(hc - hd)
                elseif I[d] == 2
                    n̂[I,d] = -(hu - hc)
                end
            # elseif (hu+hd==0.0 || hu+hd==6.0)
            #     nhat .= 0.0
            elseif abs(n̂[I,d]) > 0.5
                if (n̂[I,d]*(f[I]-0.5) >= 0.0)
                    n̂[I,d] = -(hu - hc)
                else
                    n̂[I,d] = -(hc - hd)
                end
            end
        end
    end
end


"""
    getInterfaceNormal_MYC!(f,nhat,N,I)

Mixed Youngs-Centered normal reconstructure scheme from [Aulisa et al. (2007)](https://doi.org/10.1016/j.jcp.2007.03.015), but I think best explained by 
[Duz (2005) page 81](https://doi.org/10.4233/uuid:e204277d-c334-49a2-8b2a-8a05cf603086) and [Baraldi et al. (2014)](http://doi.org/10.1016/j.compfluid.2013.12.018).
One can also be referred to the source code of [PARIS](http://www.ida.upmc.fr/~zaleski/paris/). It is in vofnonmodule.f90.
"""
function getInterfaceNormal_MYC!(f::AbstractArray{T,n},nhat,N,I) where {T,n}
    getInterfaceNormal_Y!(f,nhat,I)
    CCNhat = zeros(T,n)
    curm0 = 0
    CCiz = 0
    for iz∈1:n
        curNhat = getInterfaceNormal_CCi(f,nhat,I,iz)
        if abs(curNhat[iz])>curm0 CCNhat .= curNhat; CCiz = iz end
        curm0 = abs(curNhat[iz])
    end
    if abs(CCNhat[CCiz]) < maximum(abs,nhat)
        nhat .= CCNhat
    end
end

"""
    getInterfaceNormal_CCi!(f,nCD,I,dc)

Normal reconstructure scheme from Center column method but only in `dc` direction.
Assume we have already calculated a guessed normal to set the direction (sign) of interface in `nCD`. 
"""
function getInterfaceNormal_CCi(f::AbstractArray{T,n},nCD,I,dc) where {T,n}
    nhat = zeros(T,n)
    for d ∈ 1:n
        if (d == dc)
            nhat[d] = copysign(1.0,nCD[d])
        else
            hu = get3CellHeight(I+δ(d,I), f, dc)
            hd = get3CellHeight(I-δ(d,I), f, dc)
            nhat[d] = -(hu-hd)*0.5
        end
    end
    return nhat./sum(abs,nhat)
end

"""
    getInterfaceNormal_Y!(f, nhat, I)

Calculate the interface normal from [Youngs (1982)](https://www.researchgate.net/publication/249970655_Time-Dependent_Multi-material_Flow_with_Large_Fluid_Distortion).
Note that `nhat` is view of `n̂[I,:]`.
"""
function getInterfaceNormal_Y!(f::AbstractArray{T,D},nhat,I) where {T,D}
    for d ∈ 1:D
        nhat[d] = (YoungSum(f,I-δ(d,I),d) - YoungSum(f,I+δ(d,I),d))*0.5
    end
    nhat ./= sum(abs,nhat)
end
function YoungSum(f,I,d)
    δxy = oneunit(I)-δ(d,I)
    a = 0
    for II∈I-δxy:I
        for III∈II:II+δxy
            a+=f[III]
        end
    end
    return a
end


"""
    getInterfaceNormal_CD!(f, nhat, I)

Calculate the interface normal from the central difference scheme with the closest neighbor considered (4 neighbor in 3D).
Note that `nhat` is view of `n̂[I,:]`.
"""
function getInterfaceNormal_CD!(f::AbstractArray{T,n},n̂,I) where {T,n}
    for d ∈ 1:n
        n̂[I,d] = (crossSummation(f,I-δ(d,I),d)-crossSummation(f,I+δ(d,I),d))*0.5
    end
end
function crossSummation(f::AbstractArray{T,n},I,d) where {T,n}
    a = f[I]
    for iDir∈getAnotherDir(d,n)
        a += f[I-δ(iDir,I)]+f[I+δ(iDir,I)]
    end
    return a
end




"""
    getVOFFaceFlux!(d,f,α,n̂,u,u⁰,uMulp,fᶠ)

- `d`: the direction of cell faces that flux is calculated at
- `f`: volume fraction field
- `α`: intercept
- `n̂`: interface normal
- `u`, `u⁰`: the VOF is fluxed with the average of two velocity
- `uMulp`: the multiplier of the velocity, to take care of operator splitting coefficients and time step size
- `fᶠ`: where the flux is stored
"""
function getVOFFaceFlux!(d,f,α,n̂,u,u⁰,uMulp,fᶠ)
    fᶠ .= 0.0
    @loop fᶠ[IFace] = getVOFFaceFlux!(d,f,α,n̂,0.5*(u[IFace,d]+u⁰[IFace,d])*uMulp,IFace) over IFace ∈ inside_uWB(size(f),d)
end
function getVOFFaceFlux!(d,fIn::AbstractArray{T,n},α::AbstractArray{T,n},n̂::AbstractArray{T,nv},dl,IFace::CartesianIndex) where {T,n,nv}
    ICell = IFace;
    flux = 0.0
    if dl == 0.0
    else
        if (dl > 0.0) ICell -= δ(d,IFace) end
        f = fIn[ICell]
        sumAbsNhat = 0
        for ii∈1:n sumAbsNhat+= abs(n̂[ICell,ii]) end
        if (sumAbsNhat==0.0 || f == 0.0 || f == 1.0)
            flux = f*dl
        else
            dl = dl
            a = α[ICell]
            if (dl > 0.0) a -= n̂[ICell,d]*(1.0-dl) end
            nhatOrig = n̂[ICell,d]
            n̂[ICell,d] *= abs(dl)
            flux = getVolumeFraction(n̂,ICell, a)*dl
            n̂[ICell,d] = nhatOrig
        end
    end
    return flux
end

"""
    smoothVelocity!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody,oldp)

Smooth the velocity with top hat filter base on algorithm proposed by [Fu et al. (2010)](https://arxiv.org/abs/1410.1818), as the shear velocity is not well-resolved across fluid interface. 
We smooth the area where air dominates (α≤0.5, I suppose there is typo in eq. (1)). 
Note that we need an additional `oldp` variable to store the old pressure field in order to avoid the potential field here messing up with the old one.
"""
function smoothVelocity!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody,oldp;ω=1)
    oldp .= a.p; a.p .= 0
    a.u⁰ .= a.u;

    # smooth velocity field base on the (smoothed) VOF field
    smoothVOF!(0, c.f, c.fᶠ, c.α;perdir=c.perdir)
    smoothVelocity!(a.u,c.fᶠ,c.n̂,c.λρ)
    @. a.u = ω*a.u + (1-ω)*a.u⁰
    BC!(a.u,a.U,a.exitBC,a.perdir)

    # update the poisson solver and project the smoothed velcity field into the solenoidal velocity field
    measure!(a,d,c;t=0,ϵ=1)
    calculateL!(a,c)
    update!(b)
    project!(a,b);
    BC!(a.u,a.U,a.exitBC,a.perdir)

    a.p .= oldp

    # reporting routine
    @loop a.σ[I] = div(I,a.u) over I ∈ inside(a.p)
    diver = max(maximum(@views a.σ[inside(a.p)]),-minimum(@views a.σ[inside(a.p)]))
    @printf("Smoothed velocity: ∇⋅u = %.6e, nPois = %2d, res0 = %.6e, res = %.6e\n", diver, b.n[end], b.res0[end], b.res[end])

    # remove footprint from the general logging
    pop!(b.res); pop!(b.res0); pop!(b.n)
    pop!(a.Δt)
    push!(a.Δt,min(CFL(a,c),1.1a.Δt[end]))
end
function smoothVelocity!(u::AbstractArray{T,dv}, f::AbstractArray{T,d}, buffer::AbstractArray{T,dv},λρ) where {T,d,dv}
    buffer .= 0
    for i∈1:d
        @loop smoothVelocity!(u,f,buffer,λρ,i,I) over I ∈ inside(f)
    end
    u .= buffer
end
function smoothVelocity!(u::AbstractArray{T,dv}, f::AbstractArray{T,d}, buffer::AbstractArray{T,dv},λρ,i,I) where {T,d,dv}
    fM = ϕ(i,I,f)
    if fM<=0.5
        a = zero(eltype(u))
        # for II ∈ boxAroundI(I)
        #     rhoII = calculateρ(i,II,f,λρ)
        #     buffer[I,i] += rhoII*u[II,i]
        #     a += rhoII
        # end
        for j∈1:d
            rhoI0 = calculateρ(i,I-δ(j,I),f,λρ)
            rhoI1 = calculateρ(i,I,f,λρ)
            rhoI2 = calculateρ(i,I+δ(j,I),f,λρ)
            buffer[I,i] += rhoI0*u[I-δ(j,I),i] + 2rhoI1*u[I,i] + rhoI2*u[I+δ(j,I),i]
            a += rhoI0 + 2rhoI1 + rhoI2
        end
        buffer[I,i] /= a
    else
        buffer[I,i] = u[I,i]
    end
end


"""
    applyVOF!(f,α,n̂,FreeSurfsdf)

Given a distance function (FreeSurfsdf) for the initial free-surface, yield the volume fraction field (`f`).
Assume the gradient at the center is the normal vector. Use cell center to estimate the distance.
"""
function applyVOF!(f::AbstractArray{T,D},α::AbstractArray{T,D},n̂::AbstractArray{T,Dv},FreeSurfsdf::Function) where {T,D,Dv}
    # set up the field
    @loop applyVOF!(f,α,n̂,FreeSurfsdf,I) over I ∈ inside(f)
    # Clear Wisps in the flow
    cleanWisp!(f)
end
function applyVOF!(f::AbstractArray{T,D},α::AbstractArray{T,D},n̂::AbstractArray{T,Dv},FreeSurfsdf::Function,I::CartesianIndex{D}) where {T,D,Dv}
    α[I] = FreeSurfsdf(loc(0,I))  # the coordinate of the cell center
    n̂[I,:] = ForwardDiff.gradient(FreeSurfsdf, loc(0,I))
    sumN = 0; sumN2= 0; for i∈1:D sumN += n̂[I,i]; sumN2+= n̂[I,i]^2 end
    α[I] = 0.5sumN-√sumN2*α[I]
    f[I] = getVolumeFraction(n̂,I,α[I])
    # f[I] = clamp((√D/2 - α[I])/(√D),0,1)  # convert distance to volume fraction assume `n̂ = (1,1,...)` 
end

"""
    smoothVOF!(itm, f, sf, rf)

Smooth the cell-centered VOF field with moving average technique.
itm: smooth steps
f: befor smooth
sf: after smooth
rf: buffer
"""
function smoothVOF!(itm, f::AbstractArray{T,d}, sf::AbstractArray{T,d}, rf::AbstractArray{T,d};perdir=(0,),kelli=false) where {T,d}
    (itm!=0)&&(rf .= f)
    for it ∈ 1:itm
        sf .= 0
        @loop sumAvg!(sf,rf,I) over I ∈ inside(rf)
        BC!(sf;perdir)
        rf .= sf
    end
    kelli && interpol!(sf)
    (itm==0)&&(sf .= f)
end

function sumAvg!(sf::AbstractArray{T,d},rf::AbstractArray{T,d},I) where {T,d}
    α,β,γ = 1,1,2
    for j∈1:d
        sf[I] += α*rf[I+δ(j, I)] + β*rf[I-δ(j, I)] + γ*rf[I]
    end
    sf[I] /= (α+β+γ)*d
end

function interpol!(f)
    fin = SA[0.0,0.000128601,0.00270062,0.0239198,0.116512,0.344136,0.655864,0.883488,0.97608,0.997299,0.999871,1.0]
    fout= SA[0,0.00104741,0.0123377,0.0397531,0.131237,0.347321,0.664065,0.929867,1,1,1,1]
    nterp_linear = linear_interpolation(fin, fout)
    @loop f[I] = nterp_linear(f[I]) over I ∈ CartesianIndices(f)
end

"""
    BCVOF!(f,α,n̂)

Apply boundary condition to volume fraction, intercept, and normal with Neumann or Periodic ways
"""
function BCVOF!(f,α,n̂;perdir=(0,),dirdir=(0,))
    N,n = size_u(n̂)
    for j ∈ 1:n
        if j in perdir
            @loop f[I] = f[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
            @loop f[I] = f[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
            for i ∈ 1:n
                @loop n̂[I,i] = n̂[CIj(j,I,N[j]-1),i] over I ∈ slice(N,1,j)
                @loop n̂[I,i] = n̂[CIj(j,I,2),i] over I ∈ slice(N,N[j],j)
            end
            @loop α[I] = α[CIj(j,I,N[j]-1)] over I ∈ slice(N,1,j)
            @loop α[I] = α[CIj(j,I,2)] over I ∈ slice(N,N[j],j)
        elseif j in dirdir
        else
            @loop f[I] = f[I+δ(j,I)] over I ∈ slice(N,1,j)
            @loop f[I] = f[I-δ(j,I)] over I ∈ slice(N,N[j],j)
        end
    end
end




"""
    getIntercept(v, g)

Calculate intersection from volume fraction.
These functions prepare `n̂` and `g` for `f2α`.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
getIntercept(n̂::AbstractArray{T,3},I::CartesianIndex{2},g) where T = getIntercept(n̂[I,1],n̂[I,2],zero(T),g)
getIntercept(n̂::AbstractArray{T,4},I::CartesianIndex{3},g) where T = getIntercept(n̂[I,1],n̂[I,2],n̂[I,3],g)
getIntercept(v::AbstractArray{T,1}, g) where T = (
    length(v)==2 ?
    getIntercept(v[1], v[2], zero(T), g) :
    getIntercept(v[1], v[2], v[3], g)
)
function getIntercept(n1, n2, n3, g)
    t = abs(n1) + abs(n2) + abs(n3)
    if g != 0.5
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        a = f2α(m1, m2, m3, ifelse(g < 0.5, g, 1.0 - g))
    else
        a = 0.5
    end
    return ifelse(g < 0.5, a, 1.0 - a)*t + min(n1, 0.0) + min(n2, 0.0) + min(n3, 0.0)
end

"""
    getVolumeFraction(v, b)

Calculate intersection from volume fraction.
These functions prepare `n̂` and `b` for `α2f`.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
getVolumeFraction(n̂::AbstractArray{T,3},I::CartesianIndex{2},b) where T = getVolumeFraction(n̂[I,1],n̂[I,2],zero(T),b)
getVolumeFraction(n̂::AbstractArray{T,4},I::CartesianIndex{3},b) where T = getVolumeFraction(n̂[I,1],n̂[I,2],n̂[I,3],b)
getVolumeFraction(v::AbstractArray{T,1}, b) where T = (
    length(v)==2 ?
    getVolumeFraction(v[1], v[2], zero(T), b) :
    getVolumeFraction(v[1], v[2], v[3], b)
)
function getVolumeFraction(n1, n2, n3, b)
    t = abs(n1) + abs(n2) + abs(n3)
    a = (b - min(n1, 0.0) - min(n2, 0.0) - min(n3, 0.0))/t

    if a <= 0.0 || a == 0.5 || a >= 1.0
        return min(max(a, 0.0), 1.0)
    else
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        t = α2f(m1, m2, m3, ifelse(a < 0.5, a, 1.0 - a))
        return ifelse(a < 0.5, t, 1.0 - t)
    end
end

"""
    α2f(m1, m2, m3, a)

Three-Dimensional Forward Problem.
Get volume fraction from intersection.
This is restricted to (1) 3D, (2) n̂ᵢ ≥ 0 ∀ i, (3) ∑ᵢ n̂ᵢ = 1, (4) a < 0.5.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
function α2f(m1, m2, m3, a)
    m12 = m1 + m2

    if a < m1
        f3 = a^3/(6.0*m1*m2*m3)
    elseif a < m2
        f3 = a*(a - m1)/(2.0*m2*m3) + ifelse(m2 == 0.0, 1.0, m1 / m2) * (m1 / (6.0 * m3))  # change proposed by Kelli Hendricson to avoid the divided by zero issue
    elseif a < min(m3, m12)
        f3 = (a^2*(3.0*m12 - a) + m1^2*(m1 - 3.0*a) + m2^2*(m2 - 3.0*a))/(6*m1*m2*m3)
    elseif m3 < m12
        f3 = (a^2*(3.0 - 2.0*a) + m1^2*(m1 - 3.0*a) + m2^2*(m2 - 3.0*a) + m3^2*(m3 - 3.0*a))/(6*m1*m2*m3)
    else
        f3 = (2.0*a - m12)/(2.0*m3)
    end

    return f3
end

"""
    f2α(m1, m2, m3, v)

Three-Dimensional Inverse Problem.
Get intercept with volume fraction.
This is restricted to (1) 3D, (2) n̂ᵢ ≥ 0 ∀ i, (3) ∑ᵢ n̂ᵢ = 1, (4) v < 0.5.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
function f2α(m1, m2, m3, v)
    m12 = m1 + m2
    
    p = 6.0*m1*m2*m3
    v1 = ifelse(m2 == 0.0, 1.0, m1 / m2) * (m1 / (6.0 * m3))    # change proposed by Kelli Hendricson to avoid the divided by zero issue
    v2 = v1 + (m2 - m1)/(2.0*m3)
    v3 = ifelse(
        m3 < m12, 
        (m3^2*(3.0*m12 - m3) + m1^2*(m1 - 3.0*m3) + m2^2*(m2 - 3.0*m3))/p,
        m12*0.5/m3
    )

    if v < v1
        a3 = (p*v)^(1.0/3.0)
    elseif v < v2
        a3 = 0.5*(m1 + sqrt(m1^2 + 8.0*m2*m3*(v - v1)))
    elseif v < v3
        c0 = m1^3 + m2^3 - p*v
        c1 = -3.0*(m1^2 + m2^2)
        c2 = 3.0*m12
        c3 = -1
        a3 = proot(c0, c1, c2, c3)
    elseif m3 < m12
        c0 = m1^3 + m2^3 + m3^3 - p*v
        c1 = -3.0*(m1^2 + m2^2 + m3^2)
        c2 = 3
        c3 = -2
        a3 = proot(c0, c1, c2, c3)
    else
        a3 = m3*v + m12*0.5
    end

    return a3
end

"""
    get3CellHeight(I, f, i)

Calculate accumulate liquid height (amount) of location `I` along `i`ᵗʰ direction (I-δᵢ, I, I+δᵢ).
"""
function get3CellHeight(I, f, i)
    return f[I-δ(i,I)] + f[I] + f[I+δ(i,I)]
end

@inline calculateρ(d,I,f,λ) = (ϕ(d,I,f)*(1-λ) + λ)

@inline function calculateμ(i,j,I,f,λ,μ)
    (i==j) && return (f[I-δ(i,I)]*(1-λ) + λ)*μ
    n = length(I)
    s = zero(eltype(f))
    for II in (I-oneunit(I)):I
        s += f[II]
        # s += 1/(f[II]/1+(1-f[II])/λ)
    end
    s /= 2^n
    return (s * (1-λ) + λ)*μ
    # return s*μ
end

calculateL!(a::Flow{D}, c::cVOF) where {D} = calculateL!(a.μ₀,c.fᶠ,c.λρ,a.perdir)
function calculateL!(μ₀,f::AbstractArray{T,D},λρ,perdir) where {T,D}
    for d ∈ 1:D
        @loop μ₀[I,d] /= calculateρ(d,I,f,λρ) over I∈inside(f)
    end
    BC!(μ₀,zeros(SVector{D,T}),false,perdir)
end


# +++++++ Surface tension

function surfTen!(r,f::AbstractArray{T,D},fbuffer,α,n̂,η;perdir=(0,),dirdir=(0,)) where {T,D}
    N = size(f)
    for d∈1:D
        n̂ .= 0
        @loop fbuffer[I] = ϕ(d,I,f) over I∈inside(f)
        BC!(fbuffer;perdir)
        @loop containInterface(fbuffer[I]) && getInterfaceNormal_WY!(fbuffer,n̂,N,I) over I ∈ inside(f)
        @loop r[I,d] += containInterface(fbuffer[I]) ? η*getCurvature(I,fbuffer,majorDir(n̂,I))*-∂(d,I,f) : 0 over I∈inside(f) 
        # @loop r[I,d] += containInterface(fbuffer[I]) ? η*-1/(0.8*64)*-∂(d,I,f) : 0 over I∈inside(f) 
    end
end



"""
    getCurvature(I,f,i)

Formula from [Patel et al. (2019)](https://doi.org/10.1016/j.compfluid.2019.104263) or on [Basilisk.fr](http://basilisk.fr/src/curvature.h).
Cross derivaties from [Wikipedia](https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences).
This function has been dispatched for 2D and 3D.
"""
function getCurvature(I::CartesianIndex{3},f::AbstractArray{T,3},i) where {T}
    ix,iy = getXYdir(i)
    H = @SMatrix [
        getPopinetHeight(I+xUnit*δd(ix,I)+yUnit*δd(iy,I),f,i)
        for xUnit∈-1:1,yUnit∈-1:1
    ]
    Hx = (H[3,2] - H[1,2])/2
    Hy = (H[2,3] - H[2,1])/2
    filter = 0.2
    Hxx= (
            (H[3,2] + H[1,2] - 2*H[2,2]) + 
            (H[3,1] + H[1,1] - 2*H[2,1])*filter +
            (H[3,3] + H[1,3] - 2*H[2,3])*filter
        )/(1+2*filter)
    Hyy= (
            (H[2,3] + H[2,1] - 2*H[2,2]) + 
            (H[1,3] + H[1,1] - 2*H[1,2])*filter +
            (H[3,3] + H[3,1] - 2*H[3,2])*filter
        )/(1+2*filter)
    Hxy= (H[3,3] + H[1,1] - H[3,1] - H[1,3])/4
    return (Hxx*(1+Hy^2) + Hyy*(1+Hx^2) - 2Hxy*Hx*Hy)/(1+Hx^2+Hy^2)^1.5
end
function getCurvature(I::CartesianIndex{2},f::AbstractArray{T,2},i,returnH=false) where {T}
    ix = getXdir(i)
    H = @SArray [
        getPopinetHeight(I+xUnit*δd(ix,I),f,i)
        for xUnit∈-1:1
    ]
    Hₓ = (H[3]-H[1])/2
    Hₓₓ= (H[3]+H[1]-2H[2])
    returnH && return H
    return Hₓₓ/(1+Hₓ^2)^1.5
end


"""
    getPopinetHeight(I,f,i)

Calculate water height of a single column with methods considering the adaptive cell height and if not working, switch to the traditional 3x7 column configuration.
"""
function getPopinetHeight(I,f,i)
    H,consistent = getPopinetHeightAdaptive(I,f,i,false)
    consistent && return H
    H,consistent = getPopinetHeightAdaptive(I,f,i,true)
    consistent && return H
    H = getPopinetHeightFixed3(I,f,i)
    return H
end


"""
    getPopinetHeightAdaptive(I,f,i)

Return the column height relative to cell `I` center along signed `i` direction, which points to where there is no water.
The function is based on the Algorithm 4 from [Popinet, JCP (2009)](https://doi.org/10.1016/j.jcp.2009.04.042).
If `monotonic` is activated, the summation will only cover the monotonic range. The monotonic condition is based on [Guo et al., Appl. Math. Model. (2015)](https://doi.org/10.1016/j.apm.2015.04.022).
"""
function getPopinetHeightAdaptive(I,f,i,monotonic=true)
    consistent = true
    Inow = I; fnow = f[Inow]; H = (fnow-0.5)
    # Iterate till reach the cell full of air
    finishInd = fnow<1
    while !finishInd || containInterface(fnow)
        Inow += δd(i,I); !validCI(Inow,f) && break
        fnow = ifelse(monotonic && f[Inow]>fnow, 0, f[Inow])
        H += fnow
        finishInd = ifelse(containInterface(fnow),true,finishInd)
    end
    consistent = (fnow==0) && consistent
    Inow = I; fnow = f[Inow]
    # Iterate till reach the cell full of water
    finishInd = fnow>0
    while !finishInd || containInterface(fnow)
        Inow -= δd(i,I); !validCI(Inow,f) && break
        fnow = ifelse(monotonic && f[Inow]<fnow, 1, f[Inow])
        H += fnow-1  # a little trick that make `I` cell center the origin
        finishInd = ifelse(containInterface(fnow),true,finishInd)
    end
    consistent = (fnow==1) && consistent
    return H,consistent
end


"""
    getPopinetHeightFixed3(I,f,i)

Traditional 3x7 height function proposed by [Cummins et al. (2005)](https://doi.org/10.1016/j.compstruc.2004.08.017).
"""
function getPopinetHeightFixed3(I,f,i,hh=3)
    consistent = true
    Inow = I; fnow = f[Inow]; H = (fnow-0.5)
    # Iterate till reach the cell full of air
    finishInd = fnow<1
    for ii = 1:hh
        Inow += δd(i,I); !validCI(Inow,f) && break
        fnow = f[Inow]
        H += fnow
    end
    consistent = (fnow==0) && consistent
    Inow = I; fnow = f[Inow]
    # Iterate till reach the cell full of water
    finishInd = fnow>0
    for ii = 1:hh
        Inow -= δd(i,I); !validCI(Inow,f) && break
        fnow = f[Inow]
        H += fnow-1  # a little trick that make `I` cell center the origin
    end
    consistent = (fnow==1) && consistent
    return H
end

getHeightFD(I,f,i) = f[I]
getHeightFixed1(I,f,i) = get3CellHeight(I,f,i)


"""
    getCurvature2D_Fit(I,f,α,n̂)

Get curvature from parabola-fitting method with the constraint of interface point and interface normal.
"""
function getCurvature2D_Fit(I,f,α,n̂)
    iy = majorDir(n̂,I)
    ix = getXdir(iy)
    point = []
    norma = []
    for II∈boxAroundI(I)
        if containInterface(f[II]) && sum(abs,n̂[II,:])>0.1
            coord = getInterfaceCenter(n̂,α,II).+II.I.-I.I
            push!(point,transformCoord(coord,[ix,iy]))
            push!(norma,transformCoord(n̂[II,:],[ix,iy]))
        end
    end
    cenCoord = transformCoord(getInterfaceCenter(n̂,α,I),[ix,iy])
    a = getPara2D(point,norma)
    return 2a[1]/(1+(2a[1]*cenCoord[1]+a[2])^2)^1.5
end

"""
    getPara2D(point,norma)

Return coefficients of 2d parabola from given points and normal.
Note that the parabola cannot be an inclined one it should either be y = f(x) or x = g(y). No xy term present.
"""
function getPara2D(point,norma)
    nPoint = length(point)
    A = zeros(2nPoint,3)
    b = zeros(2nPoint)
    for i∈1:nPoint
        x,y = point[i]
        nx,ny = norma[i]
        A[2i-1:2i,:] = [x^2 x 1; 2ny*x ny 0]
        b[2i-1:2i] = [y;-nx]
    end
    return (A'*A)\(A'*b)
end


"""
    getInterfaceCenter(n̂,α,I)

To calculate the quasi-center of line or plane segments in cell `I` by projecting the cell center to the plane.
"""
function getInterfaceCenter(n̂::AbstractArray{T,nv},α::AbstractArray{T,n},I::CartesianIndex{n}) where{T,n,nv}
    nLocal = @views n̂[I,:]
    dis = (0.5sum(nLocal) - α[I])/√sum(abs2,nLocal)
    return -dis*nLocal/√sum(abs2,nLocal)
end



# +++++++ Auxiliary functions for multiphase calculation

"""
    proot(c0, c1, c2, c3)

Calculate the roots of a third order polynomial, which has three real roots:
    c3 x³ + c2 x² + c1 x¹ + c0 = 0
"""
function proot(c0, c1, c2, c3)
    a0 = c0/c3
    a1 = c1/c3
    a2 = c2/c3

    p0 = a1/3.0 - a2^2/9.0
    q0 = (a1*a2 - 3.0*a0)/6.0 - a2^3/27.0
    a = q0/sqrt(-p0^3)
    t = acos(abs2(a)<=1 ? a : 0)/3.0

    proot = sqrt(-p0)*(sqrt(3.0)*sin(t) - cos(t)) - a2/3.0

    return proot
end


"""
    sort3(a, b, c)

Sort three numbers with bubble sort algorithm to avoid too much memory assignment due to array creation.
see https://codereview.stackexchange.com/a/91920
"""
function sort3(a, b, c)
    if (a>c) a,c = c,a end
    if (a>b) a,b = b,a end
    if (b>c) b,c = c,b end
    return a,b,c
end


"""
    sortArg(a,b(,c))

Sort argument according to its absolute value. Support 2 or 3D
"""
function sortArg(a,b,c)
    α,β,γ = copysign(1,a),copysign(2,b),copysign(3,c)
    if (abs(a)<abs(c)) α,γ = γ,α; a,c = c,a end
    if (abs(a)<abs(b)) α,β = β,α; a,b = b,a end
    if (abs(b)<abs(c)) β,γ = γ,β; b,c = c,b end
    return α,β,γ
end
function sortArg(a,b)
    α,β = copysign(1,a),copysign(2,b)
    if (abs(a)<abs(b)) α,β = β,α; a,b = b,a end
    return α,β
end


"""
    myargmax(n,vec)

Return where is the maximum since the original `argmax` function in julia is not working in GPU environment.
`n` is the length of `vec`
"""
function myargmax(n,vec)
    max = abs2(vec[1])
    iMax = 1
    for i∈2:n
        cur = abs2(vec[i])
        if cur > max
            max = cur
            iMax = i
        end
    end
    return iMax
end
function myargmax(n,vec,I)
    max = abs2(vec[I,1])
    iMax = 1
    for i∈2:n
        cur = abs2(vec[I,i])
        if cur > max
            max = cur
            iMax = i
        end
    end
    return iMax
end


"""
    boxAroundI(I::CartesianIndex{D})

Return 3 surrunding cells in each direction of I, including diagonal ones.
The return grid number adds up to 3ᴰ 
"""
boxAroundI(I::CartesianIndex{D}) where D = (I-oneunit(I)):(I+oneunit(I))

function cleanWisp!(f::AbstractArray{T,D},tol=10eps(T)) where {T,D}
    @loop f[I] = f[I] <= tol ? 0.0 : f[I] over I ∈ inside(f)
    @loop f[I] = f[I] >= 1-tol ? 1.0 : f[I] over I ∈ inside(f)
end

function checkNaNInf(f)
    for I∈CartesianIndices(f)
        if isnan(f[I])
            error("There is NaN in $(I.I)!")
        elseif isinf(f[I])
            error("There is Inf in $(I.I)!")
        end
    end
end

function calke!(σ,u,f,f1,λρ,ke,keN)
    σ.=0
    @inside σ[I] = ρkeI(I,u,f ,λρ)
    push!(ke,Statistics.mean(@views σ[inside(σ)]))
    @inside σ[I] = ρkeI(I,u,f1,λρ)
    push!(keN,Statistics.mean(@views σ[inside(σ)]))
end

"""
    δd(i,I)

It is still coordinate shifting stuff but with direction (+/-) support for `i`.
"""
δd(i,I) = sign(i)*δ(abs(i),I)

containInterface(f) = (f≠0) && (f≠1)
validCI(I::CartesianIndex{D},f::AbstractArray{T,D}) where {T,D} = all(ntuple(i->1,D) .≤ I.I .≤ size(f))
getXdir(i) = ifelse(abs(i)==1,-2sign(i),sign(i))
getXYdir(i) = (sign(i)*((abs(i))%3+1),(abs(i)+1)%3+1)
getAnotherDir(d,n) = filter(i-> i≠d,(1:n...,))
transformCoord(coord,perm) = sign.(perm).*coord[abs.(perm)]
function majorDir(n̂,I::CartesianIndex{D}) where D
    i = myargmax(D,n̂,I)
    return copysign(i,n̂[I,i])
end