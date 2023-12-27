using ForwardDiff
using Printf
using JLD2
using Combinatorics


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
    n̂ :: Vf  # normal vector of the surfaces in cell
    α :: Sf  # intercept of intercace in cell: norm ⋅ x = α
    c̄ :: AbstractArray{Int8}  # color function at the cell center
    perdir :: NTuple  # periodic directions
    dirdir :: NTuple  # Dirichlet directions
    λρ :: T   # ratio of density (air/water)
    λμ :: T   # ratio of dynamic viscosity (air/water)
    ke ::Vector{T}
    strang :: Vector{Int}
    function cVOF(N::NTuple{D}, n̂place, αplace; arr=Array, InterfaceSDF::Function=(x) -> 5-x[1], T=Float64, perdir=(0,), dirdir=(0,),λμ=1e-2,λρ=1e-3) where D
        Ng = N .+ 2  # scalar field size
        Nd = (Ng..., D)  # vector field size
        f = ones(T, Ng) |> arr
        fᶠ = copy(f)
        n̂ = n̂place; α = αplace
        c̄ = zeros(Int8, Ng) |> arr
        applyVOF!(f,α,InterfaceSDF)
        BCVOF!(f,α,n̂;perdir=perdir,dirdir=dirdir)
        f⁰ = copy(f)
        smoothVOF!(0, f, fᶠ, α;perdir=perdir)
        new{D,T,typeof(f),typeof(n̂)}(f,f⁰,fᶠ,n̂,α,c̄,perdir,dirdir,λρ,λμ,[],[0])
    end
end

"""
    applyVOF!(f,α,FreeSurfsdf)

Given a distance function (FreeSurfsdf) for the initial free-surface, yield the volume fraction field (`f`)
"""
function applyVOF!(f::AbstractArray{T,D},α::AbstractArray{T,D},FreeSurfsdf::Function) where {T,D}
    tol = 10eps(T)  # tolerance for the Floating points

    # set up the field
    @loop applyVOF!(f,α,FreeSurfsdf,I) over I ∈ inside(f)

    # Clear wasps in the flow
    @loop f[I] = f[I] <= tol ? 0.0 : f[I] over I ∈ inside(f)
    @loop f[I] = f[I] >= 1-tol ? 1.0 : f[I] over I ∈ inside(f)
end
function applyVOF!(f::AbstractArray{T,D},α::AbstractArray{T,D},FreeSurfsdf::Function,I::CartesianIndex{D}) where {T,D}
    α[I] = FreeSurfsdf(loc(0,I))  # the coordinate of the cell center
    f[I] = clamp((√D/2 - α[I])/(√D),0,1)  # convert distance to volume fraction assume `n̂ = (1,1,...)` 
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
    smoothVOF!(itm, f, sf)

Smooth the cell-centered VOF field with moving average technique.
itm: smooth steps
f: befor smooth
sf: after smooth
"""
function smoothVOF!(itm, f::AbstractArray{T,d}, sf::AbstractArray{T,d}, rf::AbstractArray{T,d};perdir=(0,)) where {T,d}
    (itm!=0)&&(rf .= f)
    for it ∈ 1:itm
        sf .= 0
        for j ∈ 1:d
            @loop sf[I] += rf[I+δ(j, I)] + rf[I-δ(j, I)] + 2*rf[I] over I ∈ inside(rf)
        end
        BCPerNeu!(sf,perdir=perdir)
        sf ./= 4*d
        rf .= sf
    end
    (itm==0)&&(sf .= f)
end


"""
    smoothVelocity!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody,oldp)

Smooth the velocity with top hat filter base on algorithm proposed by [Fu et al. (2010)](https://arxiv.org/abs/1410.1818), as the shear velocity is not well-resolved across fluid interface. 
We smooth the area where air dominates (α≤0.5, I suppose there is typo in eq. (1)). 
Note that we need an additional `oldp` variable to store the old pressure field in order to avoid the potential field here messing up with the old one.
"""
function smoothVelocity!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody,oldp)
    oldp .= a.p; a.p .= 0

    # smooth velocity field base on the (smoothed) VOF field
    smoothVOF!(0, c.f, c.fᶠ, c.α;perdir=c.perdir)
    smoothVelocity!(a.u,c.fᶠ,c.n̂,c.λρ)
    BCVecPerNeu!(a.u;Dirichlet=true, A=a.U, perdir=a.perdir)

    # update the poisson solver and project the smoothed velcity field into the solenoidal velocity field
    measure!(a,d;t=0,ϵ=1,perdir=a.perdir)
    calculateL!(a,c)
    update!(b)
    project!(a,b);
    BCVecPerNeu!(a.u;Dirichlet=true, A=a.U, perdir=a.perdir)

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
        for II ∈ boxAroundI(I)
            rhoII = calculateρ(i,II,f,λρ)
            buffer[I,i] += rhoII*u[II,i]
            a += rhoII
        end
        buffer[I,i] /= a
    else
        buffer[I,i] = u[I,i]
    end
end


"""
    getIntercept(v, g)
Calculate intersection from volume fraction.
These functions prepare `n̂` and `g` for `f2α`.
Following algorithm proposed by [Scardovelli & Zaleski (2000)](https://doi.org/10.1006/jcph.2000.6567).
"""
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
    nhat = @views n̂[I,:] #nzeros(T,n)
    if (fc==0.0 || fc==1.0)
        f[I] = fc
        for i∈1:n n̂[I,i] = 0 end
    else
        for d ∈ 1:n
            nhat[d] = (f[I-δ(d,I)]-f[I+δ(d,I)])*0.5
        end
        dc = myargmax(n,nhat)
        for d ∈ 1:n
            if (d == dc)
                nhat[d] = copysign(1.0,nhat[d])
            else
                hu = get3CellHeight(I+δ(d,I), f, dc)
                hc = get3CellHeight(I       , f, dc)
                hd = get3CellHeight(I-δ(d,I), f, dc)
                nhat[d] = -(hu-hd)*0.5
                if d ∉ dirdir && d ∉ perdir
                    if I[d] == N[d]-1
                        nhat[d] = -(hc - hd)
                    elseif I[d] == 2
                        nhat[d] = -(hu - hc)
                    end
                elseif (hu+hd==0.0 || hu+hd==6.0)
                    nhat .= 0.0
                elseif abs(nhat[d]) > 0.5
                    if (nhat[d]*(fc-0.5) >= 0.0)
                        nhat[d] = -(hu - hc)
                    else
                        nhat[d] = -(hc - hd)
                    end
                end
            end
        end
        α[I] = getIntercept(nhat, fc)
        for i∈1:n n̂[I,i] = nhat[i] end
    end
end


"""
    get3CellHeight(I, f, i)

Calculate accumulate liquid height (amount) of location `I` along `i`ᵗʰ direction (I-δᵢ, I, I+δᵢ).
"""
function get3CellHeight(I, f, i)
    return f[I-δ(i,I)] + f[I] + f[I+δ(i,I)]
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
    N,n = size_u(n̂)
    fᶠ .= 0.0
    @loop getVOFFaceFlux!(fᶠ,d,f,α,n̂,0.5*(u[IFace,d]+u⁰[IFace,d])*uMulp,IFace) over IFace ∈ inside_uWB(N,d)
end
function getVOFFaceFlux!(fᶠ::AbstractArray{T,n},d,fIn::AbstractArray{T,n},α::AbstractArray{T,n},n̂::AbstractArray{T,nv},dl,IFace::CartesianIndex) where {T,n,nv}
    ICell = IFace;
    flux = 0.0
    if dl == 0.0
    else
        if (dl > 0.0) ICell -= δ(d,IFace) end
        f = fIn[ICell]
        nhat = @views n̂[ICell,:]
        if (sum(abs,nhat)==0.0 || f == 0.0 || f == 1.0)
            flux = f*dl
        else
            dl = dl
            a = α[ICell]
            if (dl > 0.0) a -= nhat[d]*(1.0-dl) end
            nhatOrig = nhat[d]
            nhat[d] *= abs(dl)
            flux = getVolumeFraction(nhat, a)*dl
            nhat[d] = nhatOrig
        end
    end
    fᶠ[IFace] = flux
end


"""
    advect!(a::Flow{n}, c::cVOF, f=c.f, u¹=a.u⁰, u²=a.u)

This is the spirit of the operator-split cVOF calculation.
It calculates the volume fraction after one fluxing.
Volume fraction field `f` is being fluxed with the averaged of two velocity -- `u¹` and `u²`.
"""
advect!(a::Flow{D}, c::cVOF, f=c.f, u¹=a.u⁰, u²=a.u) where {D} = updateVOF!(
    a.Δt[end], f, c.fᶠ, c.n̂, c.α, u¹, u², c.c̄, c.strang;perdir=a.perdir,dirdir=c.dirdir
)
function updateVOF!(
        δt, f::AbstractArray{T,D}, fᶠ::AbstractArray{T,D}, 
        n̂::AbstractArray{T,Dv}, α::AbstractArray{T,D}, 
        u::AbstractArray{T,Dv}, u⁰::AbstractArray{T,Dv}, c̄, strang;perdir=(0,),dirdir=(0,)
    ) where {T,D,Dv}
    tol = 10eps(eltype(f))

    # Gil Strang splitting: see https://www.asc.tuwien.ac.at/~winfried/splitting/
    if D==2
        opOrder = [2,1,2]
        opCoeff = [0.5,1.0,0.5]
    elseif D==3
        opOrder = [3,2,1,2,3]
        opCoeff = [0.5,0.5,1.0,0.5,0.5]
    end

    # opOrder = [(i+strang[1])%D+1 for i∈1:D] #nthperm(collect(1:D),rand(1:factorial(D)))
    # opCoeff = [1.0 for i∈1:D]
    # strang[1] +=1

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
            println("∇⋅u⁰: ", div(maxid,u⁰))
            println("∇⋅u : ", div(maxid,u))
            errorMsg = "max VOF @ $(maxid.I) ∉ [0,1] @ iOp=$iOp which is $d, Δf = $(maxf-1)"
            try
                error(errorMsg)
            catch e
                println("∇⋅u + ∇⋅u⁰: ", div(maxid,u)+div(maxid,u⁰))
                Base.printstyled("ERROR: "; color=:red, bold=true)
                Base.showerror(stdout, e, Base.catch_backtrace())
                println()
            end
        end
        if minf < -tol
            println("∇⋅u : ", div(maxid,u⁰))
            println("∇⋅u⁰: ", div(maxid,u))
            errorMsg = "min VOF @ $(minid.I) ∉ [0,1] @ iOp=$iOp which is $d, Δf = $(-minf)"
            try
                error(errorMsg)
            catch e
                println("∇⋅u + ∇⋅u⁰: ", div(minid,u)+div(minid,u⁰))
                Base.printstyled("ERROR: "; color=:red, bold=true)
                Base.showerror(stdout, e, Base.catch_backtrace())
                println()
            end
        end

        # cleanup wasp
        @loop f[I] = f[I] <= tol ? 0.0 : f[I] over I ∈ inside(f)
        @loop f[I] = f[I] >= 1-tol ? 1.0 : f[I] over I ∈ inside(f)
        BCVOF!(f,α,n̂,perdir=perdir,dirdir=dirdir)
    end
end

function measure!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody,t=0)
    measure!(a,d;t=0,ϵ=1,perdir=a.perdir)
    calculateL!(a,c)
    update!(b)
end


function conv_diff2p!(r,u,Φ,fᶠ,λμ,λρ,ν;perdir=(0,),g=(0,0,0))
# function conv_diff!(r,u,Φ,ν,ρ,perdir,g)
    r .= 0.
    N,n = size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        !tagper && lowBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N)
        tagper && lowBoundaryPerDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N)
        # inner cells
        innerCellDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N)
        # treatment for upper boundary with BCs
        !tagper && upperBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N)
        tagper && upperBoundaryPer!(r,u,Φ,ν,i,j,N)
    end
    for i ∈ 1:n
        @loop r[I,i] = r[I,i]/calculateρ(i,I,fᶠ,λρ) + g[i] over I ∈ inside_uWB(N,i)
    end
    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        !tagper && lowBoundaryConv!(r,u,Φ,ν,i,j,N)
        tagper && lowBoundaryPerConv!(r,u,Φ,ν,i,j,N)
        # inner cells
        innerCellConv!(r,u,Φ,ν,i,j,N)
        # treatment for upper boundary with BCs
        !tagper && upperBoundaryConv!(r,u,Φ,ν,i,j,N)
        tagper && upperBoundaryPer!(r,u,Φ,ν,i,j,N)
    end
end

innerCellConv!(r,u,Φ,ν,i,j,N) = (
    @loop (
        Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u));
        r[I,i] += Φ[I]
    ) over I ∈ inside_u(N,j);
    @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
)
innerCellDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N) = (
    @loop (
        Φ[I] = -calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u));
        r[I,i] += Φ[I]
    ) over I ∈ inside_u(N,j);
    @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)
)

# Neumann BC Building block
lowBoundaryConv!(r,u,Φ,ν,i,j,N) = @loop r[I,i] += ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u) over I ∈ slice(N,2,j,2)
lowBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N) = @loop r[I,i] += -calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u)) over I ∈ slice(N,2,j,2)

upperBoundaryConv!(r,u,Φ,ν,i,j,N) = @loop r[I-δ(j,I),i] += - ϕ(j,CI(I,i),u)*ϕ(i,CI(I,j),u) over I ∈ slice(N,N[j],j,2)
upperBoundaryDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N) = @loop r[I-δ(j,I),i] += calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u)) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowBoundaryPerConv!(r,u,Φ,ν,i,j,N) = @loop (
    Φ[I] = ϕuSelf(CIj(j,CI(I,i),N[j]-2),CI(I,i)-δ(j,CI(I,i)),CI(I,i),CI(I,i)+δ(j,CI(I,i)),u,ϕ(i,CI(I,j),u));
    r[I,i] += Φ[I]
) over I ∈ slice(N,2,j,2)
lowBoundaryPerDiff!(r,u,Φ,fᶠ,λμ,ν,i,j,N) = @loop (
    Φ[I] = -calculateμ(i,j,I,fᶠ,λμ,ν)*(∂(j,CI(I,i),u)+∂(i,CI(I,j),u));
    r[I,i] += Φ[I]
) over I ∈ slice(N,2,j,2)
upperBoundaryPer!(r,u,Φ,ν,i,j,N) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)


"""
    mom_step!(a::Flow,b::AbstractPoisson,c::cVOF,sim::TwoPhaseSimulation)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
"""
@fastmath function mom_step!(a::Flow,b::AbstractPoisson,c::cVOF,d::AbstractBody)
    a.u⁰ .= a.u;
    smoothStep = 0

    # predictor u → u'
    advect!(a,c,c.f,a.u⁰,a.u)
    measure!(a,d;t=0,ϵ=1,perdir=a.perdir)
    smoothVOF!(smoothStep, c.f⁰, c.fᶠ, c.α;perdir=c.perdir)
    calke && (a.σ .= 0; @inside a.σ[I] = ρkeI(I,a.u,c.fᶠ,c.λρ); push!(c.ke,sum(a.σ)))
    a.u .= 0
    a.σ .= 0
    conv_diff2p!(a.f,a.u⁰,a.σ,c.fᶠ,c.λμ,c.λρ,a.ν, perdir=a.perdir,g=a.g)
    BDIM!(a)
    BCVecPerNeu!(a.u;Dirichlet=true, A=a.U, perdir=a.perdir)
    calculateL!(a,c)
    update!(b)
    calke && (a.σ .= 0; @inside a.σ[I] = ρkeI(I,a.u,c.fᶠ,c.λρ); push!(c.ke,sum(a.σ)))
    project!(a,b); 
    BCVecPerNeu!(a.u;Dirichlet=true, A=a.U, perdir=a.perdir)
    calke && (a.σ .= 0; @inside a.σ[I] = ρkeI(I,a.u,c.fᶠ,c.λρ); push!(c.ke,sum(a.σ)))

    # corrector u → u¹
    advect!(a,c,c.f⁰,a.u⁰,a.u)
    measure!(a,d;t=0,ϵ=1,perdir=a.perdir)
    smoothVOF!(smoothStep, c.f, c.fᶠ, c.α;perdir=c.perdir)
    calke && (a.σ .= 0; @inside a.σ[I] = ρkeI(I,a.u,c.fᶠ,c.λρ); push!(c.ke,sum(a.σ)))
    a.σ .= 0
    conv_diff2p!(a.f,a.u,a.σ,c.fᶠ,c.λμ,c.λρ,a.ν, perdir=a.perdir,g=a.g)
    BDIM!(a); a.u ./= 2;
    BCVecPerNeu!(a.u;Dirichlet=true, A=a.U, f=2, perdir=a.perdir)
    calculateL!(a,c)
    update!(b)
    calke && (a.σ .= 0; @inside a.σ[I] = ρkeI(I,a.u,c.fᶠ,c.λρ); push!(c.ke,sum(a.σ)))
    project!(a,b,2);  
    BCVecPerNeu!(a.u;Dirichlet=true, A=a.U, perdir=a.perdir)
    calke && (a.σ .= 0; @inside a.σ[I] = ρkeI(I,a.u,c.fᶠ,c.λρ); push!(c.ke,sum(a.σ)))
    c.f .= c.f⁰

    calke && (a.σ .= 0; @inside a.σ[I] = ρkeI(I,a.u,c.f,c.λρ); push!(c.ke,sum(a.σ)))
    push!(a.Δt,min(CFL(a,c),1.1a.Δt[end]))
end

@fastmath @inline function MaxTotalflux(I::CartesianIndex{d},u) where {d}
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(abs(u[I,i]),abs(u[I+δ(i,I),i])))
    end
    return s
end

function CFL(a::Flow,c::cVOF)
    @inside a.σ[I] = flux_out(I,a.u)
    fluxLimit = inv(maximum(@views a.σ[inside(a.σ)])+5*a.ν*max(1,c.λμ/c.λρ))
    @inside a.σ[I] = MaxTotalflux(I,a.u)
    cVOFLimit = 0.5*inv(maximum(@views a.σ[inside(a.σ)]))
    0.8min(10.,fluxLimit,cVOFLimit)
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

calculateL!(a::Flow{n}, c::cVOF) where {n} = calculateL!(a.μ₀,c.fᶠ,c.λρ,n,a.U,a.perdir)
function calculateL!(μ₀,f,λρ,n,U,perdir)
    for d ∈ 1:n
        @loop μ₀[I,d] /= calculateρ(d,I,f,λρ) over I∈inside(f)
    end
    BCVecPerNeu!(μ₀;Dirichlet=true, A=U, perdir=perdir)
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


"""
    boxAroundI(I::CartesianIndex{D})

Return 3 surrunding cells in each diraction of I, including diagonal ones.
The return grid number adds up to 3ᴰ 
"""
boxAroundI(I::CartesianIndex{D}) where D = (I-oneunit(I)):(I+oneunit(I))