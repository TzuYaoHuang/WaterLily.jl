using ForwardDiff
using LinearAlgebra

struct cVOF{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}}
    f :: Sf  # cell-averaged color function
    fᶠ:: Sf  # place to store flux or smoothed vof
    n̂ :: Vf  # norm of the surfaces in cell
    α :: Sf  # intercept of intercace in cell: norm ⋅ x = α
    c̄ :: AbstractArray{Int8}  # color function
    perdir :: NTuple  # periodic directions
    dirdir :: NTuple  # Dirichlet directions
    function cVOF(N::NTuple{D}, n̂place, αplace; arr=Array, InterfaceSDF::Function=(x) -> 5-x[1], T=Float64, perdir=(0,), dirdir=(0,)) where D
        Ng = N .+ 2
        f = ones(T, Ng) |> arr
        fᶠ = copy(f)
        n̂ = n̂place; α = αplace
        c̄ = zeros(Int8, Ng) |> arr
        applyVOF!(InterfaceSDF,f,α,n̂)
        BCVOF!(f,α,n̂;perdir=perdir,dirdir=dirdir)
        new{D,T,typeof(f),typeof(n̂)}(f,fᶠ,n̂,α,c̄,perdir,dirdir)
    end
end

"""
    applyVOF!(FreeSurfsdf,f,α,n̂)

Given a distance function (FreeSurfsdf) for the initial free-surface, yield the volume fraction (f), intercept (α), normal (n̂)
"""
function applyVOF!(FreeSurfsdf,f,α,n̂)
    N,n = size_u(n̂)
    @loop (
        α[I] = FreeSurfsdf(loc(I));
        if abs2(α[I])>n 
            f[I] = ifelse(α[I]>0,0,1)
        else
            n̂[I,:] = ForwardDiff.gradient(FreeSurfsdf,(loc(0,I)+loc(I))*0.5);
            n̂[I,:] /= (n̂[I,1]^2+n̂[I,2]^2+n̂[I,3]^2)^0.5;
            f[I] = vof_vol(n̂[I,1],n̂[I,2],n̂[I,3],-α[I]);
        end
    ) over I ∈ inside(f)
end



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
            for i ∈ 1:n
                mulp = ifelse(i==j, -1, 1)
                @loop n̂[I,i] = mulp*n̂[I+δ(j,I),i] over I ∈ slice(N,1,j)
                @loop n̂[I,i] = mulp*n̂[I-δ(j,I),i] over I ∈ slice(N,N[j],j)
            end
            @loop α[I] = vof_int(n̂[I,1],n̂[I,2],n̂[I,3],f[I]) over I ∈ slice(N,1,j)
            @loop α[I] = vof_int(n̂[I,1],n̂[I,2],n̂[I,3],f[I]) over I ∈ slice(N,N[j],j)
        end
    end
end



"""
    vof_smooth!(itm, f, sf)

Smooth the cell-centered VOF field with moving average technique.
itm: smooth steps
f: befor smooth
sf: after smooth
"""
function vof_smooth!(itm, f::AbstractArray{T,d}, sf::AbstractArray{T,d}) where {T,d}
    N = size(f)
    rf = f
    for it ∈ 1:itm
        sf .= 0.0
        for j ∈ 1:d
            @loop sf[I] += rf[I+δ(j, I)] + rf[I-δ(j, I)] + 2*rf[I] over I ∈ inside(rf)
        end
        !per && BC!(sf)
        per && BCVecPerNeu!(sf)
        sf /= 12
        rf = sf
    end
end

"""
    f3(m1, m2, m3, a)

Three-Dimensional Forward Problem.
Get volume fraction from intersection.
"""
function f3(m1, m2, m3, a)
    m12 = m1 + m2
    tol = 1e-10

    if a < m1
        f3 = a^3/(6.0*m1*m2*m3)
    elseif a < m2
        f3 = a*(a - m1)/(2.0*m2*m3) + m1^2/(6.0*m2*m3 + tol)
    elseif a < min(m3, m12)
        f3 = (a^2*(3.0*m12 - a) + m1^2*(m1 - 3.0*a) + m2^2*(m2 - 3.0*a))/(6*m1*m2*m3)
    elseif m3 < m12
        f3 = (a^2*(3.0 - 2.0*a) + m1^2*(m1 - 3.0*a) + m2^2*(m2 - 3.0*a) + m3^2*(m3 - 3.0*a))/(6*m1*m2*m3)
    else
        f3 = (2.0*a - m12)/(2.0*m3)
    end

    return f3
end

function proot(c0, c1, c2, c3)
    a0 = c0/c3
    a1 = c1/c3
    a2 = c2/c3

    p0 = a1/3.0 - a2^2/9.0
    q0 = (a1*a2 - 3.0*a0)/6.0 - a2^3/27.0
    t = acos(q0/sqrt(-p0^3))/3.0

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
    a3(m1, m2, m3, v)

Three-Dimensional Inverse Problem.
Get intercept with volume fraction.
"""
function a3(m1, m2, m3, v)
    m12 = m1 + m2
    tol = 1e-10

    p = 6.0*m1*m2*m3
    v1 = m1^2/(6.0*m2*m3 + tol)
    v2 = v1 + (m2 - m1)/(2.0*m3)
    v3 = ifelse(m3 < m12, (m3^2*(3.0*m12 - m3) + m1^2*(m1 - 3.0*m3) + m2^2*(m2 - 3.0*m3))/p,m12*0.5/m3)

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

function vof_int(n1, n2, n3, g)
    t = abs(n1) + abs(n2) + abs(n3)
    if g != 0.5
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        a = a3(m1, m2, m3, ifelse(g < 0.5, g, 1.0 - g))
    else
        a = 0.5
    end

    vof_int = ifelse(g < 0.5, a, 1.0 - a)*t + min(n1, 0.0) + min(n2, 0.0) + min(n3, 0.0)
end

function vof_vol(n1, n2, n3, b)
    t = abs(n1) + abs(n2) + abs(n3)
    a = (b - min(n1, 0.0) - min(n2, 0.0) - min(n3, 0.0))/t

    if a <= 0.0 || a == 0.5 || a >= 1.0
        vof_vol = min(max(a, 0.0), 1.0)
    else
        m1, m2, m3 = sort3(abs(n1)/t, abs(n2)/t, abs(n3)/t)
        t = f3(m1, m2, m3, ifelse(a < 0.5, a, 1.0 - a))
        vof_vol = ifelse(a < 0.5, t, 1.0 - t)
    end
end

function vof_height(I, f, i)
    h = 0.0
    I -= δ(i,I)
    for j ∈ 1:3
        h += f[I]
        I += δ(i,I)
    end
    return h
end

function vof_reconstruct!(f,α,n̂;perdir=(0,),dirdir=(0,))
    N,n = size_u(n̂)
    @loop α[I],n̂[I,:] = vof_reconstruct!(I,f,α,n̂,N,n) over I ∈ inside(f)
    BCVOF!(f,α,n̂,perdir=perdir,dirdir=dirdir)
end

function vof_reconstruct!(I,f,α,n̂,N,n)
    fc = f[I]
    nhat = n̂[I,:]
    alpha = α[I]
    if (fc==0.0 || fc==1.0)
        alpha = fc
    else
        for d ∈ 1:n
            nhat[d] = (f[I-δ(d,I)]-f[I+δ(d,I)])*0.5
        end
        dc = argmax(abs.(nhat))
        for d ∈ 1:n
            if (d == dc)
                nhat[d] = copysign(1.0,nhat[d])
            else
                hu = vof_height(I+δ(d,I), f, dc)
                hc = vof_height(I       , f, dc)
                hd = vof_height(I-δ(d,I), f, dc)
                nhat[d] = -(hu-hd)*0.5
                if I[d] == N[d]-1
                    nhat[d] = -(hc - hd)
                elseif I[d] == 2
                    nhat[d] = -(hu - hc)
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
        alpha = vof_int(nhat[1],nhat[2],nhat[3], fc)
    end
    return alpha,nhat
end

function vof_flux!(d,f,α,n̂,u,u⁰,uMulp,fᶠ)
    N,n = size_u(n̂)
    fᶠ .= 0.0
    @loop fᶠ[IFace] = vof_flux(IFace, d,f,α,n̂,0.5*(u[IFace,d]+u⁰[IFace,d])*uMulp) over IFace ∈ inside_uWB(N,d)
end

function vof_flux(IFace::CartesianIndex, d,fIn,α,n̂,dl)
    v = dl;
    ICell = IFace;
    flux = 0.0
    if v == 0.0
    else
        if (v > 0.0) ICell -= δ(d,IFace) end
        f = fIn[ICell]
        if ((abs(n̂[I,1])+abs(n̂[I,2])+abs(n̂[I,3]))==0.0 || f == 0.0 || f == 1.0)
            flux = f*v
        else
            dl = v
            a = α[ICell]
            if (dl > 0.0) a -= n̂[ICell,d]*(1.0-dl) end
            n̂[ICell,d] *= abs(dl)
            flux = vof_vol(n̂[ICell,1],n̂[ICell,2],n̂[ICell,3], a)*v
        end
    end
    return flux
end

function vof_face!(f)
    N = size(f)
    n = size(N)[1]
    vof_smooth!(2, f, fsmooth)
    for d ∈ 1:n
    end
end


function freeint_update!(δt, f, fᶠ, n̂, α, u, u⁰, c̄;perdir=(0,),dirdir=(0,))
    tol = 1e-10
    N,n = size_u(u)
    insideI = inside(f)

    # Gil Strang splitting: see https://www.asc.tuwien.ac.at/~winfried/splitting/
    opOrder = [3,2,1,2,3]
    opCoeff = [0.5,0.5,1.0,0.5,0.5]

    c̄ .= ifelse.(f .<= 0.5, 0, 1)  # inside the operator or not????
    for iOp ∈ CartesianIndices(opOrder)
        d = opOrder[iOp]
        uMulp = opCoeff[iOp]*δt
        vof_reconstruct!(f,α,n̂,perdir=perdir,dirdir=dirdir)
        vof_flux!(d,f,α,n̂,u,u⁰,uMulp,fᶠ)
        @loop (
            f[I] += -∂(d,I+δ(d,I),fᶠ) + c̄[I]*(∂(d,I,u)+∂(d,I,u⁰))*0.5uMulp
        ) over I ∈ inside(f)

        maxf, maxid = findmax(f[insideI])
        minf, minid = findmin(f[insideI])
        if maxf-1 > tol
            throw(DomainError(maxf, "f$maxid ∉ [0,1]"))
        end
        if minf < -tol
            throw(DomainError(minf, "f$minid ∉ [0,1]"))
        end
        # clamp!(f,0.0,1.0)
        f[abs.(f).<=tol] .= 0.0
        f[abs.(f .-1).<=tol] .= 1.0

        BCVOF!(f,α,n̂,perdir=perdir,dirdir=dirdir)
    end
    fᶠ .= f
end
