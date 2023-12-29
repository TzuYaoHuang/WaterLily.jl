using StaticArrays


function sineWave2D(N; Re=4000, Fr=1, λμ=1e-3, λρ=1e-2, T=Float32, mem=Array)
    NN = (N,N)
    λ = N
    g = 1
    U = Fr*√(g*λ)
    ν = U*λ/Re

    grav(i,t) = i==1 ? 0 : -g

    # Interface function
    function Inter(xyz)
        x,y = @. (xyz-1.5-N/2)
        return y + 0.4λ*cos(2π*x/λ)
    end

    return WaterLily.TwoPhaseSimulation(
        NN, (0, 0), λ;
        U=U, Δt=0.001, ν=ν, InterfaceSDF=Inter, T=T, λμ=λμ, λρ=λρ, mem=mem, g=grav
    )
end

function TGVBubble2D(N; Re=4000, Fr=1, λμ=1e-3, λρ=1e-2, T=Float32, mem=Array,pert=0.1)
    NN = (N,N)
    U = 1
    R = N/4
    ν = U*N/Re*0

    function uλ(i,xyz)
        x,y = @. (xyz-1.5)*2π/N                # scaled coordinates
        i==1 && return -U*(sin(x)*cos(y)+pert*sin(5x)*cos(5y)) # u_x
        i==2 && return  U*(cos(x)*sin(y)+pert*cos(5x)*sin(5y)) # u_y
        return 0.                              # u_z
    end

    # Interface function
    function Inter(xyz)
        x,y = @. (xyz-1.5-N/2)
        return √(x^2+y^2) - R
    end

    return WaterLily.TwoPhaseSimulation(
        NN, (0, 0), R;
        U=U, Δt=0.001, ν=ν, InterfaceSDF=Inter, T=T, uλ=uλ, λμ=λμ, λρ=λρ, mem=mem
    )
end

function damBreak2D(N; Re=493.954, Fr=1, λμ=1e-3, λρ=1e-2, T=Float32, mem=Array)
    NN = (N,N)
    H = N/4
    g=1
    U = Fr*√(g*H)
    ν = U*H/Re

    grav(i,t) = i==1 ? 0 : -g

    function Inter(xx)
        x = xx .-1.5 .- SA[H,2H]
        return √sum((xi) -> max(0,xi)^2,x) + min(0, maximum(x))
    end

    return WaterLily.TwoPhaseSimulation(
        NN, (0,0), H;
        U=U, Δt=0.001, ν=ν, InterfaceSDF=Inter, T=T, λμ=λμ, λρ=λρ, mem=mem, g=grav
    )
end

function fallingDroplet2D(N; Re=493.954, Fr=1, λμ=1e-3, λρ=1e-2, T=Float32, mem=Array)
    NN = (N,N)
    R = N/4
    g = 1
    U = Fr*√(g*R)
    ν = U*R/Re

    uλ=(i,x) -> ifelse(i==1,0,0)
    grav(i,t) = i==1 ? 0 : -g

    function Inter(xx)
        x,y = @. xx-1.5
        return (((x-N/2)^2+(y-N/2)^2)^0.5 - R)
    end

    return WaterLily.TwoPhaseSimulation(
        NN, (0,0), R;
        U=U, Δt=0.001, ν=ν, InterfaceSDF=Inter, T=T, uλ=uλ, λμ=λμ, λρ=λρ, mem=mem, g=grav,perdir=(1,)
    )
end

function sineWave3D(N; Re=4000, Fr=1, λμ=1e-3, λρ=1e-2, T=Float32, mem=Array)
    NN = (N,N,N)
    λ = N
    g = 1
    U = Fr*√(g*λ)
    ν = U*λ/Re

    grav(i,t) = i==3 ? -g : 0

    # Interface function
    function Inter(xyz)
        x,y,z = @. (xyz-1.5-N/2)
        return z - 0.25λ*cos(2π*x/λ)*cos(2π*y/λ)
    end

    return WaterLily.TwoPhaseSimulation(
        NN, (0, 0, 0), λ;
        U=U, Δt=0.001, ν=ν, InterfaceSDF=Inter, T=T, λμ=λμ, λρ=λρ, mem=mem, g=grav
    )
end

function VerticalJet2D(N; Re=4000, Fr=1, λμ=1e-3, λρ=1e-2, T=Float32, mem=Array, aₙ=[0.9701,0.3270])
    NN = (N,2N)
    λ = N
    g = 1
    U = Fr*√(g*λ)
    ν = U*λ/Re

    grav(i,t) = i==1 ? 0 : -g

    function xLH(NPoint, an)
        ξ = collect(LinRange(0., 2π, NPoint))
        x = ξ * 1
        for n ∈ 1:size(an)[1]
            x .+= 1/n*an[n]*sin.(n*ξ)
        end
        return x .- π
    end
    
    function yLH(NPoint, an)
        ξ = collect(LinRange(0., 2π, NPoint))
        y₀=-sum((n)->1/(2*n)*an[n]^2, 1:size(an)[1])
        println(y₀)
        y = ξ*0 .+ y₀
        for n ∈ 1:size(an)[1]
            y .+= 1/n*an[n]*cos.(n*ξ)
        end
        return y
    end

    nPoints = 10000
    xList = xLH(nPoints,aₙ)/2π*λ
    yList = yLH(nPoints,aₙ)/2π*λ
    interpLH = linear_interpolation(xList, yList)

    # Interface function
    function Inter(xyz)
        x,y = @. (xyz-1.5-N/2)
        return y - interpLH(x)
    end

    return WaterLily.TwoPhaseSimulation(
        NN, (0, 0), λ;
        U=U, Δt=0.001, ν=ν, InterfaceSDF=Inter, T=T, λμ=λμ, λρ=λρ, mem=mem, g=grav
    )
end

