using WaterLily
using Profile
using CUDA
using NVTX

println(pathof(WaterLily))

function TGV3D(N; Re=Inf, We=Inf, λμ=1e-2, λρ=1e-3, T=Float64, mem=CuArray,pert=0.,mp=true,perdir=(1,2,3))
    NN = (N,N,N)
    U = T(1)
    κ=T(2π/N)
    function uλ(i,xyz)
        x,y,z = @. xyz*κ
        i==1 && return -U*sin(x)*cos(y)*cos(z)
        i==2 && return  U*cos(x)*sin(y)*cos(z)
        return 0*U
    end

    return Simulation(
        NN, (0, 0, 0), N;
        T=T, uλ=uλ, mem=mem, perdir=(1,2,3)
    )
end

function TGVpert3D(N; Re=Inf, We=Inf, λμ=1e-2, λρ=1e-3, T=Float64, mem=CuArray,pert=0.8,mp=true,perdir=(1,2,3))
    NN = (N,N,N)
    U = 1
    zeroT = T(0)
    κ = T(π/N)
    pertT = T(pert)

    function uλ(i,xyz)
        x,y,z = @. (2xyz - N)*κ                # scaled coordinates
        i==1 && return -U*(sin(x)*cos(y)*cos(z)+pertT*sin(5x)*cos(5y)*cos(5z)) # u_x
        i==2 && return  U*(cos(x)*sin(y)*cos(z)+pertT*cos(5x)*sin(5y)*cos(5z)) # u_y
        return zeroT                              # u_z
    end

    return Simulation(
        NN, (0, 0, 0), N;
        T=T, uλ=uλ, mem=mem, perdir
    )
end

function prerunsim(N)
    sim = TGV3D(N;T=Float32)
    for i ∈ 1:100
        mom_step!(sim.flow,sim.pois)
        print("$(i) ")
    end
    println()
    return sim
end

function runsim!(sim)
    for i ∈ 1:500
        mom_step!(sim.flow,sim.pois)
    end
end

function main()
    sim = prerunsim(128)
    runsim!(sim)
end

sim = prerunsim(256)

@timev runsim!(sim)

Profile.init(n = 10^9, delay = 0.001)

# @profview main()
# @profview @timev runsim!(sim)
# CUDA.@profile external=true runsimNVTX!(sim)