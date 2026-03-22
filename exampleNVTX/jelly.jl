using WaterLily
using StaticArrays
using Profile
using CUDA

function jelly(p, backend; Re=5e2, U=1, T=Float32)
    n = 2^p; R = T(2n/3); h = 4n - 2R; ν = U*R/Re
    ω = 2U/R
    @fastmath @inline A(t) = 1 .- SA[1,1,0]*cos(ω*t)/10
    @fastmath @inline B(t) = SA[0,0,1]*((cos(ω*t) - 1)*R/4-h)
    @fastmath @inline C(t) = SA[0,0,1]*sin(ω*t)*R/4
    sphere = AutoBody((x,t)->abs(√sum(abs2, x) - R) - 1,
                      (x,t)->A(t).*x + B(t) + C(t))
    plane = AutoBody((x,t)->x[3] - h, (x, t) -> x + C(t))
    body =  sphere - plane
    Simulation((n, n, 4n), (0, 0, -U), R; ν, body, T, mem=backend)
end

sim = jelly(7,CuArray)
for i ∈ 1:10
    measure!(sim)
    WaterLily.mom_step!(sim.flow, sim.pois)
    print("$(i)\n")
end