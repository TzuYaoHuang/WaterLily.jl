module WaterLilyCUDAExt

using CUDA, WaterLily
import WaterLily: L₂, backend_sync!

"""
    __init__()

Asserts CUDA is functional when loading this extension.
"""
__init__() = @assert CUDA.functional()

backend_sync!(::CuArray) = CUDA.synchronize()

"""
    L₂(a)

L₂ norm of CUDA array `a` excluding ghosts.
"""
L₂(a::CuArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module
