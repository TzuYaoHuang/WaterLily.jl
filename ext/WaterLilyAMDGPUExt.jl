module WaterLilyAMDGPUExt

using AMDGPU, WaterLily
import WaterLily: L₂, backend_sync!

"""
    __init__()

Asserts AMDGPU is functional when loading this extension.
"""
__init__() = @assert AMDGPU.functional()

backend_sync!(::ROCArray) = AMDGPU.synchronize()

"""
    L₂(a)

L₂ norm of ROCArray `a` excluding ghosts.
"""
L₂(a::ROCArray,R::CartesianIndices=inside(a)) = mapreduce(abs2,+,@inbounds(a[R]))

end # module
