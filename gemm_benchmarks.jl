using InteractiveUtils: versioninfo
using CUDA
using MKL
using LinearAlgebra
using BenchmarkTools
using BFloat16s
using JLD2

BLAS.set_num_threads(Sys.CPU_THREADS)

@info "Julia version info"
versioninfo()
@info "BLAS config" BLAS.get_config()
@info "BLAS num threads" BLAS.get_num_threads()
@info "CUDA version info" CUDA.versioninfo()

abstract type Device end
struct CPU <: Device end
struct GPU <: Device end

struct TFloat32 end

const cpu_types = [Float32, Float64]
const gpu_types = [BFloat16, Float16, Float32, TFloat32, Float64]

function Base.rand(dev::Device, args...)
    if dev isa CPU
        rand(args...)
    elseif dev isa GPU 
        CUDA.rand(args...)
    end
end

random_matrix(T, size, device) = 2rand(device, T, size, size) .- one(T)
gemm_init(T, size, device) = random_matrix(T, size, device), random_matrix(T, size, device)

if abspath(PROGRAM_FILE) == @__FILE__
    const filename = ARGS[1]
    const sizes = map(s -> parse(Int, s), ARGS[2:end])

    suite = BenchmarkGroup()

    for T in cpu_types
        suite["cpu", T] = BenchmarkGroup()
        for size in sizes
            suite["cpu", T][size] = @benchmarkable A * B setup=((A, B) = gemm_init($T, $size, CPU()))
        end
    end

    for T in gpu_types
        suite["gpu", T] = BenchmarkGroup()
        if T == TFloat32
            mode = CUDA.FAST_MATH
            eltype = Float32
        else
            mode = CUDA.DEFAULT_MATH
            eltype = T
        end
        for size in sizes
            suite["gpu", T][size] = @benchmarkable (CUDA.@sync A * B) setup=((A, B) = gemm_init($eltype, $size, GPU()); CUDA.math_mode!($mode))
        end
    end

    tune!(suite)
    results = run(suite, verbose=true)

    jldsave(filename; results)
end