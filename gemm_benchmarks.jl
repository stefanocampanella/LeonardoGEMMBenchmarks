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

const cpu_types = [Float32, Float64]
const gpu_types = [BFloat16, Float16, Float32, Float64]
const gpu_modes = [CUDA.DEFAULT_MATH, CUDA.FAST_MATH]

function rand_xpu(device, args...)
    if device == :cpu 
        rand(args...)
    elseif device == :gpu
        CUDA.rand(args...)
    else
        throw(ArgumentError("Device must be either :gpu or :cpu, got $device"))
    end
end

random_matrix(T, size, device) = 2rand_xpu(device, T, size, size) .- one(T)
gemm_init(T, size, device) = random_matrix(T, size, device), random_matrix(T, size, device)

if abspath(PROGRAM_FILE) == @__FILE__
    const filename = ARGS[1]
    const sizes = map(s -> parse(Int, s), split(ARGS[2]))

    suite = BenchmarkGroup()

    suite["cpu"] = BenchmarkGroup()
    for T in cpu_types, size in sizes
        suite["cpu"][T, size] = @benchmarkable A * B setup=((A, B) = gemm_init($T, $size, :cpu))
    end

    for mode in [CUDA.DEFAULT_MATH, CUDA.FAST_MATH]
        suite["gpu", string(mode)] = BenchmarkGroup()
        for T in [BFloat16, Float16, Float32, Float64], size in sizes
            suite["gpu", string(mode)][T, size] = @benchmarkable (CUDA.@sync A * B) setup=((A, B) = gemm_init($T, $size, :gpu); CUDA.math_mode!($mode))
        end
    end

    tune!(suite)
    results = run(suite, verbose=true)

    jldsave(filename; results)
end