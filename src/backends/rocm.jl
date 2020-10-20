import AMDGPU
import AMDGPU: rocfunction, DevicePtr, HSAAgent, HSAQueue, HSASignal, Mem

const FREE_QUEUES = HSAQueue[]
const QUEUES = HSAQueue[]
const QUEUE_GC_THRESHOLD = Ref{Int}(16)

# This code is loaded after an `@init` step
if haskey(ENV, "KERNELABSTRACTIONS_QUEUES_GC_THRESHOLD")
    global QUEUE_GC_THRESHOLD[] = parse(Int, ENV["KERNELABSTRACTIONS_QUEUES_GC_THRESHOLD"])
end

## Stream GC
# Simplistic stream gc design in which when we have a total number
# of streams bigger than a threshold, we start scanning the streams
# and add them back to the freelist if all work on them has completed.
# Alternative designs:
# - Enqueue a host function on the stream that adds the stream back to the freelist
# - Attach a finalizer to events that adds the stream back to the freelist
# Possible improvements
# - Add a background task that occasionally scans all streams
# - Add a hysterisis by checking a "since last scanned" timestamp
# - Add locking
function next_queue()
    if !isempty(FREE_QUEUES)
        return pop!(FREE_QUEUES)
    end

    if length(QUEUES) > QUEUE_GC_THRESHOLD[]
        for queue in QUEUES
            if AMDGPU.queued_packets(queue) == 0
                push!(FREE_QUEUES, queue)
            end
        end
    end

    if !isempty(FREE_QUEUES)
        return pop!(FREE_QUEUES)
    end

    # FIXME: Which agent?
    queue = HSAQueue()
    push!(QUEUES, queue)
    return queue
end

struct ROCEvent <: Event
    event::HSASignal
end

function Event(::ROC)
    queue = AMDGPU.get_default_queue()
    event = HSASignal(AMDGPU.EVENT_DISABLE_TIMING)
    AMDGPU.record(event, queue)
    ROCEvent(event)
end

wait(ev::ROCEvent, progress=nothing) = wait(CPU(), ev, progress)

function wait(::CPU, ev::ROCEvent, progress=nothing)
    if progress === nothing
        AMDGPU.synchronize(ev.event)
    else
        while !AMDGPU.query(ev.event)
            progress()
            # do we need to `yield` here?
        end
    end
end

# Use this to synchronize between computation using the CuDefaultStream
wait(::ROC, ev::ROCEvent, progress=nothing, queue=AMDGPU.CuDefaultStream()) = AMDGPU.wait(ev.event, queue)
wait(::ROC, ev::NoneEvent, progress=nothing, queue=nothing) = nothing

# There is no efficient wait for CPU->GPU synchronization, so instead we
# do a CPU wait, and therefore block anyone from submitting more work.
# We maybe could do a spinning wait on the GPU and atomic flag to signal from the CPU,
# but which queue would we target?
wait(::ROC, ev::CPUEvent, progress=nothing, queue=nothing) = wait(CPU(), ev, progress)

function wait(::ROC, ev::MultiEvent, progress=nothing, queue=AMDGPU.CuDefaultStream())
    dependencies = collect(ev.events)
    cudadeps  = filter(d->d isa ROCEvent,    dependencies)
    otherdeps = filter(d->!(d isa ROCEvent), dependencies)
    for event in cudadeps
        AMDGPU.wait(event.event, queue)
    end
    for event in otherdeps
        wait(ROC(), event, progress)
    end
end

###
# async_copy
###
# - IdDict does not free the memory
# - WeakRef dict does not unique the key by objectid
const __pinned_memory = Dict{UInt64, WeakRef}()

function __pin!(a)
    # use pointer instead of objectid?
    oid = objectid(a)
    if haskey(__pinned_memory, oid) && __pinned_memory[oid].value !== nothing
        return nothing
    end
    ad = Mem.register(Mem.Host, pointer(a), sizeof(a))
    finalizer(_ -> Mem.unregister(ad), a)
    __pinned_memory[oid] = WeakRef(a)
    return nothing
end

function async_copy!(::ROC, A, B; dependencies=nothing, progress=yield)
    A isa Array && __pin!(A)
    B isa Array && __pin!(B)

    queue = next_queue()
    wait(ROC(), MultiEvent(dependencies), progress, queue)
    event = HSASignal(AMDGPU.EVENT_DISABLE_TIMING)
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        unsafe_copyto!(destptr, srcptr, N, async=true, queue=queue)
    end

    AMDGPU.record(event, queue)

    return ROCEvent(event)
end



###
# Kernel launch
###
function (obj::Kernel{ROC})(args...; ndrange=nothing, dependencies=nothing, workgroupsize=nothing, progress=yield)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    queue = next_queue()
    wait(ROC(), MultiEvent(dependencies), progress, queue)

    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        # TODO: allow for NDRange{1, DynamicSize, DynamicSize}(nothing, nothing)
        #       and actually use AMDGPU autotuning
        workgroupsize = (256,)
    end
    # If the kernel is statically sized we can tell the compiler about that
    if KernelAbstractions.workgroupsize(obj) <: StaticSize
        maxthreads = prod(get(KernelAbstractions.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    iterspace, dynamic = partition(obj, ndrange, workgroupsize)

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    ctx = mkcontext(obj, ndrange, iterspace)
    # Launch kernel
    event = HSASignal(AMDGPU.EVENT_DISABLE_TIMING)
    AMDGPU.@cuda(threads=threads, blocks=nblocks, queue=queue,
                     name=String(nameof(obj.f)), maxthreads=maxthreads,
                     Cassette.overdub(ctx, obj.f, args...))

    AMDGPU.record(event, queue)
    return ROCEvent(event)
end

Cassette.@context ROCCtx

function mkcontext(kernel::Kernel{ROC}, _ndrange, iterspace)
    metadata = CompilerMetadata{ndrange(kernel), true}(_ndrange, iterspace)
    Cassette.disablehooks(ROCCtx(pass = CompilerPass, metadata=metadata))
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__index_Local_Linear))
    return AMDGPU.threadIdx().x
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__index_Group_Linear))
    return AMDGPU.blockIdx().x
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__index_Global_Linear))
    I =  @inbounds expand(__iterspace(ctx.metadata), AMDGPU.blockIdx().x, AMDGPU.threadIdx().x)
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx.metadata))[I]
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__index_Local_Cartesian))
    @inbounds workitems(__iterspace(ctx.metadata))[AMDGPU.threadIdx().x]
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__index_Group_Cartesian))
    @inbounds blocks(__iterspace(ctx.metadata))[AMDGPU.blockIdx().x]
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__index_Global_Cartesian))
    return @inbounds expand(__iterspace(ctx.metadata), AMDGPU.blockIdx().x, AMDGPU.threadIdx().x)
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__validindex))
    if __dynamic_checkbounds(ctx.metadata)
        I = @inbounds expand(__iterspace(ctx.metadata), AMDGPU.blockIdx().x, AMDGPU.threadIdx().x)
        return I in __ndrange(ctx.metadata)
    else
        return true
    end
end

generate_overdubs(ROCCtx)

###
# ROC specific method rewrites
###

@inline Cassette.overdub(::ROCCtx, ::typeof(^), x::Float64, y::Float64) = AMDGPU.pow(x, y)
@inline Cassette.overdub(::ROCCtx, ::typeof(^), x::Float32, y::Float32) = AMDGPU.pow(x, y)
@inline Cassette.overdub(::ROCCtx, ::typeof(^), x::Float64, y::Int32)   = AMDGPU.pow(x, y)
@inline Cassette.overdub(::ROCCtx, ::typeof(^), x::Float32, y::Int32)   = AMDGPU.pow(x, y)
@inline Cassette.overdub(::ROCCtx, ::typeof(^), x::Union{Float32, Float64}, y::Int64) = AMDGPU.pow(x, y)

# libdevice.jl
const cudafuns = (:cos, :cospi, :sin, :sinpi, :tan,
          :acos, :asin, :atan,
          :cosh, :sinh, :tanh,
          :acosh, :asinh, :atanh,
          :log, :log10, :log1p, :log2,
          :exp, :exp2, :exp10, :expm1, :ldexp,
          # :isfinite, :isinf, :isnan, :signbit,
          :abs,
          :sqrt, :cbrt,
          :ceil, :floor,)
for f in cudafuns
    @eval function Cassette.overdub(ctx::ROCCtx, ::typeof(Base.$f), x::Union{Float32, Float64})
        @Base._inline_meta
        return AMDGPU.$f(x)
    end
end

@inline Cassette.overdub(::ROCCtx, ::typeof(sincos), x::Union{Float32, Float64}) = (AMDGPU.sin(x), AMDGPU.cos(x))
@inline Cassette.overdub(::ROCCtx, ::typeof(exp), x::Union{ComplexF32, ComplexF64}) = AMDGPU.exp(x)


###
# GPU implementation of shared memory
###
@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(SharedMemory), ::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = AMDGPU._shmem(Val(Id), T, Val(prod(Dims)))
    AMDGPU.ROCDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(Scratchpad), ::Type{T}, ::Val{Dims}) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__synchronize))
    AMDGPU.sync_threads()
end

@inline function Cassette.overdub(ctx::ROCCtx, ::typeof(__print), args...)
    AMDGPU._cuprint(args...)
end

###
# GPU implementation of `@Const`
###
struct ConstROCDeviceArray{T,N,A} <: AbstractArray{T,N}
    shape::Dims{N}
    ptr::DevicePtr{T,A}

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    ConstROCDeviceArray{T,N,A}(shape::Dims{N}, ptr::DevicePtr{T,A}) where {T,A,N} = new(shape,ptr)
end

Adapt.adapt_storage(to::ConstAdaptor, a::AMDGPU.ROCDeviceArray{T,N,A}) where {T,N,A} = ConstROCDeviceArray{T, N, A}(a.shape, a.ptr)

Base.pointer(a::ConstROCDeviceArray) = a.ptr
Base.pointer(a::ConstROCDeviceArray, i::Integer) =
    pointer(a) + (i - 1) * Base.elsize(a)

Base.elsize(::Type{<:ConstROCDeviceArray{T}}) where {T} = sizeof(T)
Base.size(g::ConstROCDeviceArray) = g.shape
Base.length(g::ConstROCDeviceArray) = prod(g.shape)
Base.IndexStyle(::Type{<:ConstROCDeviceArray}) = Base.IndexLinear()

Base.unsafe_convert(::Type{DevicePtr{T,A}}, a::ConstROCDeviceArray{T,N,A}) where {T,A,N} = pointer(a)

@inline function Base.getindex(A::ConstROCDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    AMDGPU.unsafe_cached_load(pointer(A), index, Val(align))::T
end

@inline function Base.unsafe_view(arr::ConstROCDeviceArray{T, 1, A}, I::Vararg{Base.ViewIndex,1}) where {T, A}
    ptr = pointer(arr) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    return ConstROCDeviceArray{T,1,A}(len, ptr)
end
