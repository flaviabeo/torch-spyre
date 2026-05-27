# Direct c10d Lowering Prototype

## Problem Statement 
As workloads scale across multiple AIUs, collective communication becomes a significant component of execution time. While Torch-Spyre can lower compute operations through the compiler stack, communication operations are traditionally handled outside the compiler's optimization domain. 

As a result, communication and compute follow separate execution paths, creating boundaries that can contribute to device idle time and limiting opportunities for communication-aware optimizations.

The long-term goal of this work is to reduce device idle time in multi-rank workloads by bringing collective communication into the Torch-Spyre execution model, enabling future optimizations.

## Current Implementation

Keep collectives inside the Torch-Spyre compilation flow with async operations, routing them through the native Spyre communication stack.

Instead of:

```
FX Graph
   ↓
c10d Functional Collective
   ↓
Fallback / external communication path
```

we now have:

```
FX Graph
   ↓
Direct Lowering (async ops)
   ↓
Torch-Spyre Compilation
   ↓
C++ Dispatcher (non-blocking)
   ↓
spyre-comms (WorkSchedule tracking)
```

## What Was Implemented

### 1. Direct c10d Lowering with Async Operations

PyTorch's `_c10d_functional.broadcast` and `wait_tensor` are lowered directly to async IR nodes without FX graph transformation.

Example:
```python
torch.ops._c10d_functional.broadcast(x, 0, "default")
torch.ops._c10d_functional.wait_tensor(x)
```

is lowered to async operations:
```python
torch.ops.spyre.broadcast_async(x, 0, "default")  # Non-blocking
torch.ops.spyre.wait_work(x)  # Synchronize
```

### 2. Compiler Compatibility

Custom Spyre collectives remain visible during:
- AOT Autograd tracing
- Inductor compilation

through custom op registration and fake/meta implementations.

This allows collectives to survive the compilation pipeline rather than being removed or causing graph breaks.

### 3. Runtime Integration

Lowered collectives are dispatched through:

```
Torch-Spyre
   ↓
C++ Dispatcher
   ↓
spyre-comms API
```

allowing execution on the native Spyre communication stack.

## What This Enables

**Architectural Foundation**: Functional collectives are now first-class operations in the Torch-Spyre compilation flow with async execution primitives.

Communication is visible to the compiler:

```
FX Graph (_c10d_functional.broadcast)
   ↓
Direct Lowering (no FX pass)
   ↓
IR Node (SpyreBroadcastAsyncFallback)
   ↓
Generated Code (torch.ops.spyre.broadcast_async)
   ↓
C++ Dispatcher (non-blocking)
   ↓
spyre-comms (WorkSchedule tracking)
   ↓
wait_tensor → wait_work (synchronize)
```

This implementation provides:
- ✅ Async collective primitives (broadcast_async, wait_work)
- ✅ WorkSchedule tracking infrastructure (PendingWork struct)
- ✅ Compiler visibility into communication operations
- ✅ Prevents eager fallback (reduces device idle time)
- ✅ Memory safety and error handling

**Note**: This enables future **compiler-driven optimizations** beyond what async streams alone provide. The compiler can now see communication operations as first-class IR nodes, providing the foundation for future scheduling improvements.

Future compiler-driven optimizations enabled by this work:
- Automatic overlap scheduling (requires scheduler enhancements to reorder independent ops between async/wait)
- Memory residency hints for communication buffers
- Scratchpad-aware collective scheduling
- Communication-aware fusion decisions
- Cross-collective optimization opportunities

## Implementation Details

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PyTorch User Code                                        │
│    torch.ops._c10d_functional.broadcast(x, 0, "default")    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Direct Lowering (lowering.py)                            │
│    @register_spyre_lowering(_c10d_functional.broadcast)     │
│    Creates IR node: SpyreBroadcastAsyncFallback             │
│    @register_spyre_lowering(_c10d_functional.wait_tensor)   │
│    Creates IR node: SpyreWaitWorkFallback                   │
│    NO FX pass transformation                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. IR Nodes (ir.py)                                         │
│    SpyreBroadcastAsyncFallback.codegen()                    │
│    Generates: torch.ops.spyre.broadcast_async(x, 0, "...")  │
│    SpyreWaitWorkFallback.codegen()                          │
│    Generates: torch.ops.spyre.wait_work(x)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Custom Op Registration (spyre_library.py)               │
│    @torch.library.custom_op("spyre::broadcast_async")       │
│    @torch.library.custom_op("spyre::wait_work")             │
│    - Defines schemas for PyTorch dispatcher                 │
│    - Provides fake implementations for shape inference      │
│    - NO runtime logic (delegated to C++)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. C++ Dispatcher (spyre_distributed.cpp)                   │
│    TORCH_LIBRARY(spyre, m)                                  │
│    TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m)                │
│    m.impl("broadcast_async", &spyre_broadcast_async_impl);  │
│    m.impl("wait_work", &spyre_wait_work_impl);              │
│    - PendingWork struct (WorkSchedule + tensor)             │
│    - Thread-safe map with data_ptr keys                     │
│    - Direct spyre-comms calls (non-blocking)                │
└─────────────────────────────────────────────────────────────┘
```

### Key Files

**torch_spyre/_inductor/lowering.py**:
Registers async lowering functions for broadcast and wait_tensor

**torch_spyre/_inductor/ir.py**:
Defines `SpyreBroadcastAsyncFallback` and `SpyreWaitWorkFallback` IR nodes

**torch_spyre/_inductor/distributed/spyre_library.py**:
Custom op registration for broadcast_async and wait_work with fake implementations

**torch_spyre/csrc/distributed/spyre_distributed.cpp**:
C++ async implementation with PendingWork tracking:
```cpp
struct PendingWork {
  std::shared_ptr<spyre_comms::WorkSchedule> work;
  at::Tensor output;  // Keeps storage alive
};
std::unordered_map<void*, PendingWork> pending_work_map_;
```

**torch_spyre/_inductor/distributed/kernels.py**:
Eager mode placeholders for c10d ops

**examples/test_c10d_async_lowering.py**:
Demo showing automatic async broadcast lowering

## Future Work and Compiler-Driven Optimizations (needs exploration)

1. Add scratchpad / residency hints
2. Expand to more collectives (all_reduce, all_gather, reduce_scatter)
3. Explore overlap between compute and communication using streams
4. Identify other optimizations for reduced device idle time
5. Gather evidence of this prototype and performance gains over pure asynchronous runtime