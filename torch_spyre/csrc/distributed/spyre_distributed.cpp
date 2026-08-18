/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#include <algorithm>
#include <flex/flex.hpp>
#include <memory>
#include <mutex>
#include <spyre_comms.hpp>
#include <spyre_comms_tensor.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../logging.h"
#include "../spyre_allocator.h"
#include "../spyre_stream.h"
#include "../spyre_tensor_impl.h"

namespace spyre {

// Identifies which collective produced the pending work
enum class CollectiveKind { Broadcast, AllReduce, Reduce };

// Structure to hold pending async work
struct PendingWork {
  std::shared_ptr<spyre_comms::WorkSchedule> work;
  CollectiveKind kind;
  // Keep tensors alive while communication is in-flight to avoid UAF
  std::vector<at::Tensor> hold_tensors;
};

// Global map to track pending async operations
// Key: SharedOwnerCtx* (stable per-allocation identity, never reused), Value:
// PendingWork
static std::unordered_map<spyre::SharedOwnerCtx*, PendingWork>
    pending_work_map_;
static std::mutex work_map_mutex_;

// Compile-time plan cache.
enum class PlanKind { Broadcast, AllReduce, Reduce };

struct CachedPlan {
  PlanKind kind;
  spyre_comms::TensorDataTypeEnum dtype;
  int64_t rank_param;  // src_rank (broadcast) or dst_rank (reduce); unused for
                       // allreduce
  spyre_comms::SpyreReductionOpType reduce_op;  // only for allreduce/reduce

  // tensor_info MUST outlive wsi — spyre_comms stores a non-owning reference
  // to it inside the WorkScheduleInfo's sentinel envelope.
  std::unique_ptr<spyre_comms::TensorInfo> tensor_info;
  std::unique_ptr<spyre_comms::WorkScheduleInfo> wsi;
  int64_t num_elems = 0;
};
static std::vector<CachedPlan> wsi_cache_;
static std::mutex wsi_cache_mutex_;

// Helper to convert PyTorch ScalarType to spyre_comms TensorDataTypeEnum
spyre_comms::TensorDataTypeEnum torch_dtype_to_spyre_comms(
    c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return spyre_comms::TensorDataTypeEnum::float32;
    case c10::ScalarType::Double:
      return spyre_comms::TensorDataTypeEnum::float64;
    case c10::ScalarType::Half:
      return spyre_comms::TensorDataTypeEnum::float16;
    case c10::ScalarType::BFloat16:
      return spyre_comms::TensorDataTypeEnum::bfloat16;
    case c10::ScalarType::Int:
      return spyre_comms::TensorDataTypeEnum::int32;
    case c10::ScalarType::Long:
      return spyre_comms::TensorDataTypeEnum::int64;
    case c10::ScalarType::Short:
      return spyre_comms::TensorDataTypeEnum::int16;
    case c10::ScalarType::Char:
      return spyre_comms::TensorDataTypeEnum::int8;
    case c10::ScalarType::Byte:
      return spyre_comms::TensorDataTypeEnum::uint8;
    case c10::ScalarType::Bool:
      return spyre_comms::TensorDataTypeEnum::boolean;
    default:
      TORCH_CHECK(false, "Unsupported dtype for spyre_comms: ", dtype);
  }
}

// Helper to get CompositeAddress pointer from a Spyre tensor
// NOTE: The returned pointer is valid only as long as the tensor's storage
// context remains valid. Caller must keep the tensor alive.
const flex::CompositeAddress* get_composite_address(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "Tensor must be on Spyre device for distributed operations");

  TORCH_CHECK(tensor.is_contiguous(),
              "Tensor must be contiguous for distributed operations");

  auto* spyre_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  TORCH_CHECK(spyre_impl != nullptr, "SpyreTensorImpl is null");

  auto& storage = spyre_impl->storage();
  auto* data_ptr = storage.data_ptr().get();
  TORCH_CHECK(data_ptr != nullptr, "Storage data pointer is null");

  auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null");

  // Return a pointer to the CompositeAddress inside the context
  return &ctx->composite_addr;
}

// Drain all pending async collective operations.
void drain_pending_work() {
  std::lock_guard<std::mutex> lock(work_map_mutex_);
  for (auto& [key, pw] : pending_work_map_) {
    if (pw.work) {
      pw.work->wait();
      pw.work = nullptr;
    }
  }
}

// Ensure spyre_comms is initialized and return the world context.
std::shared_ptr<spyre_comms::Context> ensure_context() {
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::getDefaultStreamRuntimeHandle());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }
  return context;
}

// Helper to convert reduce_op string to SpyreReductionOpType
spyre_comms::SpyreReductionOpType parse_reduce_op(
    const std::string& reduce_op) {
  if (reduce_op == "sum") {
    return spyre_comms::SpyreReductionOpType::SUM;
  }
  TORCH_CHECK(false, "Unsupported reduce_op for spyre allreduce: ", reduce_op,
              ". Only 'sum' is currently supported.");
}

// ============================================================================
// Compile-time plan ops — store collective parameters at graph load.
// The actual WSI is created lazily on the first run call when we have the
// real tensor and its device layout.
// ============================================================================

int64_t spyre_broadcast_plan_impl(int64_t dtype_code, int64_t src_rank,
                                  const std::string& group_name) {
  DEBUGINFO("spyre::broadcast_plan called with dtype=", dtype_code,
            ", src_rank=", src_rank);

  auto context = ensure_context();
  TORCH_CHECK(
      src_rank >= 0 && src_rank < static_cast<int64_t>(context->getSize()),
      "src_rank out of range: ", src_rank, " (world size is ",
      context->getSize(), ")");

  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(CachedPlan{PlanKind::Broadcast, dtype, src_rank,
                                  spyre_comms::SpyreReductionOpType::SUM,
                                  nullptr, 0});
  DEBUGINFO("broadcast_plan: stored params at handle=", handle);
  return handle;
}

int64_t spyre_allreduce_plan_impl(int64_t dtype_code,
                                  const std::string& reduce_op,
                                  const std::string& group_name) {
  DEBUGINFO("spyre::allreduce_plan called with dtype=", dtype_code,
            ", reduce_op=", reduce_op);

  ensure_context();
  auto op_type = parse_reduce_op(reduce_op);
  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(
      CachedPlan{PlanKind::AllReduce, dtype, 0, op_type, nullptr, 0});
  DEBUGINFO("allreduce_plan: stored params at handle=", handle);
  return handle;
}

int64_t spyre_reduce_plan_impl(int64_t dtype_code, int64_t dst_rank,
                               const std::string& reduce_op,
                               const std::string& group_name) {
  DEBUGINFO("spyre::reduce_plan called with dtype=", dtype_code,
            ", dst_rank=", dst_rank, ", reduce_op=", reduce_op);

  auto context = ensure_context();
  TORCH_CHECK(
      dst_rank >= 0 && dst_rank < static_cast<int64_t>(context->getSize()),
      "dst_rank out of range: ", dst_rank, " (world_size=", context->getSize(),
      ")");

  auto op_type = parse_reduce_op(reduce_op);
  auto dtype =
      torch_dtype_to_spyre_comms(static_cast<c10::ScalarType>(dtype_code));

  std::lock_guard<std::mutex> lock(wsi_cache_mutex_);
  int64_t handle = static_cast<int64_t>(wsi_cache_.size());
  wsi_cache_.push_back(
      CachedPlan{PlanKind::Reduce, dtype, dst_rank, op_type, nullptr, 0});
  DEBUGINFO("reduce_plan: stored params at handle=", handle);
  return handle;
}

// ============================================================================
// Runtime run ops — create WSI on first call, then reuse it.
// ============================================================================

// Compute device element count from a Spyre tensor's layout.
// This is the flat 1D element count that spyre_comms expects.
int64_t compute_device_elems(const at::Tensor& tensor) {
  SpyreTensorLayout stl = get_spyre_tensor_layout(tensor);
  uint64_t nbytes = get_device_size_in_bytes(stl);
  return static_cast<int64_t>(nbytes / tensor.element_size());
}

// Ensure the WSI is created for a cached plan entry.
// Must be called with wsi_cache_mutex_ held.
void ensure_wsi(CachedPlan& plan, int64_t num_elems,
                std::shared_ptr<spyre_comms::Context>& context) {
  if (plan.wsi != nullptr) return;

  spyre_comms::TensorShape shape({num_elems});
  plan.tensor_info =
      std::make_unique<spyre_comms::TensorInfo>(plan.dtype, shape);
  plan.num_elems = num_elems;

  switch (plan.kind) {
    case PlanKind::Broadcast:
      plan.wsi = context->broadcast(
          *plan.tensor_info,
          static_cast<spyre_comms::process_id_t>(plan.rank_param));
      break;
    case PlanKind::AllReduce:
      plan.wsi = context->allreduce(*plan.tensor_info, plan.reduce_op);
      break;
    case PlanKind::Reduce:
      plan.wsi = context->reduce(
          *plan.tensor_info, plan.reduce_op,
          static_cast<spyre_comms::process_id_t>(plan.rank_param));
      break;
  }
  TORCH_CHECK(plan.wsi != nullptr, "Failed to create WSI");
}

at::Tensor spyre_broadcast_run_impl(const at::Tensor& input,
                                    int64_t plan_handle, int64_t src_rank) {
  DEBUGINFO("spyre::broadcast_run called with plan_handle=", plan_handle,
            ", src_rank=", src_rank);

  drain_pending_work();

  auto context = ensure_context();

  // Look up plan and lazily create WSI
  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "broadcast_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  int64_t num_elems = compute_device_elems(input);
  ensure_wsi(plan, num_elems, context);

  // Create output tensor
  at::Tensor output = at::empty_like(input);
  TORCH_CHECK(output.nbytes() > 0,
              "Tensor must have non-zero size for broadcast");

  // Copy input to output if we're the source rank
  int current_rank = context->getRank();
  if (current_rank == src_rank) {
    output.copy_(input);
  }

  // Get SharedOwnerCtx for map key
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      output.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for output tensor");

  // Build spyre_comms::Tensor using the plan's TensorInfo (must stay alive)
  spyre_comms::Tensor buffer_tensor(*plan.tensor_info);
  buffer_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  auto work_schedule = context->broadcast_applyTensor(*plan.wsi, buffer_tensor);
  TORCH_CHECK(work_schedule != nullptr, "broadcast_run: applyTensor failed");

  work_schedule->start();

  // Store pending work
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "broadcast_run called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(ctx, PendingWork{std::move(work_schedule),
                                               CollectiveKind::Broadcast,
                                               {output}});
  }

  return output;
}

at::Tensor spyre_allreduce_run_impl(const at::Tensor& input,
                                    int64_t plan_handle) {
  DEBUGINFO("spyre::allreduce_run called with plan_handle=", plan_handle);

  auto context = ensure_context();

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for all_reduce");
  TORCH_CHECK(input.is_contiguous(),
              "Tensor must be contiguous for all_reduce");
  TORCH_CHECK(input.nbytes() > 0,
              "Tensor must have non-zero size for all_reduce");

  // Look up plan and create WSI
  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "allreduce_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  int64_t num_elems = compute_device_elems(input);
  ensure_wsi(plan, num_elems, context);

  // Get SharedOwnerCtx
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      input.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for input tensor");

  // Build spyre_comms::Tensor using the plan's TensorInfo (must stay alive)
  spyre_comms::Tensor inout_tensor(*plan.tensor_info,
                                   input.storage().data_ptr().get());
  inout_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  auto work_schedule = context->allreduce_applyTensor(*plan.wsi, inout_tensor);
  TORCH_CHECK(work_schedule != nullptr, "allreduce_run: applyTensor failed");

  work_schedule->start();

  // Store pending work
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "allreduce_run called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(
        ctx, PendingWork{
                 std::move(work_schedule), CollectiveKind::AllReduce, {input}});
  }

  return input;
}

at::Tensor spyre_reduce_run_impl(const at::Tensor& input, int64_t plan_handle,
                                 int64_t dst_rank) {
  DEBUGINFO("spyre::reduce_run called with plan_handle=", plan_handle,
            ", dst_rank=", dst_rank);

  drain_pending_work();

  auto context = ensure_context();

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for reduce");
  TORCH_CHECK(input.is_contiguous(), "Tensor must be contiguous for reduce");
  TORCH_CHECK(input.nbytes() > 0, "Tensor must have non-zero size for reduce");

  // Look up plan and create WSI
  std::lock_guard<std::mutex> cache_lock(wsi_cache_mutex_);
  TORCH_CHECK(
      plan_handle >= 0 && plan_handle < static_cast<int64_t>(wsi_cache_.size()),
      "reduce_run: invalid plan_handle=", plan_handle);
  auto& plan = wsi_cache_[static_cast<size_t>(plan_handle)];

  int64_t num_elems = compute_device_elems(input);
  ensure_wsi(plan, num_elems, context);

  // Get SharedOwnerCtx
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      input.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for input tensor");

  // Build spyre_comms::Tensor using the plan's TensorInfo (must stay alive)
  spyre_comms::Tensor inout_tensor(*plan.tensor_info,
                                   input.storage().data_ptr().get());
  inout_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  auto work_schedule = context->reduce_applyTensor(*plan.wsi, inout_tensor);
  TORCH_CHECK(work_schedule != nullptr, "reduce_run: applyTensor failed");

  work_schedule->start();

  // Store pending work
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "reduce_run called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(
        ctx,
        PendingWork{std::move(work_schedule), CollectiveKind::Reduce, {input}});
  }

  return input;
}

// ============================================================================
// Legacy async ops — used by the eager/interpreted path.
// ============================================================================

// Async broadcast implementation - returns immediately
at::Tensor spyre_broadcast_async_impl(const at::Tensor& input, int64_t src_rank,
                                      const std::string& group_name) {
  DEBUGINFO("spyre::broadcast_async called with src_rank=", src_rank,
            ", group=", group_name);

  // Drain prior in-flight collectives — hardware cannot overlap.
  drain_pending_work();

  // Get world context
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::getDefaultStreamRuntimeHandle());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }

  // Validate src_rank is in bounds
  TORCH_CHECK(
      src_rank >= 0 && src_rank < static_cast<int64_t>(context->getSize()),
      "src_rank out of range: ", src_rank, " (world size is ",
      context->getSize(), ")");

  // Create output tensor
  at::Tensor output = at::empty_like(input);
  TORCH_CHECK(output.nbytes() > 0,
              "Tensor must have non-zero size for broadcast");

  // Get SharedOwnerCtx for map key (stable per-allocation identity)
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      output.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for output tensor");

  // Get the device layout — pass the total buffer element count as a flat 1D
  // shape to avoid spyre_comms interpreting stick dimensions specially.
  SpyreTensorLayout stl = get_spyre_tensor_layout(output);
  uint64_t bcast_nbytes = get_device_size_in_bytes(stl);
  int64_t bcast_total_elems =
      static_cast<int64_t>(bcast_nbytes / input.element_size());

  spyre_comms::TensorDataTypeEnum dtype =
      torch_dtype_to_spyre_comms(input.scalar_type());

  spyre_comms::TensorShape shape({bcast_total_elems});
  spyre_comms::TensorInfo tensor_info(dtype, shape);

  // Copy input to output if we're the source rank
  int current_rank = context->getRank();
  if (current_rank == src_rank) {
    output.copy_(input);
  }

  // Create spyre_comms Tensor with device address
  spyre_comms::Tensor buffer_tensor(tensor_info);
  buffer_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  auto wsi = context->broadcast(
      tensor_info, static_cast<spyre_comms::process_id_t>(src_rank));
  TORCH_CHECK(wsi != nullptr, "Broadcast failed to create WorkScheduleInfo");
  auto work_schedule = context->broadcast_applyTensor(*wsi, buffer_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "Broadcast applyTensor failed to create WorkSchedule");

  work_schedule->start();  // Start but DON'T wait

  // Store WorkSchedule in map; hold_tensors keeps the allocation alive
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "broadcast_async called twice on the same allocation without "
                "intervening wait_work");
    pending_work_map_.emplace(ctx, PendingWork{std::move(work_schedule),
                                               CollectiveKind::Broadcast,
                                               {output}});
    DEBUGINFO("Stored PendingWork at ctx=", ctx,
              ", pending_work_map size=", pending_work_map_.size());
  }

  return output;  // Return immediately without waiting
}

// All_reduce implementation — operates in-place on the input buffer.
// Non-blocking: starts the reduction and returns immediately; the caller
// must use wait_work to block until the operation completes.
at::Tensor spyre_allreduce_async_impl(const at::Tensor& input,
                                      const std::string& reduce_op,
                                      const std::string& group_name) {
  DEBUGINFO("spyre::all_reduce_async called with reduce_op=", reduce_op,
            ", group=", group_name);

  // Get world context
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::getDefaultStreamRuntimeHandle());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }

  auto op_type = parse_reduce_op(reduce_op);

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for all_reduce");
  TORCH_CHECK(input.is_contiguous(),
              "Tensor must be contiguous for all_reduce");
  TORCH_CHECK(input.nbytes() > 0,
              "Tensor must have non-zero size for all_reduce");

  // Get SharedOwnerCtx for the input tensor
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      input.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for input tensor");

  SpyreTensorLayout stl = get_spyre_tensor_layout(input);
  uint64_t ar_nbytes = get_device_size_in_bytes(stl);
  int64_t ar_total_elems =
      static_cast<int64_t>(ar_nbytes / input.element_size());

  spyre_comms::TensorDataTypeEnum dtype =
      torch_dtype_to_spyre_comms(input.scalar_type());
  spyre_comms::TensorShape shape({ar_total_elems});
  spyre_comms::TensorInfo tensor_info(dtype, shape);

  // Create tensor with host pointer + device address.
  spyre_comms::Tensor inout_tensor(tensor_info,
                                   input.storage().data_ptr().get());
  inout_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  auto wsi = context->allreduce(tensor_info, op_type);
  TORCH_CHECK(wsi != nullptr, "Allreduce failed to create WorkScheduleInfo");
  auto work_schedule = context->allreduce_applyTensor(*wsi, inout_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "Allreduce applyTensor failed to create WorkSchedule");

  work_schedule->start();

  // Store WorkSchedule in map for later wait_work call
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "reduce_async called twice on the same "
                "allocation without intervening wait_work");
    pending_work_map_.emplace(
        ctx, PendingWork{
                 std::move(work_schedule), CollectiveKind::AllReduce, {input}});
    DEBUGINFO("Stored PendingWork for all_reduce at ctx=", ctx,
              ", pending_work_map size=", pending_work_map_.size());
  }

  return input;  // Return the same tensor (allreduce operates in-place)
}

// Reduce implementation — reduces tensor across all ranks to dst_rank.
// Operates in-place on the input buffer. Synchronous: device cannot overlap
// compute and comms.
at::Tensor spyre_reduce_async_impl(const at::Tensor& input, int64_t dst_rank,
                                   const std::string& reduce_op,
                                   const std::string& group_name) {
  DEBUGINFO("spyre::reduce_async called with dst_rank=", dst_rank,
            ", reduce_op=", reduce_op, ", group=", group_name);

  // Drain prior in-flight collectives — hardware cannot overlap.
  drain_pending_work();

  // Get world context
  auto context = spyre_comms::get_world_context();
  if (context == nullptr) {
    DEBUGINFO("Initializing spyre-comms library");
    spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
                                    spyre::getDefaultStreamRuntimeHandle());
    context = spyre_comms::get_world_context();
    TORCH_CHECK(context != nullptr, "Failed to get spyre-comms world context");
  }

  auto op_type = parse_reduce_op(reduce_op);

  TORCH_CHECK(input.is_privateuseone(),
              "Tensor must be on Spyre device for reduce");
  TORCH_CHECK(input.is_contiguous(), "Tensor must be contiguous for reduce");
  TORCH_CHECK(input.nbytes() > 0, "Tensor must have non-zero size for reduce");
  TORCH_CHECK(
      dst_rank >= 0 && dst_rank < static_cast<int64_t>(context->getSize()),
      "dst_rank out of range: ", dst_rank, " (world_size=", context->getSize(),
      ")");

  // Get SharedOwnerCtx for the input tensor
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      input.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "SharedOwnerCtx is null for input tensor");

  SpyreTensorLayout stl = get_spyre_tensor_layout(input);
  uint64_t ar_nbytes = get_device_size_in_bytes(stl);
  int64_t ar_total_elems =
      static_cast<int64_t>(ar_nbytes / input.element_size());

  spyre_comms::TensorDataTypeEnum dtype =
      torch_dtype_to_spyre_comms(input.scalar_type());
  spyre_comms::TensorShape shape({ar_total_elems});
  spyre_comms::TensorInfo tensor_info(dtype, shape);

  // Create tensor with host pointer + device address.
  spyre_comms::Tensor inout_tensor(tensor_info,
                                   input.storage().data_ptr().get());
  inout_tensor.SetSpyreDeviceAddressBorrowed(&ctx->composite_addr);

  // Use allreduce as the underlying primitive — the standalone reduce
  // primitive is not yet supported by the hardware.  Allreduce writes the
  // result to all ranks, so the subsequent broadcast_async in the
  // decomposition becomes a redundant copy but remains correct.
  auto wsi = context->reduce(tensor_info, op_type,
                             static_cast<spyre_comms::process_id_t>(dst_rank));
  TORCH_CHECK(wsi != nullptr, "Reduce failed to create WorkScheduleInfo");
  auto work_schedule = context->reduce_applyTensor(*wsi, inout_tensor);
  TORCH_CHECK(work_schedule != nullptr,
              "Reduce applyTensor failed to create WorkSchedule");

  work_schedule->start();

  // Store WorkSchedule in map for later wait_work call
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    TORCH_CHECK(pending_work_map_.find(ctx) == pending_work_map_.end(),
                "reduce_async called twice on the same "
                "allocation without intervening wait_work");
    pending_work_map_.emplace(
        ctx, PendingWork{
                 std::move(work_schedule), CollectiveKind::AllReduce, {input}});
    DEBUGINFO("Stored PendingWork for all_reduce at ctx=", ctx,
              ", pending_work_map size=", pending_work_map_.size());
  }

  return input;
}

// Wait for async operation to complete
at::Tensor spyre_wait_work_impl(const at::Tensor& tensor) {
  DEBUGINFO("spyre::wait_work called");

  // Get SharedOwnerCtx for map lookup
  auto* ctx = static_cast<spyre::SharedOwnerCtx*>(
      tensor.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "SharedOwnerCtx is null — is this tensor from broadcast_async?");

  // Extract WorkSchedule under lock, erase map entry, release lock, then wait
  std::shared_ptr<spyre_comms::WorkSchedule> work_to_wait;
  {
    std::lock_guard<std::mutex> lock(work_map_mutex_);
    auto it = pending_work_map_.find(ctx);
    TORCH_CHECK(it != pending_work_map_.end(),
                "No pending async work found for tensor. "
                "wait_work must be called on a tensor returned from "
                "broadcast_async or all_reduce_async.");

    work_to_wait = std::move(it->second.work);
    pending_work_map_.erase(it);
    DEBUGINFO("Extracted and erased PendingWork, map size=",
              pending_work_map_.size());
  }

  // Lock released — concurrent wait_work and broadcast_async can now proceed.
  // work may be null if drain_pending_work() already completed it.
  if (work_to_wait) {
    work_to_wait->wait();
    DEBUGINFO("WorkSchedule wait completed");
  }

  // Return the tensor with completed collective data (broadcast or allreduce)
  return tensor;
}

}  // namespace spyre

// Define the spyre namespace and operations
TORCH_LIBRARY(spyre, m) {
  m.def(
      "broadcast_async(Tensor input, int src_rank, str group_name) -> Tensor");
  m.def(
      "all_reduce_async(Tensor(a!) input, str reduce_op=\"sum\", "
      "str group_name=\"default\") -> Tensor(a)");
  m.def(
      "reduce_async(Tensor(a!) input, int dst_rank, str reduce_op=\"sum\", "
      "str group_name=\"default\") -> Tensor(a)");
  m.def("wait_work(Tensor(a!) tensor) -> Tensor(a)");

  // Compile-time plan ops — scalar-only, registered with impl directly
  // so they dispatch via CompositeImplicitAutograd (no tensor to key off).
  m.def("broadcast_plan(int dtype, int src_rank, str group_name) -> int",
        &spyre::spyre_broadcast_plan_impl);
  m.def("allreduce_plan(int dtype, str reduce_op, str group_name) -> int",
        &spyre::spyre_allreduce_plan_impl);
  m.def(
      "reduce_plan(int dtype, int dst_rank, str reduce_op, "
      "str group_name) -> int",
      &spyre::spyre_reduce_plan_impl);

  // Runtime run ops — bind cached WSI to a tensor and execute
  m.def(
      "broadcast_run(Tensor input, int plan_handle, int src_rank) "
      "-> Tensor");
  m.def("allreduce_run(Tensor(a!) input, int plan_handle) -> Tensor(a)");
  m.def(
      "reduce_run(Tensor(a!) input, int plan_handle, int dst_rank) "
      "-> Tensor(a)");
}

// Register the implementations with PyTorch's dispatcher
TORCH_LIBRARY_IMPL(spyre, PrivateUse1, m) {
  m.impl("broadcast_async", &spyre::spyre_broadcast_async_impl);
  m.impl("all_reduce_async", &spyre::spyre_allreduce_async_impl);
  m.impl("reduce_async", &spyre::spyre_reduce_async_impl);
  m.impl("wait_work", &spyre::spyre_wait_work_impl);

  m.impl("broadcast_run", &spyre::spyre_broadcast_run_impl);
  m.impl("allreduce_run", &spyre::spyre_allreduce_run_impl);
  m.impl("reduce_run", &spyre::spyre_reduce_run_impl);
}
