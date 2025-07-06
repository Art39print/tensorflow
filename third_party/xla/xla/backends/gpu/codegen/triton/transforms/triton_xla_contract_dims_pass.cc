/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLACONTRACTDIMSPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Returns list of dimensions with size equal to 1. If they all dimensions have
// size 1, do not return the last dimension to avoid tripping up Triton.
SmallVector<uint32_t> FindUnitDimensions(RankedTensorType type) {
  SmallVector<uint32_t> result;
  for (auto [dim, size] : llvm::enumerate(type.getShape())) {
    if (size == 1) {
      result.push_back(dim);
    }
  }
  if (result.size() == type.getRank()) {
    result.pop_back();  // Keep one unit dimension.
  }
  return result;
}

// Returns a new container with the given dimensions removed.
template <typename ContainerT>
auto GetContractedValues(ContainerT values, ArrayRef<uint32_t> contract_dims) {
  auto it = values.begin();
  SmallVector<typename std::iterator_traits<decltype(it)>::value_type> result;
  for (uint32_t dim : contract_dims) {
    auto end = values.begin() + dim;
    std::copy(it, end, std::back_inserter(result));
    it = std::next(end);
  }
  std::copy(it, values.end(), std::back_inserter(result));
  return result;
}

// Returns a new boundary check with the given dimensions removed.
SmallVector<int32_t> GetContractedBoundaryCheck(
    ArrayRef<int32_t> boundary_check, ArrayRef<uint32_t> contract_dims) {
  CHECK(absl::c_is_sorted(boundary_check));
  CHECK(absl::c_is_sorted(contract_dims));
  SmallVector<int32_t> result;
  auto it = contract_dims.begin();
  for (int32_t dim : boundary_check) {
    it = std::lower_bound(it, contract_dims.end(), dim);
    if (it != contract_dims.end() && *it == dim) {
      continue;
    }
    result.push_back(dim - (it - contract_dims.begin()));
  }
  return result;
}

// Returns a new tensor type with the given dimensions removed.
RankedTensorType GetContractedTensorType(RankedTensorType type,
                                         ArrayRef<uint32_t> contract_dims) {
  auto shape = GetContractedValues(type.getShape(), contract_dims);
  Attribute encoding = type.getEncoding();
  if (encoding) {
    auto inferLayoutInterface =
        cast<DialectInferLayoutInterface>(&encoding.getDialect());
    [[maybe_unused]] auto result = inferLayoutInterface->inferReshapeOpEncoding(
        type.getShape(), encoding, shape, encoding, std::nullopt);
    CHECK(succeeded(result));
  }
  return RankedTensorType::get(shape, type.getElementType(), encoding);
}

// Rewrites tt.make_tensor_ptr with unit dimensions. Returns the
// new MakeTensorPtrOp result and the dimensions that were removed.
Value ContractMakeTensorPtr(PatternRewriter& rewriter, MakeTensorPtrOp op,
                            ArrayRef<uint32_t> contract_dims) {
  auto tensor_type = cast<RankedTensorType>(op.getType().getPointeeType());
  auto contracted_type = GetContractedTensorType(tensor_type, contract_dims);
  auto ptr_type =
      PointerType::get(contracted_type, op.getType().getAddressSpace());

  // Strides already encode the layout, so we can use the default order.
  // Note that the order attribute is ignored in the Triton lowering.
  SmallVector<int32_t> order(contracted_type.getShape().size());
  std::iota(order.rbegin(), order.rend(), 0);

  Value result = rewriter.create<MakeTensorPtrOp>(
      op.getLoc(), ptr_type, op.getBase(),
      GetContractedValues(op.getShape(), contract_dims),
      GetContractedValues(op.getStrides(), contract_dims),
      GetContractedValues(op.getOffsets(), contract_dims), order);
  return result;
}

// Folds contract_dims into tt.load(tt.make_tensor_ptr).
// TODO(csigg): Add support for tt.load(tt.make_tensor_descriptor).
LogicalResult FoldContractDimsOfLoad(ContractDimsOp op,
                                     PatternRewriter& rewriter) {
  auto load = op.getSrc().getDefiningOp<LoadOp>();
  if (!load) {
    return rewriter.notifyMatchFailure(op, "Expected load producer.");
  }
  auto make_tensor_ptr = load.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_tensor_ptr) {
    return rewriter.notifyMatchFailure(
        load, "Expected ptr to be defined by make_tensor_ptr.");
  }

  Value pointer =
      ContractMakeTensorPtr(rewriter, make_tensor_ptr, op.getAxis());

  rewriter.replaceOpWithNewOp<LoadOp>(
      op, pointer,
      GetContractedBoundaryCheck(load.getBoundaryCheck(), op.getAxis()),
      load.getPadding(), load.getCache(), load.getEvict(),
      load.getIsVolatile());
  return success();
}

// Extracts unit dimensions from tt.store and prepends them as contract_dims.
LogicalResult StoreExtractContractDims(StoreOp op, PatternRewriter& rewriter) {
  auto make_tensor_ptr = op.getPtr().getDefiningOp<MakeTensorPtrOp>();
  if (!make_tensor_ptr) {
    return rewriter.notifyMatchFailure(
        op, "Expected ptr to be defined by make_tensor_ptr.");
  }
  auto tensor_type = dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!tensor_type || tensor_type.getRank() == 0) {
    return rewriter.notifyMatchFailure(op, "Expected tensor type.");
  }

  auto contract_dims = FindUnitDimensions(tensor_type);
  if (contract_dims.empty()) {
    return rewriter.notifyMatchFailure(op, "No unit dimensions.");
  }

  Value pointer =
      ContractMakeTensorPtr(rewriter, make_tensor_ptr, contract_dims);

  Value value = op.getValue();
  for (uint32_t dim : llvm::reverse(contract_dims)) {
    Type contracted_type =
        GetContractedTensorType(cast<RankedTensorType>(value.getType()), dim);
    value = rewriter.create<ContractDimsOp>(op.getLoc(), contracted_type, value,
                                            dim);
  }

  rewriter.replaceOpWithNewOp<StoreOp>(
      op, pointer, value,
      GetContractedBoundaryCheck(op.getBoundaryCheck(), contract_dims),
      op.getCache(), op.getEvict());
  return success();
}

// Extracts unit dimensions from tt.reshape and prepends them as contract_dims.
LogicalResult ReshapeExtractContractDims(ReshapeOp op,
                                         PatternRewriter& rewriter) {
  auto contract_dims = FindUnitDimensions(op.getSrc().getType());
  if (contract_dims.empty()) {
    return rewriter.notifyMatchFailure(op, "No unit dimensions.");
  }

  auto value = op.getSrc();
  // Contract from highest index to lowest to keep indices valid.
  for (int axis : llvm::reverse(contract_dims)) {
    Type contracted_type =
        GetContractedTensorType(cast<RankedTensorType>(value.getType()), axis);
    value = rewriter.create<ContractDimsOp>(op.getLoc(), contracted_type, value,
                                            axis);
  }
  op.setOperand(value);
  return success();
}

// Pushes tt.contract_dims up through element-wise operations.
// Example:
//   %1 = elementwise-op %0
//   %2 = tt.contract_dims %1
// is rewritten to:
//   %1 = tt.contract_dims %0
//   %2 = elementwise-op %1
LogicalResult PushContractDimsUpThroughElementwise(ContractDimsOp op,
                                                   PatternRewriter& rewriter) {
  Operation* producer = op.getSrc().getDefiningOp();
  if (!producer || !producer->hasTrait<OpTrait::Elementwise>() ||
      producer->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "Expected elementwise producer.");
  }

  auto result_type = op.getType();
  auto source_type = op.getSrc().getType();

  SmallVector<Value> new_operands;
  new_operands.reserve(producer->getOperands().size());
  for (Value operand : producer->getOperands()) {
    if (auto operand_type = dyn_cast<RankedTensorType>(operand.getType())) {
      CHECK(operand_type.getShape() == source_type.getShape());
      Type contracted_type =
          GetContractedTensorType(operand_type, op.getAxis());
      new_operands.push_back(rewriter.create<ContractDimsOp>(
          producer->getLoc(), contracted_type, operand, op.getAxis()));
    } else {
      new_operands.push_back(operand);
    }
  }

  OperationState new_producer_state(producer->getLoc(), producer->getName());
  new_producer_state.addOperands(new_operands);
  new_producer_state.addAttributes(producer->getAttrs());
  new_producer_state.addTypes(result_type);
  Operation* new_producer = rewriter.create(new_producer_state);
  rewriter.replaceOp(op, new_producer->getResult(0));
  return success();
}

// Pushes tt.contract_dims up through tt.broadcast.
// Example:
//   %1 = tt.broadcast %0
//   %2 = tt.contract_dims %1
// is rewritten to:
//   %1 = tt.contract_dims %0
//   %2 = tt.broadcast %1
LogicalResult PushContractDimsUpThroughBroadcast(ContractDimsOp op,
                                                 PatternRewriter& rewriter) {
  auto broadcast = op.getSrc().getDefiningOp<BroadcastOp>();
  if (!broadcast) {
    return rewriter.notifyMatchFailure(op, "Expected broadcast producer.");
  }

  Type contracted_type =
      GetContractedTensorType(broadcast.getSrc().getType(), op.getAxis());
  Value contract_dims = rewriter.create<ContractDimsOp>(
      op.getLoc(), contracted_type, broadcast.getSrc(), op.getAxis());
  rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), contract_dims);
  return success();
}

// Pushes tt.contract_dims up through tt.trans.
// Example:
//   %1 = tt.trans %0, perm = [1, 0]
//   %2 = tt.contract_dims %1, axis = 0
// is rewritten to:
//   %1 = tt.contract_dims %0, axis = 1
//   %2 = tt.trans %1, perm = [0]
LogicalResult PushContractDimsUpThroughTrans(ContractDimsOp op,
                                             PatternRewriter& rewriter) {
  auto trans = op.getSrc().getDefiningOp<TransOp>();
  if (!trans) {
    return rewriter.notifyMatchFailure(op, "Expected trans producer.");
  }

  // The axis to contract in the source of the transpose.
  auto order = trans.getOrder();
  uint32_t dst_axis = op.getAxis();
  uint32_t src_axis = order[dst_axis];

  // Contract the dimension in the source of the transpose.
  Type contracted_type =
      GetContractedTensorType(trans.getSrc().getType(), src_axis);
  Value contract_dims = rewriter.create<ContractDimsOp>(
      op.getLoc(), contracted_type, trans.getSrc(), src_axis);

  // Compute the new permutation for the transpose.
  auto new_order = GetContractedValues(order, dst_axis);
  for (auto& dim : new_order) {
    dim -= dim > src_axis;
  }
  rewriter.replaceOpWithNewOp<TransOp>(op, contract_dims, new_order);
  return success();
}

// Pushes tt.contract_dims up through tt.join.
// Example:
//   %2 = tt.join %0, %1
//   %3 = tt.contract_dims %2
// is rewritten to:
//   %2 = tt.contract_dims %0
//   %3 = tt.contract_dims %1
//   %4 = tt.join %2, %3
LogicalResult PushContractDimsUpThroughJoin(ContractDimsOp op,
                                            PatternRewriter& rewriter) {
  auto join = op.getSrc().getDefiningOp<JoinOp>();
  if (!join) {
    return rewriter.notifyMatchFailure(op, "Expected join producer.");
  }

  SmallVector<Value> operands;
  operands.reserve(join.getOperands().size());
  for (Value operand : join.getOperands()) {
    auto operand_type = cast<RankedTensorType>(operand.getType());
    Type contracted_type = GetContractedTensorType(operand_type, op.getAxis());
    operands.push_back(rewriter.create<ContractDimsOp>(
        op.getLoc(), contracted_type, operand, op.getAxis()));
  }

  rewriter.replaceOpWithNewOp<JoinOp>(op, op.getType(), operands);
  return success();
}

// Pushes tt.contract_dims up through tt.reduce.
// Example:
//   %1 = tt.reduce(%0) axis=1
//   %2 = tt.contract_dims %1, axis=0
// is rewritten to:
//   %1 = tt.contract_dims %0, axis=0
//   %2 = tt.reduce(%1) axis=0
LogicalResult PushContractDimsUpThroughReduce(ContractDimsOp op,
                                              PatternRewriter& rewriter) {
  auto reduce = op.getSrc().getDefiningOp<ReduceOp>();
  if (!reduce || reduce.getNumResults() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Expected single-result reduce producer.");
  }

  uint32_t contract_axis = op.getAxis() + (op.getAxis() >= reduce.getAxis());
  uint32_t reduce_axis = reduce.getAxis() - (op.getAxis() < reduce.getAxis());

  SmallVector<Value> operands;
  operands.reserve(reduce.getOperands().size());
  for (Value operand : reduce.getOperands()) {
    auto operand_type = cast<RankedTensorType>(operand.getType());
    Type contracted_type = GetContractedTensorType(operand_type, contract_axis);
    operands.push_back(rewriter.create<ContractDimsOp>(
        op.getLoc(), contracted_type, operand, contract_axis));
  }

  auto new_op = rewriter.replaceOpWithNewOp<ReduceOp>(op, op.getType(),
                                                      operands, reduce_axis);
  rewriter.cloneRegionBefore(reduce->getRegion(0), new_op->getRegion(0),
                             new_op->getRegion(0).begin());
  return success();
}

// Pushes tt.contract_dims up through tt.expand_dims, or folds them.
// Example:
//   %1 = tt.expand_dims %0, axis=1
//   %2 = tt.contract_dims %1, axis=0
// is rewritten to:
//   %1 = tt.contract_dims %0, axis=0
//   %2 = tt.expand_dims %1, axis=0
LogicalResult PushContractDimsUpThroughExpandDims(ContractDimsOp op,
                                                  PatternRewriter& rewriter) {
  auto expand_dims = op.getSrc().getDefiningOp<ExpandDimsOp>();
  if (!expand_dims) {
    return rewriter.notifyMatchFailure(op, "Expected expand_dims producer.");
  }

  if (expand_dims.getSrc().getType() == op.getType()) {
    rewriter.replaceOp(op, expand_dims.getSrc());
    return success();
  }

  // Swap: contract_dims(expand_dims) -> expand_dims(contract_dims)
  uint32_t expand_axis = expand_dims.getAxis();
  uint32_t contract_axis = op.getAxis();
  CHECK_NE(expand_axis, contract_axis);

  contract_axis -= contract_axis > expand_axis;
  expand_axis -= expand_axis > contract_axis;

  Type contracted_type =
      GetContractedTensorType(expand_dims.getSrc().getType(), contract_axis);
  Value contract_dims = rewriter.create<ContractDimsOp>(
      op.getLoc(), contracted_type, expand_dims.getSrc(), contract_axis);
  rewriter.replaceOpWithNewOp<ExpandDimsOp>(op, op.getType(), contract_dims,
                                            expand_axis);
  return success();
}

// Converts tt.contract_dims to tt.reshape.
LogicalResult ContractDimsToReshape(ContractDimsOp op,
                                    PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getSrc());
  return success();
}

// This pass removes unit dimensions from tensors to generate faster code
// through Triton.
//
// - contract_dims ops are extracted from tt.store and tt.reshape with unit
//   dimensions.
// - A series of patterns push contract_dims up through the graph, past
//   element-wise ops, broadcast, transpose, join, reduce.
// - contract_dims are folded into load and expand_dims, or converted back to
//   reshape.
class TritonXLAContractDimsPass
    : public impl::TritonXLAContractDimsPassBase<TritonXLAContractDimsPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(FoldContractDimsOfLoad);
    patterns.add(StoreExtractContractDims);
    patterns.add(ReshapeExtractContractDims);
    patterns.add(PushContractDimsUpThroughElementwise);
    patterns.add(PushContractDimsUpThroughBroadcast);
    patterns.add(PushContractDimsUpThroughTrans);
    patterns.add(PushContractDimsUpThroughJoin);
    patterns.add(PushContractDimsUpThroughReduce);
    patterns.add(PushContractDimsUpThroughExpandDims);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }

    if (!convert_to_reshape_) {
      return;
    }

    RewritePatternSet finalization_patterns(&getContext());
    finalization_patterns.add(ContractDimsToReshape);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(finalization_patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAContractDimsPass() {
  return std::make_unique<TritonXLAContractDimsPass>();
}

}  // namespace mlir::triton::xla
