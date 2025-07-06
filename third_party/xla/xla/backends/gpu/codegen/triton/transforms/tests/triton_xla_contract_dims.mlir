// RUN: xla-opt %s --triton-xla-contract-dims=convert-to-reshape=false \
// RUN: | FileCheck %s

// RUN: xla-opt %s --triton-xla-contract-dims \
// RUN: | FileCheck %s --check-prefix=RESHAPE

// CHECK-LABEL: func @push_contract_dims_up_through_elementwise
tt.func @push_contract_dims_up_through_elementwise(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK: arith.negf {{.*}} : tensor<4x8xf32>
  %0 = arith.negf %arg0 : tensor<4x1x8xf32>
  %1 = triton_xla.contract_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @push_contract_dims_up_through_broadcast
tt.func @push_contract_dims_up_through_broadcast(%arg0: tensor<1x4x1x8xf32>) -> tensor<4x16x8xf32> {
  // CHECK: tt.broadcast {{.*}} : tensor<4x1x8xf32> -> tensor<4x16x8xf32>
  %0 = tt.broadcast %arg0 : tensor<1x4x1x8xf32> -> tensor<1x4x16x8xf32>
  %1 = triton_xla.contract_dims %0 {axis = 0 : i32} : tensor<1x4x16x8xf32> -> tensor<4x16x8xf32>
  tt.return %1 : tensor<4x16x8xf32>
}

// CHECK-LABEL: func @push_contract_dims_up_through_trans
tt.func @push_contract_dims_up_through_trans(%arg0: tensor<4x1x8xf32>) -> tensor<8x4xf32> {
  // CHECK: tt.trans {{.*}} {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
  %0 = tt.trans %arg0 {order = array<i32: 2, 0, 1>} : tensor<4x1x8xf32> -> tensor<8x4x1xf32>
  %1 = triton_xla.contract_dims %0 {axis = 2 : i32} : tensor<8x4x1xf32> -> tensor<8x4xf32>
  tt.return %1 : tensor<8x4xf32>
}

// CHECK-LABEL: func @push_contract_dims_up_through_join
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x4xf32>, %[[ARG1:.+]]: tensor<1x4xf32>
tt.func @push_contract_dims_up_through_join(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<4x2xf32> {
  // CHECK-DAG: %[[LHS:.+]] = triton_xla.contract_dims %[[ARG0]] {axis = 0 : i32} : tensor<1x4xf32> -> tensor<4xf32>
  // CHECK-DAG: %[[RHS:.+]] = triton_xla.contract_dims %[[ARG1]] {axis = 0 : i32} : tensor<1x4xf32> -> tensor<4xf32>
  // CHECK-DAG: tt.join %[[LHS]], %[[RHS]] : tensor<4xf32> -> tensor<4x2xf32>
  %0 = tt.join %arg0, %arg1 : tensor<1x4xf32> -> tensor<1x4x2xf32>
  %1 = triton_xla.contract_dims %0 {axis = 0 : i32} : tensor<1x4x2xf32> -> tensor<4x2xf32>
  tt.return %1 : tensor<4x2xf32>
}

// CHECK-LABEL: func @push_contract_dims_up_through_reduce
// CHECK-SAME:    %[[ARG0:.+]]: tensor<8x4x1xf32>
tt.func @push_contract_dims_up_through_reduce(%arg0: tensor<8x4x1xf32>) -> tensor<8xf32> {
  // CHECK: %[[CONTRACT:.+]] = triton_xla.contract_dims %[[ARG0]] {axis = 2 : i32} :
  // CHECK-SAME: tensor<8x4x1xf32> -> tensor<8x4xf32>
  // CHECK: %[[REDUCE:.+]] = "tt.reduce"(%[[CONTRACT]]) <{axis = 1 : i32}> ({
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%arg1: f32, %arg2: f32):
    %1 = arith.addf %arg1, %arg2 : f32
    tt.reduce.return %1 : f32
  // CHECK: }) : (tensor<8x4xf32>) -> tensor<8xf32>
  }) : (tensor<8x4x1xf32>) -> tensor<8x1xf32>
  %2 = triton_xla.contract_dims %0 {axis = 1 : i32} : tensor<8x1xf32> -> tensor<8xf32>
  // CHECK: tt.return %[[REDUCE]] : tensor<8xf32>
  tt.return %2 : tensor<8xf32>
}

// CHECK-LABEL: func @fold_contract_of_expand_cancelling
// CHECK-SAME:    %[[ARG0:.+]]: tensor<4x8xf32>
tt.func @fold_contract_of_expand_cancelling(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: tt.return %[[ARG0]]
  %0 = tt.expand_dims %arg0 {axis = 1 : i32} : tensor<4x8xf32> -> tensor<4x1x8xf32>
  %1 = triton_xla.contract_dims %0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @fold_contract_of_expand_swapping
// CHECK-SAME:    %[[ARG0:.+]]: tensor<4x1x8xf32>
tt.func @fold_contract_of_expand_swapping(%arg0: tensor<4x1x8xf32>) -> tensor<1x4x8xf32> {
  // CHECK: %[[CONTRACT:.+]] = triton_xla.contract_dims %[[ARG0]] {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK: tt.expand_dims %[[CONTRACT]] {axis = 0 : i32} : tensor<4x8xf32> -> tensor<1x4x8xf32>
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<4x1x8xf32> -> tensor<1x4x1x8xf32>
  %1 = triton_xla.contract_dims %0 {axis = 2 : i32} : tensor<1x4x1x8xf32> -> tensor<1x4x8xf32>
  tt.return %1 : tensor<1x4x8xf32>
}

// CHECK-LABEL: func @reshape_to_contract
// CHECK-SAME:    %[[ARG0:.+]]: tensor<4x1x8xf32>
tt.func @reshape_to_contract(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[CONTRACT:.+]] = triton_xla.contract_dims %[[ARG0]] {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  %0 = tt.reshape %arg0 : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK: tt.return %[[CONTRACT]]
  tt.return %0 : tensor<4x8xf32>
}

#arg_enc = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#res_enc = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
// CHECK-LABEL: func @reshape_with_encoding
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<1x32xf32, #[[ARG_ENC:.+]]>)
// CHECK-SAME:    -> tensor<32xf32, #[[RES_ENC:.+]]>
tt.func @reshape_with_encoding(%arg0: tensor<1x32xf32, #arg_enc>) -> tensor<32xf32, #res_enc> {
  // CHECK: %[[CONTRACT:.+]] = triton_xla.contract_dims %[[ARG0]] {axis = 0 : i32} :
  // CHECK-SAME: tensor<1x32xf32, #[[ARG_ENC]]> -> tensor<32xf32, #[[RES_ENC]]>
  %0 = tt.reshape %arg0 : tensor<1x32xf32, #arg_enc> -> tensor<32xf32, #res_enc>
  // CHECK: tt.return %[[CONTRACT]] : tensor<32xf32, #[[RES_ENC]]>
  tt.return %0 : tensor<32xf32, #res_enc>
}
}

// CHECK-LABEL: func @fold_contract_dims_of_load
tt.func @fold_contract_dims_of_load(%arg0: !tt.ptr<f32>) -> tensor<4x8xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  // CHECK: %[[PTR:.+]] = tt.make_tensor_ptr {{.*}} {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c1_i64, %c8_i64], [%c8_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x1x8xf32>>
  // CHECK: %[[LOAD:.+]] = tt.load %[[PTR]] {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<4x8xf32>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 2>, padding = 1 : i32} : !tt.ptr<tensor<4x1x8xf32>>
  // CHECK-NOT: triton_xla.contract_dims
  %2 = triton_xla.contract_dims %1 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  // CHECK: tt.return %[[LOAD]]
  tt.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @contract_store
// CHECK-SAME:    %[[ARG0:.+]]: !tt.ptr<f32>
// CHECK-SAME:    %[[ARG1:.+]]: tensor<4x1x8xf32>
tt.func @contract_store(%arg0: !tt.ptr<f32>, %arg1: tensor<4x1x8xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  // CHECK-DAG: %[[PTR:.+]] = tt.make_tensor_ptr %[[ARG0]], {{.*}} {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
  // CHECK-DAG: %[[CONTRACT:.+]] = triton_xla.contract_dims %[[ARG1]] {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  %0 = tt.make_tensor_ptr %arg0, [%c4_i64, %c1_i64, %c8_i64], [%c8_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<4x1x8xf32>>
  // CHECK: tt.store %[[PTR]], %[[CONTRACT]] {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<4x8xf32>>
  tt.store %0, %arg1 {boundaryCheck = array<i32: 2>} : !tt.ptr<tensor<4x1x8xf32>>
  tt.return
}

// CHECK-LABEL: func @contract_store_unit_tensor
tt.func @contract_store_unit_tensor(%arg0: !tt.ptr<f32>, %arg1: tensor<1x1xf32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  // CHECK-DAG: %[[PTR:.+]] = tt.make_tensor_ptr
  %0 = tt.make_tensor_ptr %arg0, [%c1_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 0>} : <tensor<1x1xf32>>
  // CHECK-DAG: %[[CONTRACT:.+]] = triton_xla.contract_dims
  // CHECK: tt.store %[[PTR]], %[[CONTRACT]] : !tt.ptr<tensor<1xf32>>
  tt.store %0, %arg1 : !tt.ptr<tensor<1x1xf32>>
  tt.return
}


// RESHAPE-LABEL: func @contract_dims_to_reshape
// RESHAPE-SAME:    %[[ARG0:.+]]: tensor<4x1x8xf32>
tt.func @contract_dims_to_reshape(%arg0: tensor<4x1x8xf32>) -> tensor<4x8xf32> {
  // RESHAPE: tt.reshape %[[ARG0]] : tensor<4x1x8xf32> -> tensor<4x8xf32>
  %0 = triton_xla.contract_dims %arg0 {axis = 1 : i32} : tensor<4x1x8xf32> -> tensor<4x8xf32>
  tt.return %0 : tensor<4x8xf32>
}

