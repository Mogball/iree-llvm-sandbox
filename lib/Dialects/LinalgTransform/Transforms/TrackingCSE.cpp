//===-- TrackingCSE.cpp - Common subexpr elimination keeping track of ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/TrackingCSE.h"
#include "TrackingListener.h"
#include "Transforms/Listener.h"
#include "Transforms/ListenerCSE.h"

using namespace mlir;
using namespace mlir::linalg;

void mlir::eliminateCommonSubexpressionsWithTrackedOps(
    Operation *root, DenseMap<Value, SmallVector<Operation *, 4>> &trackedOps,
    DominanceInfo *domInfo) {
  TrackingListener listener(trackedOps);
  (void)eliminateCommonSubexpressions(root, domInfo, &listener);
}
