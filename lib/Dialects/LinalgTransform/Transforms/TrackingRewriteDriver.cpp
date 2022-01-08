//===-- TrackingRewriteDriver.cpp - Pattern rewriter keeping track of ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/TrackingRewriteDriver.h"
#include "TrackingListener.h"
#include "Transforms/ListenerGreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

LogicalResult mlir::applyPatternsTrackAndFoldGreedily(
    Operation *root, DenseMap<Value, SmallVector<Operation *, 4>> &trackedOperations,
    const FrozenRewritePatternSet &patterns, GreedyRewriteConfig config) {
  TrackingListener listener(trackedOperations);
  return applyPatternsAndFoldGreedily(root, patterns, config, &listener);
}
