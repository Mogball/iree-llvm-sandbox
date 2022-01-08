//===- TestListenerPasses.cpp - Test passes with listeners ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transforms/Listener.h"
#include "Transforms/ListenerCSE.h"
#include "Transforms/ListenerGreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// The test listener prints stuff to `stdout` so that it can be checked by lit
/// tests.
struct TestListener : public RewriteListener {
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override {
    llvm::outs() << "REPLACED " << op->getName() << "\n";
  }
  void notifyOperationRemoved(Operation *op) override {
    llvm::outs() << "REMOVED " << op->getName() << "\n";
  }
};

struct TestCanonicalizePass : public PassWrapper<TestCanonicalizePass, Pass> {
  TestCanonicalizePass() = default;
  TestCanonicalizePass(const TestCanonicalizePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-canonicalize"; }
  StringRef getDescription() const final { return "Test canonicalize pass."; }

  void runOnOperation() override {
    TestListener listener;
    RewriteListener *listenerToUse = nullptr;
    if (withListener)
      listenerToUse = &listener;

    RewritePatternSet patterns(&getContext());
    for (Dialect *dialect : getContext().getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : getContext().getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            GreedyRewriteConfig(),
                                            listenerToUse)))
      signalPassFailure();
  }

  Pass::Option<bool> withListener{
      *this, "listener", llvm::cl::desc("Whether to run with a test listener"),
      llvm::cl::init(false)};
};

struct TestCSEPass : public PassWrapper<TestCSEPass, Pass> {
  TestCSEPass() = default;
  TestCSEPass(const TestCSEPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "test-cse"; }
  StringRef getDescription() const final { return "Test CSE pass."; }

  void runOnOperation() override {
    TestListener listener;
    RewriteListener *listenerToUse = nullptr;
    if (withListener)
      listenerToUse = &listener;

    if (failed(eliminateCommonSubexpressions(getOperation(),
                                             /*domInfo=*/nullptr,
                                             listenerToUse)))
      signalPassFailure();
  }

  Pass::Option<bool> withListener{
      *this, "listener", llvm::cl::desc("Whether to run with a test listener"),
      llvm::cl::init(false)};
};

} // namespace

namespace mlir {
namespace test_ext {
void registerTestListenerPasses() {
  PassRegistration<TestCanonicalizePass>();
  PassRegistration<TestCSEPass>();
}
} // namespace test_ext
} // namespace mlir
