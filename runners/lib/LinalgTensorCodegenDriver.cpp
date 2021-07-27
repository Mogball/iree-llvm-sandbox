//===- LinalgTensorCodegenDriver.cpp - Linalg transformation driver--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct LinalgTensorCodegenDriverPass
    : public LinalgTensorCodegenDriverBase<LinalgTensorCodegenDriverPass> {
  LinalgTensorCodegenDriverPass() = default;
  LinalgTensorCodegenDriverPass(const LinalgTensorCodegenDriverPass &pass) {}

  /// Function pass entry point.
  void runOnOperation() override;

 private:
  void fuseAll(FuncOp funcOp);
  void runOpAnchoredStrategy(FuncOp funcOp);
  void runComprehensiveBufferization();
  void runVectorLowering();
  void runLowerToLLVM();
};
}  // namespace

void LinalgTensorCodegenDriverPass::runLowerToLLVM() {
  // Module lowering pipeline.
  PassManager pm(&getContext());
  OpPassManager &nestedFuncOpPM = pm.nest<FuncOp>();
  nestedFuncOpPM.addPass(createConvertVectorToSCFPass());
  nestedFuncOpPM.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createLowerToCFGPass());
  pm.addPass(createConvertLinalgToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(getOperation()))) return signalPassFailure();
}

/// Return the neutral elementas a new Value.
/// For now, just assume it is the zero of type.
/// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<ConstantOp>(op.getOwner()->getLoc(), t, b.getZeroAttr(t));
}

/// Collect all Linalg ops, they must all have tensor semantics.
/// For now this just fuses everything.
// TODO: finer control.
void LinalgTensorCodegenDriverPass::fuseAll(FuncOp funcOp) {
  SmallVector<LinalgOp> linalgOps;
  auto walkResult = funcOp.walk([&](LinalgOp op) {
    if (!op.hasTensorSemantics()) return WalkResult::interrupt();
    linalgOps.push_back(op);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) return signalPassFailure();

  linalg::Aliases aliases;
  LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
  OpBuilder builder(funcOp.getContext());
  LinalgTilingOptions tilingOptions;
  tilingOptions = tilingOptions.setTileSizes(tileSizes).setLoopType(
      LinalgTilingLoopType::Loops);
  Optional<TiledAndFusedLinalgOps> tileAndFuseOps =
      tileAndFuseLinalgOps(builder, linalgOps, dependenceGraph, tilingOptions);
  if (tileAndFuseOps)
    linalgOps.back().getOperation()->replaceAllUsesWith(
        tileAndFuseOps->fusedLoops.front());
}

void LinalgTensorCodegenDriverPass::runOpAnchoredStrategy(FuncOp funcOp) {
  if (anchorOpName.empty()) return;

  if (fuse) return fuseAll(funcOp);

  // Set up tiling and vectorization options.
  LinalgTilingOptions tilingOptions;
  if (!tileSizes.empty()) tilingOptions = tilingOptions.setTileSizes(tileSizes);
  if (pad)
    tilingOptions =
        tilingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  CodegenStrategy strategy;
  strategy.tileIf<LinalgOp>(!tileSizes.empty(), anchorOpName, tilingOptions)
      .vectorizeIf(vectorize, anchorOpName)
      .setEnableVectorContractLowering(false)
      .setEnableVectorToSCFConversion(false)
      .transform(funcOp);
}

void LinalgTensorCodegenDriverPass::runComprehensiveBufferization() {
  // Module-level bufferization enables later vector transforms.
  StringRef pipeline =
      "canonicalize,"
      "cse,"
      "linalg-comprehensive-module-bufferize";
  PassManager pm(&getContext());
  if (failed(parsePassPipeline(pipeline, pm)) || failed(pm.run(getOperation())))
    return signalPassFailure();
}

void LinalgTensorCodegenDriverPass::runVectorLowering() {
  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          vectorizeContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  // Per-function lowering pipeline.
  getOperation().walk([&](FuncOp funcOp) {
    CodegenStrategy strategy2;
    strategy2
        // Lowering of vector contractions.
        .setEnableVectorContractLowering(true)
        // Whether to split full/partial vector.transfer ops.
        .setEnableVectorTransferPartialRewrite(
            vectorTransferSplit != vector::VectorTransferSplit::None)
        .setVectorTransformsOptions(
            vector::VectorTransformsOptions()
                .setVectorTransformsOptions(vectorContractLowering)
                .setVectorTransferSplit(vectorTransferSplit))
        // Conversion to scf.
        .setEnableVectorToSCFConversion(true)
        .setVectorTransferToSCFOptions(VectorTransferToSCFOptions()
                                           .setUnroll(unrollVectorTransfers)
                                           .setLowerPermutationMaps(true))
        .transform(funcOp);
  });
}

void LinalgTensorCodegenDriverPass::runOnOperation() {
  if (!anchorFuncOpName.empty()) {
    getOperation().walk([&](FuncOp funcOp) {
      if (funcOp.getName() != anchorFuncOpName) return;

      // Run transforms that require anchoring on a particular op. This only
      // applies if !anchorOpName.empty().
      runOpAnchoredStrategy(funcOp);

      // Run other transforms that do not require a named linalg op.
      if (hoistPadding > 0) {
        SmallVector<PadTensorOp> ops;
        funcOp.walk([&](PadTensorOp op) { ops.push_back(op); });
        for (auto op : llvm::reverse(ops))
          (void)hoistPaddingOnTensors(op, hoistPadding);
      }
      if (vectorizePadding) {
        OwningRewritePatternList extraVectorizationPatterns(
            funcOp.getContext());
        populatePadTensorOpVectorizationPatterns(extraVectorizationPatterns);
        (void)applyPatternsAndFoldGreedily(
            funcOp, std::move(extraVectorizationPatterns));
      }

      // Perform other hoistings.
      CodegenStrategy strategy;
      strategy.setEnableLICM(true)
          .setEnableHoistRedundantVectorTransfersOnTensor(true)
          .setEnableVectorContractLowering(false)
          .setEnableVectorToSCFConversion(false)
          .transform(funcOp);
    });
  }

  if (bufferize) runComprehensiveBufferization();

  if (vectorLowering) runVectorLowering();

  if (llvmLowering) runLowerToLLVM();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinalgTensorCodegenDriverPass() {
  return std::make_unique<LinalgTensorCodegenDriverPass>();
}