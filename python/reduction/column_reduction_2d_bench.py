# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'column_reduction_2d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################


class ExperimentalSplitAndFuseFillOpExpert(TransformationList):

  def __init__(self, transforms, **kwargs):
    t = transforms + [
        # TODO: bufferization before vectorization triggers
        # a bufferization issue atm.
        # This is inefficient wrt VTW/VTR forwarding.
        #Bufferize()
    ] + StagedLowerVectorsTransformationList() + [LowerToLLVM()]
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


def all_experts(problem_sizes: List[int]):
  return [
      SingleTilingExpert(
          fun_name=fun_name,
          op_name=op_name,
          # Little trick avoids tiling small dimensions and otherwise tile by 128.
          sizes=[4, 128] if problem_sizes[1] > 256 else [4],
          interchange=[],
          peel=[],
          pad=False,
          pack_paddings=[],
          hoist_paddings=[],
          # kwargs passed down to LowerVectors.
          # TODO: better composition of experts.
          multi_reduction_lowering='innerparallel',
          print_ir_after_all=False),
  ]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'K']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 100
  problem_size_list = [
      [128, 256],
      [104, 128],
      [256, 256],
      [1000, 1024],
      [8000, 6144],
  ]

  def numpy_kernel(args, sizes, types):
    A, B = args
    B.fill(0.)
    np.sum(A, axis=0, out=B)

  def pytorch_kernel(args, sizes, types):
    A, B = args
    B.fill_(0.)
    torch.sum(A, dim=0, out=B)

  for problem_sizes in problem_size_list:
    test_harness(
        lambda s, t: ColumnReduction2DProblem(), [[np.float32] * 2],
        map(make_size_list, [problem_sizes]),
        all_experts(problem_sizes),
        n_iters=n_iters,
        function_name=fun_name)


if __name__ == '__main__':
  main()
