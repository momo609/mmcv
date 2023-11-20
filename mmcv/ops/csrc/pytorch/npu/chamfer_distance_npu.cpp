
#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void chamfer_distance_forward_npu(Tensor XYZ1, Tensor XYZ2, Tensor dist1,
                                    Tensor dist2, Tensor idx1, Tensor idx2) {
  at::Tensor xyz1 = at::ones_like(XYZ1);
  at::Tensor xyz2 = at::ones_like(XYZ2);
  xyz1 = XYZ1.transpose(1,2);
  xyz2 = XYZ2.transpose(1,2);
  OpCommand cmd;
  cmd.Name("ChamferDistance")
      .Input(xyz1)
      .Input(xyz2)
      .Output(dist1)
      .Output(dist2)
      .Output(idx1)
      .Output(idx2)
      .Run();
}

void chamfer_distance_forward_impl(Tensor XYZ1, Tensor XYZ2, Tensor dist1,
                                    Tensor dist2, Tensor idx1, Tensor idx2);
REGISTER_NPU_IMPL(chamfer_distance_forward_impl,
                  chamfer_distance_forward_npu);
