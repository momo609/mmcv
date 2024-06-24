#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

constexpr int32_t N_IDX = 0;
constexpr int32_t BOX_DIM = 7;

void iou3d_boxes_overlap_bev_forward_npu(const int num_a, const Tensor boxes_a,
                                         const int num_b, const Tensor boxes_b,
                                         Tensor ans_overlap) {
  TORCH_CHECK((boxes_a.sizes()[1] == BOX_DIM),
              "Input boxes shape should be (N, 7)");
  TORCH_CHECK((boxes_b.sizes()[1] == BOX_DIM),
              "Input boxes shape should be (N, 7)");

  auto trans = false;
  auto is_clockwise = false;
  auto need_iou = false;
  EXEC_NPU_CMD(aclnnBoxesOverlapBev, boxes_a, boxes_b, trans, is_clockwise, need_iou, area_overlap);
}

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap);

REGISTER_NPU_IMPL(iou3d_boxes_overlap_bev_forward_impl, iou3d_boxes_overlap_bev_forward_npu);
