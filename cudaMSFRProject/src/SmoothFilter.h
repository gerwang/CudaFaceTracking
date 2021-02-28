#pragma once
#include <cuda_runtime_api.h>

#include "pcl\gpu\containers\device_array.h"

class SmoothFilter {
 public:
  SmoothFilter(const int size)
      : prev_1_id(0), prev_data_1(size), prev_data_2(size) {}

  std::vector<float2> &prev_data_1_() {
    if (prev_1_id == 0) {
      return prev_data_1;
    } else {
      return prev_data_2;
    }
  }

  std::vector<float2> &prev_data_2_() {
    if (prev_1_id == 1) {
      return prev_data_1;
    } else {
      return prev_data_2;
    }
  }

  void push_frame(std::vector<float2> &present_frame) {
    auto &prev_2 = prev_data_2_();
    for (auto i = 0; i < present_frame.size(); i++) {
      prev_2[i] = present_frame[i];
    }
    prev_1_id = 1 - prev_1_id;
    has_prev_2 = has_prev_1;
    has_prev_1 = true;
  }

  void push_empty() { has_prev_1 = false; }

  void smooth_data(std::vector<float2> &prediction, float lambda) {
    if (has_prev_1 && has_prev_2) {
      auto &prev_1 = prev_data_1_();
      auto &prev_2 = prev_data_2_();

      for (auto i = 0; i < prediction.size(); i++) {
        prediction[i].x =
            ((2 * prev_1[i].x - prev_2[i].x) * lambda + prediction[i].x) /
            (1 + lambda);
        prediction[i].y =
            ((2 * prev_1[i].y - prev_2[i].y) * lambda + prediction[i].y) /
            (1 + lambda);
      }
    }
    push_frame(prediction);
  }

 public:
  int prev_1_id = 0;
  bool has_prev_1, has_prev_2;
  std::vector<float2> prev_data_1, prev_data_2;
};