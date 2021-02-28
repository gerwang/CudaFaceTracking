#pragma once
#include "Eigen/Eigen"
#include <glog/logging.h>

//template<typename T>
//inline void clearCudaMem(pcl::gpu::DeviceArray<T> mem) {
//  cudaSafeCall(cudaMemset(mem.ptr(), 0,
//    mem.size() * mem.elem_size));
//}
//
//template<typename T>
//inline void clearCudaMem(pcl::gpu::DeviceArray2D<T> mem) {
//  cudaSafeCall(cudaMemset(mem.ptr(), 0,
//    mem.rows() * mem.cols() * mem.elem_size));
//}


class HostUtil
{
public:
  static float interp(float v, float y0, float x0, float y1, float x1)
  {
    if (v<x0)
      return y0;
    else if (v>x1)
      return y1;
    else
      return (v - x0)*(y1 - y0) / (x1 - x0) + y0;
  }


  static float jet_base(float v)
  {
    if (v <= -0.75f)
      return 0.0f;
    else if (v <= -0.25f)
      return interp(v, 0.0, -0.75, 1.0, -0.25);
    else if (v <= 0.25f)
      return 1.0;
    else if (v <= 0.75f)
      return interp(v, 1.0, 0.25, 0.0, 0.75);
    else
      return 0.0f;
  }

  static Eigen::Vector3f jet_colormap(const float v)
  {
    float r = HostUtil::jet_base(v*2.0f - 1.5f);
    float g = HostUtil::jet_base(v*2.0f - 1.0f);
    float b = HostUtil::jet_base(v*2.0f - 0.5f);
    return Eigen::Vector3f(r, g, b);
  }
};



