#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl\gpu\containers\device_array.h>
#include <pcl\gpu\containers\kernel_containers.h>
#include <sm_30_intrinsics.h>

#include <pcl\gpu\utils\safe_call.hpp>
#include <pcl\gpu\utils\cutil_math.h>

#include "Common.h"

__device__ __forceinline__ bool is_nostril(const int index) {
  if (index >= 6893 && index <= 7396) {
    int r = (index - 6893) % 120;
    if (r >= 0 && r <= 3) return true;
  }
  if (index >= 8573 && index <= 9056) {
    int r = (index - 6893) % 120;
    if (r >= 0 && r <= 3) return true;
  }
  return false;
}


__device__ __forceinline__ float Angle(float3 &v0, float3 &v1) {
  float w = length(v0) * length(v1);
  if (w == 0) return -1;
  float t = dot(v0, v1) / w;
  if (t > 1)
    t = 1;
  else if (t < -1)
    t = -1;
  return acosf(t);
}

__device__ __forceinline__ bool is_nostril_inner(const int index) {
  if (index == 8934 || index == 8814 || index == 8694) {
    return true;
  }
  if (index == 7254 || index == 7134 || index == 7014) {
    return true;
  }
  return false;
}

#ifdef __CUDACC__
__device__ __forceinline__ float warp_scan(float data) {
  data += __shfl_up(data, 1);
  data += __shfl_up(data, 2);
  data += __shfl_up(data, 4);
  data += __shfl_up(data, 8);
  data += __shfl_up(data, 16);
  return data;
}

__device__ __forceinline__ int inwarp_scan(int data) {
  int lane_id = threadIdx.x & 31;
  if (lane_id >= 1) {
    data += __shfl_up(data, 1);
  }
  if (lane_id >= 2) {
    data += __shfl_up(data, 2);
  }
  if (lane_id >= 4) {
    data += __shfl_up(data, 4);
  }
  if (lane_id >= 8) {
    data += __shfl_up(data, 8);
  }
  if (lane_id >= 16) {
    data += __shfl_up(data, 16);
  }
  return data;
}


__device__ __forceinline__ int shfl_add(int x, int offset) {
  int result = 0;

  asm volatile(
      "{.reg .s32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0|p, %1, %2, 0;"
      "@p add.s32 r0, r0, %1;"
      "mov.s32 %0, r0;}"
      : "=r"(result)
      : "r"(x), "r"(offset));

  return result;
}

__device__ __forceinline__ int warp_scan(int data) {
  data = shfl_add(data, 1);
  data = shfl_add(data, 2);
  data = shfl_add(data, 4);
  data = shfl_add(data, 8);
  data = shfl_add(data, 16);
  return data;
}

__device__ __forceinline__ void block_sum(int &x, int share_partial_sum[]) {
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  __syncthreads();
  x = warp_scan(x);
  if (lane_id == 31) {
    share_partial_sum[warp_id] = x;
  }
  __syncthreads();
  if (warp_id == 0) {
    x = share_partial_sum[lane_id];
    x = warp_scan(x);
  }
}

__device__ __forceinline__ void block_sum(float &x, float share_partial_sum[]) {
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  __syncthreads();
  x = warp_scan(x);
  if (lane_id == 31) {
    share_partial_sum[warp_id] = x;
  }
  __syncthreads();
  if (warp_id == 0) {
    x = share_partial_sum[lane_id];
    x = warp_scan(x);
  }
}


__device__ __forceinline__ float3 M33TxV3(const pcl::gpu::PtrSz<float> M,
                                          const float3 &v) {
  float x, y, z;
  x = M[0] * v.x + M[3] * v.y + M[6] * v.z;
  y = M[1] * v.x + M[4] * v.y + M[7] * v.z;
  z = M[2] * v.x + M[5] * v.y + M[8] * v.z;
  return make_float3(x, y, z);
}

__device__ __forceinline__ float3 M33TxV3(const float M[9], const float3 &v) {
  float x, y, z;
  x = M[0] * v.x + M[3] * v.y + M[6] * v.z;
  y = M[1] * v.x + M[4] * v.y + M[7] * v.z;
  z = M[2] * v.x + M[5] * v.y + M[8] * v.z;
  return make_float3(x, y, z);
}

__device__ __forceinline__ float3 M33xV3(const pcl::gpu::PtrSz<float> M,
                                         const float3 &v) {
  float x, y, z;
  x = M[0] * v.x + M[1] * v.y + M[2] * v.z;
  y = M[3] * v.x + M[4] * v.y + M[5] * v.z;
  z = M[6] * v.x + M[7] * v.y + M[8] * v.z;
  return make_float3(x, y, z);
}

__device__ __forceinline__ float3 M33xV3(const float M[9], const float3 &v) {
  float x, y, z;
  x = M[0] * v.x + M[1] * v.y + M[2] * v.z;
  y = M[3] * v.x + M[4] * v.y + M[5] * v.z;
  z = M[6] * v.x + M[7] * v.y + M[8] * v.z;
  return make_float3(x, y, z);
}

__device__ __forceinline__ void warp_max(int &value, float &key) {
  // const int lane_id = threadIdx.x & 31;
  float up_key;
  int up_value;

  up_key = __shfl_down(key, 16);
  up_value = __shfl_down(value, 16);
  if (key < up_key) {
    value = up_value;
    key = up_key;
  }
  up_key = __shfl_down(key, 8);
  up_value = __shfl_down(value, 8);
  if (key < up_key) {
    value = up_value;
    key = up_key;
  }
  up_key = __shfl_down(key, 4);
  up_value = __shfl_down(value, 4);
  if (key < up_key) {
    value = up_value;
    key = up_key;
  }
  up_key = __shfl_down(key, 2);
  up_value = __shfl_down(value, 2);
  if (key < up_key) {
    value = up_value;
    key = up_key;
  }
  up_key = __shfl_down(key, 1);
  up_value = __shfl_down(value, 1);
  if (key < up_key) {
    value = up_value;
    key = up_key;
  }
}

__device__ __forceinline__ void block_max(
    int &value, float &key, int *shared_value,
    float *shared_key)  /// threadDim = 1024
{
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  __syncthreads();
  warp_max(value, key);
  if (lane_id == 0) {
    shared_value[warp_id] = value;
    shared_key[warp_id] = key;
  }
  __syncthreads();
  if (warp_id == 0) {
    value = shared_value[lane_id];
    key = shared_key[lane_id];
    warp_max(value, key);
  }
}

__device__ __forceinline__ int2 &getProjectIndex(
    const msfr::intrinsics &camera_intrinsics, const float3 &v) {
  return make_int2(
      __float2int_rd(camera_intrinsics.fx * v.x / v.z + camera_intrinsics.cx),
      __float2int_rd(camera_intrinsics.fy * v.y / v.z + camera_intrinsics.cy));
}
#endif




__host__ __device__ __forceinline__ float2 &getProjectPos(
    const msfr::intrinsics &camera_intrinsics, const float3 &v) {
  return make_float2(camera_intrinsics.fx * v.x / v.z + camera_intrinsics.cx,
                     camera_intrinsics.fy * v.y / v.z + camera_intrinsics.cy);
}

/// TODO  index.x index.y should plus 0.5 here.
__host__ __device__ __forceinline__ float3 &unProjectedFromIndex(
    const msfr::intrinsics &camera_intrinsics, const float3 &index) {
  return make_float3(
      (index.x - camera_intrinsics.cx) * index.z / camera_intrinsics.fx,
      (index.y - camera_intrinsics.cy) * index.z / camera_intrinsics.fy,
      index.z);
}



__device__ __forceinline__ float norm2(const float3 &x) {
  return x.x * x.x + x.y * x.y + x.z * x.z;
}



