#include "Common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl\gpu\containers\device_array.h>
#include <pcl\gpu\utils\cutil_math.h>
#include <pcl\gpu\utils\safe_call.hpp>


#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "MSFRUtil.cu"


__global__ void kernelComputeNormal(pcl::gpu::PtrSz<float3> normal,
                                    const pcl::gpu::PtrSz<float3> x,
                                    const pcl::gpu::PtrSz<int3> tri_list,
                                    const pcl::gpu::PtrSz<int2> fvLookUpTable,
                                    const pcl::gpu::PtrSz<int1> fbegin,
                                    const pcl::gpu::PtrSz<int1> fend) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < x.size) {
    float3 vn = make_float3(0.0f, 0.0f, 0.0f);
    // float area = 0.0f;
    for (int i = fbegin[index].x; i < fend[index].x; ++i) {
      int findex = fvLookUpTable[i].x;
      float3 v0 = x[tri_list[findex].x];
      float3 v1 = x[tri_list[findex].y];
      float3 v2 = x[tri_list[findex].z];
      vn += cross(v1 - v0, v2 - v0);
    }
    normal[index] = vn / length(vn);
  }
}

void cudaUpdateNormal(pcl::gpu::DeviceArray<float3> normal,
                      const pcl::gpu::DeviceArray<float3> x,
                      const pcl::gpu::DeviceArray<int3> tri_list,
                      const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                      const pcl::gpu::DeviceArray<int1> fbegin,
                      const pcl::gpu::DeviceArray<int1> fend) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(x.size(), block.x));
  kernelComputeNormal<<<grid, block>>>(normal, x, tri_list, fvLookUpTable,
                                       fbegin, fend);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  // std::vector<float3> host_normal;
  // std::vector<int3> host_tri_list;
  // std::vector<int2> host_fvLookUpTable;
  // std::vector<int1> host_fbegin, host_fend;
  // normal.download(host_normal);
  // fbegin.download(host_fbegin);
  // fend.download(host_fend);
  // fvLookUpTable.download(host_fvLookUpTable);
  // tri_list.download(host_tri_list);
}

__global__ void kernelSmoothMesh(
    pcl::gpu::PtrSz<float3> smooth_shape, const pcl::gpu::PtrSz<float3> shape,
    const pcl::gpu::PtrSz<int3> tri_list,
    const pcl::gpu::PtrSz<int2> fvLookUpTable,
    const pcl::gpu::PtrSz<int1> fBegin, const pcl::gpu::PtrSz<int1> fEnd,
    const pcl::gpu::PtrSz<unsigned short> is_boundary) {
  const int vId = threadIdx.x + blockIdx.x * blockDim.x;
  if (vId < shape.size) {
    float weight = 0.0f;
    float3 delta = {0.0f, 0.0f, 0.0f};
    float3 vI = shape[vId];
    if (is_nostril(vId)) {
      delta = vI;
      weight += 1.0f;
    } else if (is_boundary[vId] == 0) {
      delta += 2.0f * vI;
      weight += 2.0f;
      for (int i = fBegin[vId].x; i < fEnd[vId].x; ++i) {
        const int findex = fvLookUpTable[i].x;
        const int v[3] = {tri_list[findex].x, tri_list[findex].y,
                          tri_list[findex].z};
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          if (is_boundary[v[j]] == 0 && v[j] != vId) {
            delta += shape[v[j]];
            weight += 1.0f;
          }
        }
      }
    } else {
      for (int i = fBegin[vId].x; i < fEnd[vId].x; ++i) {
        const int findex = fvLookUpTable[i].x;
        const int v[3] = {tri_list[findex].x, tri_list[findex].y,
                          tri_list[findex].z};
        int fvId = -1;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          if (v[j] == vId) {
            fvId = j;
          }
        }
        int reIndex[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          reIndex[j] = fvId + j < 3 ? fvId + j : fvId + j - 3;
        }
        float3 verts[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          verts[j] = shape[v[reIndex[j]]];
        }
        float3 p0 = verts[1] - verts[0];
        float3 p1 = verts[2] - verts[1];
        float3 p2 = verts[2] - verts[0];
        float weight_i = 1.0f;
        if (is_boundary[vId] == 1) {
          float alpha = Angle(-p0, p1);
          weight_i = tan(M_PI * 0.5 - alpha);
        }
        delta += weight_i * verts[2];
        weight += weight_i;
        if (is_boundary[vId] == 1) {
          float alpha = Angle(p1, p2);
          weight_i = tan(M_PI * 0.5 - alpha);
        }
        delta += weight_i * verts[1];
        weight += weight_i;
      }
      delta += 20 * vI;
      weight += 20.0f;
    }
    delta /= weight;
    smooth_shape[vId] = delta;
  }
}

void cudaSmoothMesh(pcl::gpu::DeviceArray<float3> shape,
                    const pcl::gpu::DeviceArray<int3> tri_list,
                    const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                    const pcl::gpu::DeviceArray<int1> fBegin,
                    const pcl::gpu::DeviceArray<int1> fEnd,
                    const pcl::gpu::DeviceArray<unsigned short> is_boundary) {
  pcl::gpu::DeviceArray<float3> smooth_shape(shape.size());
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(shape.size(), block.x));
  kernelSmoothMesh<<<grid, block>>>(smooth_shape, shape, tri_list,
                                    fvLookUpTable, fBegin, fEnd, is_boundary);
  smooth_shape.copyTo(shape);
}

__global__ void kernelUpdateBoundary(
    pcl::gpu::PtrSz<unsigned short> is_boundary,
    const pcl::gpu::PtrSz<int3> tri_list,
    const pcl::gpu::PtrSz<int2> fvLookUpTable,
    const pcl::gpu::PtrSz<int1> fbegin, const pcl::gpu::PtrSz<int1> fend) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < is_boundary.size) {
    int delta = 0;
    for (int i = fbegin[index].x; i < fend[index].x; ++i) {
      const int findex = fvLookUpTable[i].x;
      const int v0 = tri_list[findex].x;
      const int v1 = tri_list[findex].y;
      const int v2 = tri_list[findex].z;
      if (v0 == index) {
        delta += v2 - v1;
      }
      if (v1 == index) {
        delta += v0 - v2;
      }
      if (v2 == index) {
        delta += v1 - v0;
      }
    }
    if (delta == 0) {
      is_boundary[index] = 1;
    } else {
      is_boundary[index] = 0;
    }
  }
}

void cudaUpdateBoundary(pcl::gpu::DeviceArray<unsigned short> is_boundary,
                        const pcl::gpu::DeviceArray<int3> tri_list,
                        const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                        const pcl::gpu::DeviceArray<int1> fbegin,
                        const pcl::gpu::DeviceArray<int1> fend) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(is_boundary.size(), block.x));
  kernelUpdateBoundary<<<grid, block>>>(is_boundary, tri_list, fvLookUpTable,
                                        fbegin, fend);
}

__global__ void kernelUpdateFrontVertices(
    pcl::gpu::PtrSz<unsigned short> is_front,
    const pcl::gpu::PtrSz<float3> position,
    const pcl::gpu::PtrSz<float> rotation_,
    const pcl::gpu::PtrSz<float> translation_, cudaTextureObject_t depth_map,
    const msfr::intrinsics camera_, const int width_, const int height_)
// pcl::gpu::PtrSz<float3> temp_depth)
{
  __shared__ msfr::intrinsics camera;
  __shared__ float rotation[9];
  __shared__ float3 translation;
  __shared__ int width, height;
  int id = threadIdx.x;
  if (id == 1) {
    camera = camera_;
    translation.x = translation_[0];
    translation.y = translation_[1];
    translation.z = translation_[2];
    width = width_;
    height = height_;
  }
  if (id < 9) {
    rotation[id] = rotation_[id] * translation_[3];
  }
  __syncthreads();
  id += blockDim.x * blockIdx.x;

  if (id < position.size) {
    float3 pos = M33xV3(rotation, position[id]) + translation;
    float2 uv = getProjectPos(camera, pos);
    int u_d = __float2int_rd(uv.x - 0.5);  /// find the left downward pixel
    int v_d = __float2int_rd(uv.y - 0.5);
    is_front[id] = 0;
    if (u_d > 0 && v_d > 0 && u_d + 1 < width && v_d + 1 < height) {
      float depth_max, depth_min;
      float depth = tex2D<float>(depth_map, u_d, v_d);
      depth_min = depth_max = depth;
      depth = tex2D<float>(depth_map, u_d + 1, v_d);
      depth_max = max(depth_max, depth);
      depth_min = min(depth_min, depth);
      depth = tex2D<float>(depth_map, u_d, v_d + 1);
      depth_max = max(depth_max, depth);
      depth_min = min(depth_min, depth);
      depth = tex2D<float>(depth_map, u_d + 1, v_d + 1);
      depth_max = max(depth_max, depth);
      depth_min = min(depth_min, depth);
      if (pos.z <= depth_max + 1e-4f &&
          depth_min > 0.0f)  /// using this 1e-4f to remove some err caused by
                             /// OpenGL Rendering
      {
        is_front[id] = 1;
      }
      // temp_depth[id] = make_float3(depth_max, depth_min, pos.z);
    }
  }
}

void cudaUpdateFrontVertices(pcl::gpu::DeviceArray<unsigned short> is_front,
                             const pcl::gpu::DeviceArray<float3> position,
                             const pcl::gpu::DeviceArray<float> rotation,
                             const pcl::gpu::DeviceArray<float> translation,
                             cudaTextureObject_t depth_map,
                             const msfr::intrinsics camera, const int width,
                             const int height) {
  // pcl::gpu::DeviceArray<float3> temp_depth(position.size());
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position.size(), block.x));
  kernelUpdateFrontVertices<<<grid, block>>>(is_front, position, rotation,
                                             translation, depth_map, camera,
                                             width, height);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  // std::vector<unsigned short> host_temp_depth;
  // temp_depth.download(host_temp_depth);
}
#endif  // USE_CUDA