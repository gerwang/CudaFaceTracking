#include "Common.h"

#ifdef USE_CUDA
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <device_launch_parameters.h>
#include <pcl\gpu\containers\device_array.h>
#include <pcl\gpu\utils\cutil_math.h>

#include <iostream>
#include <pcl\gpu\utils\safe_call.hpp>

#include "MSFRUtil.cu"

template <typename T>
inline void clearCudaMem(pcl::gpu::DeviceArray<T> mem) {
  cudaSafeCall(cudaMemset(mem.ptr(), 0, mem.size() * mem.elem_size));
}

struct einfo {
  int sid, eid, fid, indid = -1;
  bool einfo::operator<(const einfo& b) const {
    if (sid < b.sid) {
      return true;
    } else if (sid > b.sid) {
      return false;
    } else {
      return eid < b.eid;
    }
  }
};

__device__ __forceinline__ void getVertexinVector(
    float3& vt, const pcl::gpu::PtrSz<float>& v, const int v_offset) {
  vt.x = __ldg(&v[v_offset]);
  vt.y = __ldg(&v[v_offset + 1]);
  vt.z = __ldg(&v[v_offset + 2]);
}

__global__ void kernelProductMatVectorII(pcl::gpu::PtrSz<float> x,
                                         const pcl::gpu::PtrSz<float> A,
                                         const pcl::gpu::PtrSz<float> b,
                                         const pcl::gpu::PtrSz<float> c) {
  extern __shared__ float shared_b[];
  const int step = b.size;
  const int vId_i = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < step) {
    shared_b[threadIdx.x] = __ldg(&b[threadIdx.x]);
  }
  __syncthreads();
  if (vId_i < x.size) {
    float x_i = __ldg(&c[vId_i]);
    for (int i = 0, A_offset = vId_i * step; i < step; ++i, ++A_offset) {
      x_i += __ldg(&A[A_offset]) * shared_b[i];
    }
    x[vId_i] = x_i;
  }
}

void cudaProductMatVectorII(pcl::gpu::DeviceArray<float3> x,
                            const pcl::gpu::DeviceArray<float> A,
                            const pcl::gpu::DeviceArray<float> b,
                            const pcl::gpu::DeviceArray<float3> c) {
  pcl::gpu::DeviceArray<float> x_((float*)x.ptr(), x.size() * 3),
      c_((float*)c.ptr(), c.size() * 3);
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(x_.size(), block.x));

  kernelProductMatVectorII<<<grid, block, b.size() * sizeof(float), 0>>>(x_, A,
                                                                         b, c_);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelProductMatVector_Corr(pcl::gpu::PtrSz<float> x,
                                            const pcl::gpu::PtrSz<float> A,
                                            const pcl::gpu::PtrSz<float> b,
                                            const pcl::gpu::PtrSz<float> c) {
  extern __shared__ float shared_b[];
  const int step = b.size;
  const int vId_i = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadIdx.x < step) {
    float b_i = __ldg(&b[threadIdx.x]);
    shared_b[threadIdx.x] =
        b_i > CORRECTIVE_THRESHOLD
            ? (b_i - CORRECTIVE_THRESHOLD) / ONE_MINUS_CORRECTIVE_THRESHOLD
            : 0.0f;
  }
  __syncthreads();
  if (vId_i < x.size) {
    float x_i = __ldg(&c[vId_i]);
    for (int i = 0, A_offset = vId_i * step; i < step; ++i, ++A_offset) {
      x_i += __ldg(&A[A_offset]) * shared_b[i];
    }
    x[vId_i] = x_i;
  }
}

void cudaProductMatVector_Corr(pcl::gpu::DeviceArray<float3> x,
                               const pcl::gpu::DeviceArray<float> A,
                               const pcl::gpu::DeviceArray<float> b,
                               const pcl::gpu::DeviceArray<float3> c) {
  pcl::gpu::DeviceArray<float> x_((float*)x.ptr(), x.size() * 3),
      c_((float*)c.ptr(), c.size() * 3);
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(x_.size(), block.x));

  kernelProductMatVector_Corr<<<grid, block, b.size() * sizeof(float), 0>>>(
      x_, A, b, c_);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

void cudaProductMatVectorIII(pcl::gpu::DeviceArray<float3> x,
                             const pcl::gpu::DeviceArray<float> A,
                             const pcl::gpu::DeviceArray<float> b,
                             const pcl::gpu::DeviceArray<float3> c,
                             cudaStream_t stream) {
  pcl::gpu::DeviceArray<float> x_((float*)x.ptr(), x.size() * 3),
      c_((float*)c.ptr(), c.size() * 3);
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(x_.size(), block.x));

  kernelProductMatVectorII<<<grid, block, b.size() * sizeof(float), stream>>>(
      x_, A, b, c_);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(stream));
#endif
}

// compute x=A*b+c where A is a matrix and b is a vector
__global__ void kernelProductMatVector(pcl::gpu::PtrSz<float3> x,
                                       const pcl::gpu::PtrSz<float> A,
                                       const pcl::gpu::PtrSz<float> b,
                                       const pcl::gpu::PtrSz<float3> c) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < x.size) {
    float3 value = c[index];
    int step1 = 3 * index * b.size;
    int step2 = (3 * index + 1) * b.size;
    int step3 = (3 * index + 2) * b.size;

    for (int i = 0; i < b.size; ++i) {
      __shared__ float b_i;
      if (threadIdx.x == 0) {
        b_i = __ldg(&b[i]);
      }
      __syncthreads();
      value.x += __ldg(&A[i + step1]) * b_i;
      value.y += __ldg(&A[i + step2]) * b_i;
      value.z += __ldg(&A[i + step3]) * b_i;
      __syncthreads();
    }
    x[index] = value;
  }
}

void cudaProductMatVector(pcl::gpu::DeviceArray<float3> x,
                          const pcl::gpu::DeviceArray<float> A,
                          const pcl::gpu::DeviceArray<float> b,
                          const pcl::gpu::DeviceArray<float3> c) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(x.size(), block.x));

  kernelProductMatVector<<<grid, block>>>(x, A, b, c);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__device__ float gmem_partials[1024];
__device__ float prev_delta, delta, dTq;

__global__ void kernelInitV_(pcl::gpu::PtrSz<float> v_,
                             const pcl::gpu::PtrSz<float3> position,
                             const pcl::gpu::PtrSz<int3> tri_list) {
  int triId = blockIdx.x * blockDim.x + threadIdx.x;
  float3* v_ptr = (float3*)&v_[triId * 12];
  if (triId < tri_list.size) {
    int3 vIds = tri_list[triId];
    auto v0 = position[vIds.x];
    auto v1 = position[vIds.y];
    auto v2 = position[vIds.z];

    v2 -= v0;
    v1 -= v0;
    auto v3 = cross(v1, v2);
    v3 /= length(v3);
    // Q a 0 0 | 1/a    0   0 Q^T
    //   b d 0 | -b/ad  1/d 0
    //   c 0 1 | -c/a   0   1
    float d = length(v2);
    v2 /= d;
    float c = dot(v1, v3);
    v1 -= c * v3;
    float b = dot(v1, v2);
    v1 -= b * v2;
    float a = length(v1);
    v1 /= a;

    v1 /= a;
    v2 = (v2 - b * v1) / d;
    v3 -= c * v1;
    float3 v4 = v1 + v2 + v3;
    v_ptr[0] = -v4;
    v_ptr[1] = v1;
    v_ptr[2] = v2;
    v_ptr[3] = v3;
  }
}

void cudaInitV_(pcl::gpu::DeviceArray<float> v_,
                const pcl::gpu::DeviceArray<float3> position,
                const pcl::gpu::DeviceArray<int3> tri_list) {
  dim3 block(32);
  dim3 grid(pcl::gpu::divUp(tri_list.size(), block.x));
  kernelInitV_<<<grid, block>>>(v_, position, tri_list);
}

__device__ __forceinline__ void getVertexInBase(
    float3& v, const int vId, const int dimId, const int base_dim,
    const pcl::gpu::PtrSz<float>& base) {
  int base_offset = base_dim * vId * 3 + dimId;
  v.x = __ldg(&base[base_offset]);
  v.y = __ldg(&base[base_offset + base_dim]);
  v.z = __ldg(&base[base_offset + (base_dim << 1)]);
}

__device__ __forceinline__ void updateJ0TJ1x(
    float* r, const pcl::gpu::PtrSz<float>& v0_,
    const pcl::gpu::PtrSz<float>& v1_, const pcl::gpu::PtrSz<float>& x,
    const pcl::gpu::PtrSz<int3>& tri_list,
    const pcl::gpu::PtrSz<int2>& fvLookUpTable,
    const pcl::gpu::PtrSz<int1>& fbegin, const pcl::gpu::PtrSz<int1>& fend) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int position_size = fbegin.size;
  int tri_size = tri_list.size;
  r[0] = 0.0f;
  r[1] = 0.0f;
  r[2] = 0.0f;
  if (vId < position_size) {
    for (int i = fbegin[vId].x; i < fend[vId].x; ++i) {
      const int findex = fvLookUpTable[i].x;
      const int* tri = &tri_list[findex].x;
      int offset;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        if (__ldg(&tri[j]) == vId) {
          offset = j;
          break;
        }
      }
      float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      int f_offset = 12 * findex;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        float c_j = __ldg(&v0_[f_offset + 3 * offset + j]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          weights[k] += c_j * __ldg(&v1_[f_offset + 3 * k + j]);
        }
      }
#pragma unroll
      for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
          r[j] += weights[k] * __ldg(&x[3 * __ldg(&tri[k]) + j]);
        }
        r[j] += weights[3] * __ldg(&x[3 * (position_size + findex) + j]);
      }
    }
  } else if (vId < position_size + tri_size) {
    int findex = vId - position_size;
    const int* tri = &tri_list[findex].x;
    float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int f_offset = 12 * findex;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float c_j = __ldg(&v0_[f_offset + 9 + j]);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        weights[k] += c_j * __ldg(&v1_[f_offset + 3 * k + j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        r[j] += weights[k] * __ldg(&x[3 * __ldg(&tri[k]) + j]);
      }
      r[j] += weights[3] * __ldg(&x[3 * vId + j]);
    }
  }
}

__device__ __forceinline__ void updateJ0TJ1x_is_still(
    float* r, const pcl::gpu::PtrSz<float>& is_still,
    const pcl::gpu::PtrSz<float>& v0_, const pcl::gpu::PtrSz<float>& v1_,
    const pcl::gpu::PtrSz<float>& x, const pcl::gpu::PtrSz<int3>& tri_list,
    const pcl::gpu::PtrSz<int2>& fvLookUpTable,
    const pcl::gpu::PtrSz<int1>& fbegin, const pcl::gpu::PtrSz<int1>& fend) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int position_size = fbegin.size;
  int tri_size = tri_list.size;
  r[0] = 0.0f;
  r[1] = 0.0f;
  r[2] = 0.0f;
  if (vId < position_size) {
    for (int i = fbegin[vId].x; i < fend[vId].x; ++i) {
      const int findex = fvLookUpTable[i].x;
      const int* tri = &tri_list[findex].x;
      int offset;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        if (__ldg(&tri[j]) == vId) {
          offset = j;
          break;
        }
      }
      float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      int f_offset = 12 * findex;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        float c_j = __ldg(&v0_[f_offset + 3 * offset + j]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          weights[k] += c_j * __ldg(&v1_[f_offset + 3 * k + j]);
        }
      }
#pragma unroll
      for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
          r[j] += weights[k] * __ldg(&x[3 * __ldg(&tri[k]) + j]);
        }
        r[j] += weights[3] * __ldg(&x[3 * (position_size + findex) + j]);
      }
    }
    float3 v_i;
    getVertexinVector(v_i, x, 3 * vId);
    if (is_still[vId] > 0.1f) {
      v_i *= 1e+3f;
    } else {
      v_i *= 1e+3f;
    }
    r[0] += v_i.x;
    r[1] += v_i.y;
    r[2] += v_i.z;
  } else if (vId < position_size + tri_size) {
    int findex = vId - position_size;
    const int* tri = &tri_list[findex].x;
    float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int f_offset = 12 * findex;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float c_j = __ldg(&v0_[f_offset + 9 + j]);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        weights[k] += c_j * __ldg(&v1_[f_offset + 3 * k + j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        r[j] += weights[k] * __ldg(&x[3 * __ldg(&tri[k]) + j]);
      }
      r[j] += weights[3] * __ldg(&x[3 * vId + j]);
    }
  }
}

__global__ void kernel_prepare_x0(pcl::gpu::PtrSz<float> x,
                                  const pcl::gpu::PtrSz<float> exp_base,
                                  const pcl::gpu::PtrSz<float3> mean_shape,
                                  const pcl::gpu::PtrSz<int3> tri_list,
                                  const int dimId, const int base_dim) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int v_offset = 3 * vId;
  int tri_size = tri_list.size;
  int position_size = exp_base.size / (base_dim * 3);
  if (vId < position_size) {
    int base_offset = base_dim * vId * 3 + dimId;
    float3 v = mean_shape[vId];
    x[v_offset] = __ldg(&exp_base[base_offset]) + v.x;
    x[v_offset + 1] = __ldg(&exp_base[base_offset + base_dim]) + v.y;
    x[v_offset + 2] = __ldg(&exp_base[base_offset + 2 * base_dim]) + v.z;
  } else if (vId < position_size + tri_size) {
    int triId = vId - position_size;
    int3 tri = tri_list[triId];

    float3 v0, v1, v2;
    getVertexInBase(v0, tri.x, dimId, base_dim, exp_base);
    getVertexInBase(v1, tri.y, dimId, base_dim, exp_base);
    getVertexInBase(v2, tri.z, dimId, base_dim, exp_base);
    v0 += mean_shape[tri.x];
    v1 += mean_shape[tri.y];
    v2 += mean_shape[tri.z];
    float3 n = cross(v1 - v0, v2 - v0);
    n /= length(n);
    n += v0;
    x[v_offset] = n.x;
    x[v_offset + 1] = n.y;
    x[v_offset + 2] = n.z;
  }
}

__global__ void kernel_prepare_x0_is_still(
    pcl::gpu::PtrSz<float> x, pcl::gpu::PtrSz<float> is_still,
    const pcl::gpu::PtrSz<float> exp_base,
    const pcl::gpu::PtrSz<float3> mean_shape,
    const pcl::gpu::PtrSz<int3> tri_list, const int dimId, const int base_dim) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int v_offset = 3 * vId;
  int tri_size = tri_list.size;
  int position_size = exp_base.size / (base_dim * 3);
  if (vId < position_size) {
    int base_offset = base_dim * vId * 3 + dimId;
    float3 v = mean_shape[vId];
    const float3 delta = make_float3(
        __ldg(&exp_base[base_offset]), __ldg(&exp_base[base_offset + base_dim]),
        __ldg(&exp_base[base_offset + 2 * base_dim]));
    if (abs(delta.x) < 2e-3f && abs(delta.y) < 2e-3f && abs(delta.z) < 2e-3f) {
      is_still[vId] = 0.5f;
    } else {
      is_still[vId] = 0.0f;
    }
    x[v_offset] = delta.x + v.x;
    x[v_offset + 1] = delta.y + v.y;
    x[v_offset + 2] = delta.z + v.z;
  } else if (vId < position_size + tri_size) {
    int triId = vId - position_size;
    int3 tri = tri_list[triId];

    float3 v0, v1, v2;
    getVertexInBase(v0, tri.x, dimId, base_dim, exp_base);
    getVertexInBase(v1, tri.y, dimId, base_dim, exp_base);
    getVertexInBase(v2, tri.z, dimId, base_dim, exp_base);
    v0 += mean_shape[tri.x];
    v1 += mean_shape[tri.y];
    v2 += mean_shape[tri.z];
    float3 n = cross(v1 - v0, v2 - v0);
    n /= length(n);
    n += v0;
    x[v_offset] = n.x;
    x[v_offset + 1] = n.y;
    x[v_offset + 2] = n.z;
  }
}

__global__ void kernel_update_r0_d0(
    pcl::gpu::PtrSz<float> r, pcl::gpu::PtrSz<float> d,
    const pcl::gpu::PtrSz<float> x, const pcl::gpu::PtrSz<float> M_inv,
    const pcl::gpu::PtrSz<float> vs_, const pcl::gpu::PtrSz<float> vt_,
    const pcl::gpu::PtrSz<int3> tri_list,
    const pcl::gpu::PtrSz<int2> fvLookUpTable,
    const pcl::gpu::PtrSz<int1> fbegin, const pcl::gpu::PtrSz<int1> fend) {
  __shared__ float partial_sum[32];
  int vId = blockIdx.x * blockDim.x + threadIdx.x;
  int v_offset = 3 * vId;
  int position_size = fbegin.size;
  int tri_size = tri_list.size;
  float rTd_i = 0.0f;
  if (vId < position_size + tri_size) {
    float3 r_i;

    float3 Ax_i;
    updateJ0TJ1x(&r_i.x, vt_, vs_, x, tri_list, fvLookUpTable, fbegin, fend);
    updateJ0TJ1x(&Ax_i.x, vt_, vt_, x, tri_list, fvLookUpTable, fbegin, fend);
    // r_i -= Ax_i;
    r[v_offset] = r_i.x;
    r[v_offset + 1] = r_i.y;
    r[v_offset + 2] = r_i.z;
    float3 d_i = r_i * __ldg(&M_inv[vId]);
    d[v_offset] = d_i.x;
    d[v_offset + 1] = d_i.y;
    d[v_offset + 2] = d_i.z;
    rTd_i = dot(r_i, d_i);
  }
  block_sum(rTd_i, partial_sum);
  if (threadIdx.x == 31) {
    gmem_partials[blockIdx.x] = rTd_i;
  }
}

__global__ void kernel_update_r0_d0_is_still(
    pcl::gpu::PtrSz<float> r, pcl::gpu::PtrSz<float> d,
    const pcl::gpu::PtrSz<float> x, const pcl::gpu::PtrSz<float> M_inv,
    const pcl::gpu::PtrSz<float> personalized_mean_shape,
    const pcl::gpu::PtrSz<float> is_still, const pcl::gpu::PtrSz<float> vs_,
    const pcl::gpu::PtrSz<float> vt_, const pcl::gpu::PtrSz<int3> tri_list,
    const pcl::gpu::PtrSz<int2> fvLookUpTable,
    const pcl::gpu::PtrSz<int1> fbegin, const pcl::gpu::PtrSz<int1> fend) {
  __shared__ float partial_sum[32];
  int vId = blockIdx.x * blockDim.x + threadIdx.x;
  int v_offset = 3 * vId;
  int position_size = fbegin.size;
  int tri_size = tri_list.size;
  float rTd_i = 0.0f;
  if (vId < position_size + tri_size) {
    float3 r_i;

    float3 Ax_i;
    updateJ0TJ1x(&r_i.x, vt_, vs_, x, tri_list, fvLookUpTable, fbegin, fend);
    if (vId < position_size) {
      float3 personalized_mean_shape_v_i;
      getVertexinVector(personalized_mean_shape_v_i, personalized_mean_shape,
                        vId * 3);
      float weight;
      // if (is_still[vId] > 0.0f)
      { weight = 1e+3f; }
      // else
      //{
      //  weight = 1e+2f;
      //}
      r_i += weight * personalized_mean_shape_v_i;
    }
    updateJ0TJ1x_is_still(&Ax_i.x, is_still, vt_, vt_, x, tri_list,
                          fvLookUpTable, fbegin, fend);
    r_i -= Ax_i;
    r[v_offset] = r_i.x;
    r[v_offset + 1] = r_i.y;
    r[v_offset + 2] = r_i.z;
    float3 d_i = __ldg(&M_inv[vId]) * r_i;
    d[v_offset] = d_i.x;
    d[v_offset + 1] = d_i.y;
    d[v_offset + 2] = d_i.z;
    rTd_i = dot(r_i, d_i);
  }
  block_sum(rTd_i, partial_sum);
  if (threadIdx.x == 31) {
    gmem_partials[blockIdx.x] = rTd_i;
  }
}

__global__ void kernel_update_q(pcl::gpu::PtrSz<float> q,
                                const pcl::gpu::PtrSz<float> d,
                                const pcl::gpu::PtrSz<float> v_,
                                const pcl::gpu::PtrSz<int3> tri_list,
                                const pcl::gpu::PtrSz<int2> fvLookUpTable,
                                const pcl::gpu::PtrSz<int1> fbegin,
                                const pcl::gpu::PtrSz<int1> fend) {
  __shared__ float partial_sum[32];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    prev_delta = delta;
  }

  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int v_offset = 3 * vId;
  float dTq_i = 0.0f;
  if (v_offset < q.size) {
    float3 q_i;
    float3 d_i;
    updateJ0TJ1x(&q_i.x, v_, v_, d, tri_list, fvLookUpTable, fbegin, fend);
    q[v_offset] = q_i.x;
    q[v_offset + 1] = q_i.y;
    q[v_offset + 2] = q_i.z;
    getVertexinVector(d_i, d, v_offset);
    dTq_i = dot(q_i, d_i);
  }
  block_sum(dTq_i, partial_sum);

  if (threadIdx.x == 31) {
    gmem_partials[blockIdx.x] = dTq_i;
  }
}

__global__ void kernel_update_q_is_still(
    pcl::gpu::PtrSz<float> q, const pcl::gpu::PtrSz<float> d,
    const pcl::gpu::PtrSz<float> is_still, const pcl::gpu::PtrSz<float> v_,
    const pcl::gpu::PtrSz<int3> tri_list,
    const pcl::gpu::PtrSz<int2> fvLookUpTable,
    const pcl::gpu::PtrSz<int1> fbegin, const pcl::gpu::PtrSz<int1> fend) {
  __shared__ float partial_sum[32];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    prev_delta = delta;
  }

  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int v_offset = 3 * vId;
  float dTq_i = 0.0f;
  if (v_offset < q.size) {
    float3 q_i;
    float3 d_i;
    updateJ0TJ1x_is_still(&q_i.x, is_still, v_, v_, d, tri_list, fvLookUpTable,
                          fbegin, fend);
    q[v_offset] = q_i.x;
    q[v_offset + 1] = q_i.y;
    q[v_offset + 2] = q_i.z;
    getVertexinVector(d_i, d, v_offset);
    dTq_i = dot(q_i, d_i);
  }
  block_sum(dTq_i, partial_sum);

  if (threadIdx.x == 31) {
    gmem_partials[blockIdx.x] = dTq_i;
  }
}

__global__ void kernel_update_delta(const int legal_num) {
  __shared__ float partial_sum[32];
  float rTd_i = 0.0f;
  if (threadIdx.x < legal_num) {
    rTd_i = gmem_partials[threadIdx.x];
  }
  block_sum(rTd_i, partial_sum);

  if (threadIdx.x == 31) {
    delta = rTd_i;
  }
}

__global__ void kernel_update_x_r_s(pcl::gpu::PtrSz<float> x,
                                    pcl::gpu::PtrSz<float> r,
                                    pcl::gpu::PtrSz<float> s,
                                    const pcl::gpu::PtrSz<float> d,
                                    const pcl::gpu::PtrSz<float> q,
                                    const pcl::gpu::PtrSz<float> M_inv) {
  __shared__ float alpha;
  __shared__ float partial_sum[32];
  float dTq_i = 0.0f;
  if (threadIdx.x < blockDim.x) {
    dTq_i = gmem_partials[threadIdx.x];
  }
  block_sum(dTq_i, partial_sum);

  if (blockIdx.x == 0 && threadIdx.x == 31) {
    dTq = dTq_i;
  }
  if (threadIdx.x == 31) {
    alpha = delta / dTq_i;
  }
  __syncthreads();
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int v_offset = 3 * vId;
  float rTs_i = 0.0f;
  if (v_offset < x.size) {
    float3 new_x_i, new_r_i, d_i, q_i, s_i;
    getVertexinVector(new_x_i, x, v_offset);
    getVertexinVector(new_r_i, r, v_offset);
    getVertexinVector(d_i, d, v_offset);
    getVertexinVector(q_i, q, v_offset);

    new_x_i += alpha * d_i;
    new_r_i -= alpha * q_i;

    s_i = new_r_i * __ldg(&M_inv[vId]);
    rTs_i = dot(new_r_i, s_i);
    r[v_offset] = new_r_i.x;
    r[v_offset + 1] = new_r_i.y;
    r[v_offset + 2] = new_r_i.z;
    x[v_offset] = new_x_i.x;
    x[v_offset + 1] = new_x_i.y;
    x[v_offset + 2] = new_x_i.z;
    s[v_offset] = s_i.x;
    s[v_offset + 1] = s_i.y;
    s[v_offset + 2] = s_i.z;
  }
  block_sum(rTs_i, partial_sum);

  if (threadIdx.x == 31) {
    gmem_partials[blockIdx.x] = rTs_i;
  }
}

__global__ void kernel_update_d(pcl::gpu::PtrSz<float> d,
                                const pcl::gpu::PtrSz<float> s) {
  __shared__ float beta;
  __shared__ float partial_sum[32];
  float rTs_i = 0.0f;
  if (threadIdx.x < blockDim.x) {
    rTs_i = gmem_partials[threadIdx.x];
  }
  block_sum(rTs_i, partial_sum);
  if (blockIdx.x == 0 && threadIdx.x == 31) {
    delta = rTs_i;
  }
  if (threadIdx.x == 31) {
    beta = rTs_i / prev_delta;
  }
  __syncthreads();
  int v_offset = 3 * (blockDim.x * blockIdx.x + threadIdx.x);
  if (v_offset < d.size) {
    float3 d_i, s_i;
    getVertexinVector(d_i, d, v_offset);
    getVertexinVector(s_i, s, v_offset);
    d_i = s_i + beta * d_i;
    d[v_offset] = d_i.x;
    d[v_offset + 1] = d_i.y;
    d[v_offset + 2] = d_i.z;
  }
}

__global__ void kernel_update_base(pcl::gpu::PtrSz<float> base,
                                   const pcl::gpu::PtrSz<float> base_i,
                                   const pcl::gpu::PtrSz<float3> mean_shape,
                                   const int dimId, const int base_dim) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int base_offset = 3 * vId * base_dim + dimId;
  int v_offset = vId * 3;
  if (base_offset < base.size) {
    float3 mean_shape_i = mean_shape[vId];
    base[base_offset] = base_i[v_offset] - mean_shape_i.x;
    base[base_offset + base_dim] = base_i[v_offset + 1] - mean_shape_i.y;
    base[base_offset + (base_dim << 1)] = base_i[v_offset + 2] - mean_shape_i.z;
  }
}

__global__ void kernel_init_M_inv(pcl::gpu::PtrSz<float> M_inv,
                                  const pcl::gpu::PtrSz<float> v_,
                                  const pcl::gpu::PtrSz<int3> tri_list,
                                  const pcl::gpu::PtrSz<int2> fvLookUpTable,
                                  const pcl::gpu::PtrSz<int1> fbegin,
                                  const pcl::gpu::PtrSz<int1> fend) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int position_size = fbegin.size;
  // int tri_size = tri_list.size;

  if (vId < M_inv.size) {
    float M_inv_i = 0.0f;
    if (vId < position_size) {
      for (int i = fbegin[vId].x; i < fend[vId].x; ++i) {
        const int findex = fvLookUpTable[i].x;
        const int* tri = &tri_list[findex].x;
        int offset;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          if (__ldg(&tri[j]) == vId) {
            offset = j;
            break;
          }
        }
        int f_offset = 12 * findex + 3 * offset;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          float c_j = __ldg(&v_[f_offset + j]);
          M_inv_i += c_j * c_j;
        }
      }
    } else {
      //      int findex = vId - position_size;
      //      const int* tri = &tri_list[findex].x;
      //      int f_offset = 12 * findex;
      //      float M_inv_i_max = 0.0f;
      //#pragma unroll
      //      for (int j = 0; j < 4; ++j)
      //      {
      //        M_inv_i = 0.0f;
      //        for (int k = 0; k < 3; k++)
      //        {
      //          M_inv_i += __ldg(&v_[f_offset + 9 + k]) * __ldg(&v_[f_offset +
      //          3 * j + k]);
      //        }
      //        if (M_inv_i_max < M_inv_i)
      //        {
      //          M_inv_i_max = M_inv_i;
      //        }
      //      }
      M_inv_i = 1.0f;
    }
    M_inv[vId] = 1.0f / M_inv_i;
  }
}

__global__ void kernel_init_M_inv_is_still(
    pcl::gpu::PtrSz<float> M_inv, const pcl::gpu::PtrSz<float> v_,
    const pcl::gpu::PtrSz<float> is_still, const pcl::gpu::PtrSz<int3> tri_list,
    const pcl::gpu::PtrSz<int2> fvLookUpTable,
    const pcl::gpu::PtrSz<int1> fbegin, const pcl::gpu::PtrSz<int1> fend) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  int position_size = fbegin.size;
  // int tri_size = tri_list.size;

  if (vId < M_inv.size) {
    float M_inv_i = 0.0f;
    if (vId < position_size) {
      for (int i = fbegin[vId].x; i < fend[vId].x; ++i) {
        const int findex = fvLookUpTable[i].x;
        const int* tri = &tri_list[findex].x;
        int offset;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          if (__ldg(&tri[j]) == vId) {
            offset = j;
            break;
          }
        }
        int f_offset = 12 * findex + 3 * offset;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          float c_j = __ldg(&v_[f_offset + j]);
          M_inv_i += c_j * c_j;
        }
      }
      // M_inv_i = 1.0f;
      if (is_still[vId] != 0.0f) {
        M_inv_i += 1e+3f;
      } else {
        M_inv_i += 1e+3f;
      }
    } else {
      //      int findex = vId - position_size;
      //      const int* tri = &tri_list[findex].x;
      //      int f_offset = 12 * findex;
      //      float M_inv_i_max = 0.0f;
      //#pragma unroll
      //      for (int j = 0; j < 4; ++j)
      //      {
      //        M_inv_i = 0.0f;
      //        for (int k = 0; k < 3; k++)
      //        {
      //          M_inv_i += __ldg(&v_[f_offset + 9 + k]) * __ldg(&v_[f_offset +
      //          3 * j + k]);
      //        }
      //        if (M_inv_i_max < M_inv_i)
      //        {
      //          M_inv_i_max = M_inv_i;
      //        }
      //      }
      M_inv_i = 1.0f;
    }
    M_inv[vId] = 1.0f / M_inv_i;
  }
}

__global__ void kernel_update_Atb(pcl::gpu::PtrSz<float> Atb,
                                  const pcl::gpu::PtrSz<float> x,
                                  const pcl::gpu::PtrSz<float> vs_,
                                  const pcl::gpu::PtrSz<float> vt_,
                                  const pcl::gpu::PtrSz<int3> tri_list,
                                  const pcl::gpu::PtrSz<int2> fvLookUpTable,
                                  const pcl::gpu::PtrSz<int1> fbegin,
                                  const pcl::gpu::PtrSz<int1> fend) {
  int vId = blockIdx.x * blockDim.x + threadIdx.x;
  int v_offset = 3 * vId;
  if (v_offset < Atb.size) {
    float3 r_i;
    updateJ0TJ1x(&r_i.x, vt_, vs_, x, tri_list, fvLookUpTable, fbegin, fend);
    Atb[v_offset] = r_i.x;
    Atb[v_offset + 1] = r_i.y;
    Atb[v_offset + 2] = r_i.z;
  }
}

void cudaUpdateDimIdAtb(
    pcl::gpu::DeviceArray<float> Atb, const pcl::gpu::DeviceArray<float> vs_,
    const pcl::gpu::DeviceArray<float> vt_,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<float3> personalized_mean_shape,
    const pcl::gpu::DeviceArray<int3> tri_list,
    const pcl::gpu::DeviceArray<int2> fvLookUpTable,
    const pcl::gpu::DeviceArray<int1> fbegin,
    const pcl::gpu::DeviceArray<int1> fend, const int dimId) {
  int base_dim = exp_base.size() / 3 / fbegin.size();

  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(fbegin.size() + tri_list.size(), block.x));
  pcl::gpu::DeviceArray<float> x(Atb.size());

  kernel_prepare_x0<<<grid, block>>>(x, exp_base, mean_shape, tri_list, dimId,
                                     base_dim);
  kernel_update_Atb<<<grid, block>>>(Atb, x, vs_, vt_, tri_list, fvLookUpTable,
                                     fbegin, fend);
}

// void cudaUpdateBaseReg(pcl::gpu::DeviceArray<float> reg,
//  const pcl::gpu::DeviceArray<float> base, const int dim)
//{
//  dim3 block(1024);
//  dim3 grid()
//  kernelUpdateBaseReg
//}

__global__ void kernel_update_col_index(
    pcl::gpu::PtrSz<int> col_index, const pcl::gpu::PtrSz<einfo> evLookUpTable,
    const pcl::gpu::PtrSz<int> ev_offset,
    const pcl::gpu::PtrSz<int> row_offset) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  if (vId < ev_offset.size - 1) {
    int row_offset_i = __ldg(&row_offset[vId]);
    col_index[row_offset_i] = vId;
    for (int i = ev_offset[vId]; i < ev_offset[vId + 1]; ++i) {
      if (i == 0 || __ldg(&evLookUpTable[i].indid) !=
                        __ldg(&evLookUpTable[i - 1].indid)) {
        ++row_offset_i;
        col_index[row_offset_i] = __ldg(&evLookUpTable[i].eid);
      }
    }
  }
}

__global__ void kernel_update_A_values(
    pcl::gpu::PtrSz<float> A_values, const pcl::gpu::PtrSz<float> vt_,
    const pcl::gpu::PtrSz<einfo> evLookUpTable,
    const pcl::gpu::PtrSz<int> ev_offset,
    const pcl::gpu::PtrSz<int> row_offset) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  if (vId < ev_offset.size - 1) {
    int row_offset_i = row_offset[vId];
    for (int i = ev_offset[vId]; i < ev_offset[vId + 1]; ++i) {
      int key = __ldg(&evLookUpTable[i].sid);
      int key_b = key / 4;
      int key_e = key % 4;
      int f_offset = 12 * __ldg(&evLookUpTable[i].fid);
      float value_be = 0.0f;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        value_be += __ldg(&vt_[f_offset + 3 * key_b + j]) *
                    __ldg(&vt_[f_offset + 3 * key_e + j]);
      }
      A_values[vId + __ldg(&evLookUpTable[i].indid) + 1] += value_be;
      if ((key_e - key_b + 4) % 4 == 1) {
        float value_bb = 0.0f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          float c_j = __ldg(&vt_[f_offset + 3 * key_b + j]);
          value_bb += c_j * c_j;
        }
        A_values[row_offset[vId]] += value_bb;
      }
    }
  }
}

__global__ void kernelComputeRotation(
    pcl::gpu::PtrSz<float> rotation, const pcl::gpu::PtrSz<float> exp_base,
    const pcl::gpu::PtrSz<float3> mean_shape,
    const pcl::gpu::PtrSz<unsigned short> rigid_index) {
  __shared__ float partial_sum[32];
  const int rigid_num = rigid_index.size;
  const int max_idx = ((rigid_num + 1023) >> 10) << 10;
  const int col_num = exp_base.size / (3 * mean_shape.size);
  __shared__ float3 exp_0, mean_0;
  if (threadIdx.x < 3) {
    const int rigid_idx_0 = 7950;
    (&exp_0.x)[threadIdx.x] = __ldg(
        &exp_base[blockIdx.x + (rigid_idx_0 * 3 + threadIdx.x) * col_num]);
    (&mean_0.x)[threadIdx.x] = (&mean_shape[rigid_idx_0].x)[threadIdx.x];
  }
  if (threadIdx.x < 9) {
    rotation[blockIdx.x * 9 + threadIdx.x] = 0.0f;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < max_idx; i += 1024) {
    float3 delta_exp = {0.0f, 0.0f, 0.0f};
    float3 delta_mean = {0.0f, 0.0f, 0.0f};
    if (i < rigid_num) {
      const int vId = __ldg(&rigid_index[i]);
      delta_exp.x = __ldg(&exp_base[blockIdx.x + (vId * 3 + 0) * col_num]);
      delta_exp.y = __ldg(&exp_base[blockIdx.x + (vId * 3 + 1) * col_num]);
      delta_exp.z = __ldg(&exp_base[blockIdx.x + (vId * 3 + 2) * col_num]);
      delta_exp -= exp_0;
      delta_mean = mean_shape[vId] - mean_0;
    }
    __syncthreads();
    const float* delta_exp_ptr = &delta_exp.x;
    const float* delta_mean_ptr = &delta_mean.x;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        float r_jk = delta_exp_ptr[j] * delta_mean_ptr[k];
        block_sum(r_jk, partial_sum);
        __syncthreads();
        if (threadIdx.x == 31) {
          rotation[blockIdx.x * 9 + j * 3 + k] += r_jk;
        }
      }
    }
    __syncthreads();
  }
}

void cudaComputeRotation(
    pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<unsigned short> rigid_index) {
  const int dim_exp = exp_base.size() / 3 / mean_shape.size();
  // std::cout << ((rigid_index.size() + 1023) >> 10) << 10 << std::endl;
  dim3 block(1024);
  dim3 grid(dim_exp);
  kernelComputeRotation<<<grid, block>>>(rotation, exp_base, mean_shape,
                                         rigid_index);
  cudaSafeCall(cudaStreamSynchronize(0));
  // std::vector<float> host_rotation, host_exp_base;
  // std::vector<float3> host_mean_shape;
  // std::vector<unsigned short> host_rigid_index;
  // rotation.download(host_rotation);
  // exp_base.download(host_exp_base);
  // mean_shape.download(host_mean_shape);
  // rigid_index.download(host_rigid_index);
  // std::cout << 1;
}

__global__ void kernelComputeBlendshape(
    pcl::gpu::PtrSz<float> exp_base, const pcl::gpu::PtrSz<float3> mean_shape) {
  const int vId = blockIdx.y * blockDim.x + threadIdx.x;
  const int col_num = exp_base.size / 3 / mean_shape.size;
  if (vId < mean_shape.size) {
    exp_base[blockIdx.x + (vId * 3 + 0) * col_num] += mean_shape[vId].x;
    exp_base[blockIdx.x + (vId * 3 + 1) * col_num] += mean_shape[vId].y;
    exp_base[blockIdx.x + (vId * 3 + 2) * col_num] += mean_shape[vId].z;
  }
}

void cudaComputeBlendshape(pcl::gpu::DeviceArray<float> exp_base,
                           const pcl::gpu::DeviceArray<float3> mean_shape,
                           cudaStream_t stream) {
  const int dim_exp = exp_base.size() / 3 / mean_shape.size();
  dim3 block(1024);
  dim3 grid(dim_exp, pcl::gpu::divUp(mean_shape.size(), block.x));
  kernelComputeBlendshape<<<grid, block, 0, stream>>>(exp_base, mean_shape);
}

__global__ void kernelComputeRotaionRefine(
    pcl::gpu::PtrSz<float> rotation, const pcl::gpu::PtrSz<float> exp_base,
    const pcl::gpu::PtrSz<float3> mean_shape) {
  __shared__ float partial_sum[32];
  const int verticex_num = mean_shape.size;
  const int max_idx = ((verticex_num + 1023) >> 10) << 10;
  const int col_num = exp_base.size / (3 * mean_shape.size);
  __shared__ float3 exp_0, mean_0;
  if (threadIdx.x < 3) {
    const int rigid_idx_0 = 7950;
    (&exp_0.x)[threadIdx.x] = __ldg(
        &exp_base[blockIdx.x + (rigid_idx_0 * 3 + threadIdx.x) * col_num]);
    (&mean_0.x)[threadIdx.x] = (&mean_shape[rigid_idx_0].x)[threadIdx.x];
  }
  if (threadIdx.x < 9) {
    rotation[blockIdx.x * 9 + threadIdx.x] = 0.0f;
  }
  __syncthreads();
  for (int vId = threadIdx.x; vId < max_idx; vId += 1024) {
    float3 delta_exp = {0.0f, 0.0f, 0.0f};
    float3 delta_mean = {0.0f, 0.0f, 0.0f};
    if (vId < verticex_num) {
      delta_exp.x = __ldg(&exp_base[blockIdx.x + (vId * 3 + 0) * col_num]);
      delta_exp.y = __ldg(&exp_base[blockIdx.x + (vId * 3 + 1) * col_num]);
      delta_exp.z = __ldg(&exp_base[blockIdx.x + (vId * 3 + 2) * col_num]);
      delta_mean = mean_shape[vId];
      delta_exp -= exp_0;
      delta_mean -= mean_0;
      if (norm2(delta_exp - delta_mean) > 4e-4f) {
        //} else {
        delta_exp = {0.0f, 0.0f, 0.0f};
        delta_mean = {0.0f, 0.0f, 0.0f};
      }
    }
    __syncthreads();

    const float* delta_exp_ptr = &delta_exp.x;
    const float* delta_mean_ptr = &delta_mean.x;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        float r_jk = delta_exp_ptr[j] * delta_mean_ptr[k];
        block_sum(r_jk, partial_sum);
        __syncthreads();
        if (threadIdx.x == 31) {
          rotation[blockIdx.x * 9 + j * 3 + k] += r_jk;
        }
      }
    }
    __syncthreads();
  }
}

void cudaComputeRotationRefine(pcl::gpu::DeviceArray<float> rotation,
                               const pcl::gpu::DeviceArray<float> exp_base,
                               const pcl::gpu::DeviceArray<float3> mean_shape,
                               cudaStream_t stream) {
  const int dim_exp = exp_base.size() / 3 / mean_shape.size();
  dim3 block(1024);
  dim3 grid(dim_exp);
  kernelComputeRotaionRefine<<<grid, block, 0, stream>>>(rotation, exp_base,
                                                         mean_shape);
  cudaSafeCall(cudaStreamSynchronize(stream));
}

__global__ void kernelComputeRotaionRefineMask(
    pcl::gpu::PtrSz<float> rotation, const pcl::gpu::PtrSz<float> exp_base,
    const pcl::gpu::PtrSz<float3> mean_shape,
    const pcl::gpu::PtrSz<float> exp_mask) {
  __shared__ float partial_sum[32];
  const int verticex_num = mean_shape.size;
  const int max_idx = ((verticex_num + 1023) >> 10) << 10;
  const int col_num = exp_base.size / (3 * mean_shape.size);
  __shared__ float3 exp_0, mean_0;
  if (threadIdx.x < 3) {
    const int rigid_idx_0 = 7950;
    (&exp_0.x)[threadIdx.x] = __ldg(
        &exp_base[blockIdx.x + (rigid_idx_0 * 3 + threadIdx.x) * col_num]);
    (&mean_0.x)[threadIdx.x] = (&mean_shape[rigid_idx_0].x)[threadIdx.x];
  }
  if (threadIdx.x < 9) {
    rotation[blockIdx.x * 9 + threadIdx.x] = 0.0f;
  }
  __syncthreads();
  for (int vId = threadIdx.x; vId < max_idx; vId += 1024) {
    float3 delta_exp = {0.0f, 0.0f, 0.0f};
    float3 delta_mean = {0.0f, 0.0f, 0.0f};
    if (vId < verticex_num) {
      delta_exp.x = __ldg(&exp_base[blockIdx.x + (vId * 3 + 0) * col_num]);
      delta_exp.y = __ldg(&exp_base[blockIdx.x + (vId * 3 + 1) * col_num]);
      delta_exp.z = __ldg(&exp_base[blockIdx.x + (vId * 3 + 2) * col_num]);
      delta_mean = mean_shape[vId];
      if (exp_mask[blockIdx.x + 3 * vId * col_num] < 0.2f) {
        delta_exp -= exp_0;
        delta_mean -= mean_0;
      } else {
        delta_exp = {0.0f, 0.0f, 0.0f};
        delta_mean = {0.0f, 0.0f, 0.0f};
      }
    }
    const float* delta_exp_ptr = &delta_exp.x;
    const float* delta_mean_ptr = &delta_mean.x;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        float r_jk = delta_exp_ptr[j] * delta_mean_ptr[k];
        block_sum(r_jk, partial_sum);
        __syncthreads();
        if (threadIdx.x == 31) {
          rotation[blockIdx.x * 9 + j * 3 + k] += r_jk;
        }
      }
    }
  }
}

void cudaComputeRotationRefineMask(
    pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<float> exp_mask, cudaStream_t stream) {
  const int dim_exp = exp_base.size() / 3 / mean_shape.size();
  dim3 block(1024);
  dim3 grid(dim_exp);
  kernelComputeRotaionRefineMask<<<grid, block, 0, stream>>>(
      rotation, exp_base, mean_shape, exp_mask);
  cudaSafeCall(cudaStreamSynchronize(stream));
}

__global__ void kernelComputeTranslationRefine(
    pcl::gpu::PtrSz<float> translation, const pcl::gpu::PtrSz<float> rotation,
    const pcl::gpu::PtrSz<float> exp_base,
    const pcl::gpu::PtrSz<float3> mean_shape) {
  __shared__ float partial_sum[32];
  const int vertices_num = mean_shape.size;
  const int max_idx = ((vertices_num + 1023) >> 10) << 10;
  const int col_num = exp_base.size / (3 * mean_shape.size);
  __shared__ float rotation_i[9];
  __shared__ float sum;
  if (threadIdx.x < 9) {
    rotation_i[threadIdx.x] = rotation[blockIdx.x * 9 + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    sum = 0.0f;
  }
  if (threadIdx.x < 3) {
    translation[threadIdx.x + blockIdx.x * 3] = 0.0f;
  }
  __shared__ float3 exp_0, mean_0;
  if (threadIdx.x < 3) {
    const int rigid_idx_0 = 7950;
    (&exp_0.x)[threadIdx.x] = __ldg(
        &exp_base[blockIdx.x + (rigid_idx_0 * 3 + threadIdx.x) * col_num]);
    (&mean_0.x)[threadIdx.x] = (&mean_shape[rigid_idx_0].x)[threadIdx.x];
  }
  __syncthreads();
  for (int vId = threadIdx.x; vId < max_idx; vId += 1024) {
    float3 delta = {0.0f, 0.0f, 0.0f};
    float sum_i = 0.0f;
    if (vId < vertices_num) {
      float3 delta_mean = {0.0f, 0.0f, 0.0f};
      delta.x = __ldg(&exp_base[blockIdx.x + (vId * 3 + 0) * col_num]);
      delta.y = __ldg(&exp_base[blockIdx.x + (vId * 3 + 1) * col_num]);
      delta.z = __ldg(&exp_base[blockIdx.x + (vId * 3 + 2) * col_num]);
      delta_mean = mean_shape[vId];

      if (norm2(delta - exp_0 + mean_0 - delta_mean) < 4e-4f) {
        delta = mean_shape[vId] - M33xV3(rotation_i, delta);
        sum_i = 1.0f;
      } else {
        delta = {0.0f, 0.0f, 0.0f};
      }
    }
    __syncthreads();

    const float* delta_ptr = &delta.x;
    block_sum(sum_i, partial_sum);
    __syncthreads();
    if (threadIdx.x == 31) {
      sum += sum_i;
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float d_j = delta_ptr[j];
      block_sum(d_j, partial_sum);
      __syncthreads();
      if (threadIdx.x == 31) {
        translation[blockIdx.x * 3 + j] += d_j;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 31) {
    translation[blockIdx.x * 3 + 0] /= sum;
    translation[blockIdx.x * 3 + 1] /= sum;
    translation[blockIdx.x * 3 + 2] /= sum;
  }
}

void cudaComputeTranslationRefine(
    pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape, cudaStream_t stream) {
  const int dim_exp = exp_base.size() / 3 / mean_shape.size();
  dim3 block(1024);
  dim3 grid(dim_exp);
  kernelComputeTranslationRefine<<<grid, block, 0, stream>>>(
      translation, rotation, exp_base, mean_shape);
  cudaSafeCall(cudaStreamSynchronize(0));
}

__global__ void kernelComputeTranslationRefineMask(
    pcl::gpu::PtrSz<float> translation, const pcl::gpu::PtrSz<float> rotation,
    const pcl::gpu::PtrSz<float> exp_base,
    const pcl::gpu::PtrSz<float3> mean_shape,
    const pcl::gpu::PtrSz<float> exp_mask) {
  __shared__ float partial_sum[32];
  const int vertices_num = mean_shape.size;
  const int max_idx = ((vertices_num + 1023) >> 10) << 10;
  const int col_num = exp_base.size / (3 * mean_shape.size);
  __shared__ float rotation_i[9];
  __shared__ float sum;
  if (threadIdx.x < 9) {
    rotation_i[threadIdx.x] = rotation[blockIdx.x * 9 + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    sum = 0.0f;
  }
  if (threadIdx.x < 3) {
    translation[threadIdx.x + blockIdx.x * 3] = 0.0f;
  }
  __syncthreads();
  for (int vId = threadIdx.x; vId < max_idx; vId += 1024) {
    float3 delta = {0.0f, 0.0f, 0.0f};
    float sum_i = 0.0f;
    if (vId < vertices_num) {
      delta.x = __ldg(&exp_base[blockIdx.x + (vId * 3 + 0) * col_num]);
      delta.y = __ldg(&exp_base[blockIdx.x + (vId * 3 + 1) * col_num]);
      delta.z = __ldg(&exp_base[blockIdx.x + (vId * 3 + 2) * col_num]);
      if (exp_mask[blockIdx.x + 3 * vId * col_num] < 0.2f) {
        delta = mean_shape[vId] - M33xV3(rotation_i, delta);
        sum_i = 1.0f;
      } else {
        delta = {0.0f, 0.0f, 0.0f};
      }
    }
    const float* delta_ptr = &delta.x;
    block_sum(sum_i, partial_sum);
    __syncthreads();
    if (threadIdx.x == 31) {
      sum += sum_i;
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float d_j = delta_ptr[j];
      block_sum(d_j, partial_sum);
      __syncthreads();
      if (threadIdx.x == 31) {
        translation[blockIdx.x * 3 + j] += d_j;
      }
    }
  }
  if (threadIdx.x == 31) {
    translation[blockIdx.x * 3 + 0] /= sum;
    translation[blockIdx.x * 3 + 1] /= sum;
    translation[blockIdx.x * 3 + 2] /= sum;
  }
}

void cudaComputeTranslationRefineMask(
    pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<float> exp_mask, cudaStream_t stream) {
  const int dim_exp = exp_base.size() / 3 / mean_shape.size();
  dim3 block(1024);
  dim3 grid(dim_exp);
  kernelComputeTranslationRefineMask<<<grid, block, 0, stream>>>(
      translation, rotation, exp_base, mean_shape, exp_mask);
  cudaSafeCall(cudaStreamSynchronize(0));
}

__global__ void kernelComputeTranslation(
    pcl::gpu::PtrSz<float> translation, const pcl::gpu::PtrSz<float> rotation,
    const pcl::gpu::PtrSz<float> exp_base,
    const pcl::gpu::PtrSz<float3> mean_shape,
    const pcl::gpu::PtrSz<unsigned short> rigid_index) {
  __shared__ float partial_sum[32];
  const int rigid_num = rigid_index.size;
  const int max_idx = ((rigid_num + 1023) >> 10) << 10;
  const int col_num = exp_base.size / (3 * mean_shape.size);
  __shared__ float rotation_i[9];
  if (threadIdx.x < 9) {
    rotation_i[threadIdx.x] = rotation[blockIdx.x * 9 + threadIdx.x];
  }
  if (threadIdx.x < 3) {
    translation[threadIdx.x + blockIdx.x * 3] = 0.0f;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < max_idx; i += 1024) {
    float3 delta = {0.0f, 0.0f, 0.0f};
    if (i < rigid_num) {
      const int vId = __ldg(&rigid_index[i]);
      delta.x = __ldg(&exp_base[blockIdx.x + (vId * 3 + 0) * col_num]);
      delta.y = __ldg(&exp_base[blockIdx.x + (vId * 3 + 1) * col_num]);
      delta.z = __ldg(&exp_base[blockIdx.x + (vId * 3 + 2) * col_num]);
      delta = mean_shape[vId] - M33xV3(rotation_i, delta);
    }
    __syncthreads();

    const float* delta_ptr = &delta.x;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float d_j = delta_ptr[j];
      block_sum(d_j, partial_sum);
      __syncthreads();
      if (threadIdx.x == 31) {
        translation[blockIdx.x * 3 + j] += d_j;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 31) {
    translation[blockIdx.x * 3 + 0] /= rigid_index.size;
    translation[blockIdx.x * 3 + 1] /= rigid_index.size;
    translation[blockIdx.x * 3 + 2] /= rigid_index.size;
  }
}

void cudaComputeTranslation(
    pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<unsigned short> rigid_index) {
  const int dim_exp = exp_base.size() / 3 / mean_shape.size();
  dim3 block(1024);
  dim3 grid(dim_exp);
  kernelComputeTranslation<<<grid, block>>>(translation, rotation, exp_base,
                                            mean_shape, rigid_index);
  cudaSafeCall(cudaStreamSynchronize(0));
}

__global__ void kernelComputeSRTBase(pcl::gpu::PtrSz<float> base,
                                     const pcl::gpu::PtrSz<float> rotation,
                                     const pcl::gpu::PtrSz<float> translation,
                                     const pcl::gpu::PtrSz<float3> mean_shape) {
  const auto vId = threadIdx.x + blockDim.x * blockIdx.y;
  const int col_num = base.size / 3 / mean_shape.size;
  __shared__ float rotation_i[9], translation_i[3];

  if (threadIdx.x < 9) {
    rotation_i[threadIdx.x] = rotation[blockIdx.x * 9 + threadIdx.x];
  }
  if (threadIdx.x < 3) {
    translation_i[threadIdx.x] = translation[blockIdx.x * 3 + threadIdx.x];
  }
  __syncthreads();
  if (vId < mean_shape.size) {
    float3 exp_i;
    exp_i.x = base[blockIdx.x + (vId * 3 + 0) * col_num];
    exp_i.y = base[blockIdx.x + (vId * 3 + 1) * col_num];
    exp_i.z = base[blockIdx.x + (vId * 3 + 2) * col_num];
    exp_i = M33xV3(rotation_i, exp_i);
    exp_i.x += translation_i[0];
    exp_i.y += translation_i[1];
    exp_i.z += translation_i[2];
    exp_i -= mean_shape[vId];
    base[blockIdx.x + (vId * 3 + 0) * col_num] = exp_i.x;
    base[blockIdx.x + (vId * 3 + 1) * col_num] = exp_i.y;
    base[blockIdx.x + (vId * 3 + 2) * col_num] = exp_i.z;
  }
}

void cudaComputeSRTBase(pcl::gpu::DeviceArray<float> base,
                        const pcl::gpu::DeviceArray<float> rotation,
                        const pcl::gpu::DeviceArray<float> translation,
                        const pcl::gpu::DeviceArray<float3> mean_shape,
                        cudaStream_t stream) {
  const int dim_exp = base.size() / 3 / mean_shape.size();
  dim3 block(1024);
  dim3 grid(dim_exp, pcl::gpu::divUp(mean_shape.size(), block.x));
  kernelComputeSRTBase<<<grid, block, 0, stream>>>(base, rotation, translation,
                                                   mean_shape);
}

__global__ void kernelComputeATA_diag(pcl::gpu::PtrSz<float> ATA,
                                      pcl::gpu::PtrSz<float> diag,
                                      const pcl::gpu::PtrSz<float> A,
                                      const int m) {
  __shared__ float matA[16][16];
  __shared__ float matB[16][16];
  const int tidc = threadIdx.x;
  const int tidr = threadIdx.y;
  const int bidc = blockIdx.x << 4;
  const int bidr = blockIdx.y << 4;
  if (bidc <= bidr) {
    const int n = A.size / m;
    const int step = m;
    const int Ar = tidr + bidr;
    const int Bc = tidc + bidc;
    int i, j;
    float results = 0.0f;
    float comp = 0.0f;

    for (j = 0; j < n; j += 16) {
      if (Ar < step && tidc + j < n) {
        matA[tidr][tidc] = A[(tidc + j) * step + Ar];
      } else {
        matA[tidr][tidc] = 0.0f;
      }

      if (tidr + j < n && Bc < step) {
        matB[tidr][tidc] = A[(tidr + j) * step + Bc];
      } else {
        matB[tidr][tidc] = 0.0f;
      }

      __syncthreads();

      for (i = 0; i < 16; i++) {
        if (matA[tidr][i] != 0.0f && matB[i][tidc] != 0.0f) {
          float t;
          // results += matA[tidr][i] * matB[i][tidc];
          comp -= matA[tidr][i] * matB[i][tidc];
          t = results - comp;
          comp = (t - results) + comp;
          results = t;
        }
      }

      __syncthreads();
    }

    if (Ar < step && Bc < step) {
      ATA[Ar * step + Bc] = results;
      if (bidc < bidr) {
        ATA[Bc * step + Ar] = results;
      }
      if (Ar == Bc) {
        diag[Ar] = results;
      }
    }
  }
}

void cudaCompuateATA_diag(pcl::gpu::DeviceArray<float> ATA,
                          pcl::gpu::DeviceArray<float> diag,
                          const pcl::gpu::DeviceArray<float> A, const int n,
                          const int m) {
  dim3 block(16, 16);
  dim3 grid(pcl::gpu::divUp(m, block.x), pcl::gpu::divUp(m, block.x));
  kernelComputeATA_diag<<<grid, block>>>(ATA, diag, A, m);
}
#endif  // USE_CUDA