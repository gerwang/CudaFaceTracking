#pragma once
#include "Common.h"
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <pcl\gpu\containers\device_array.h>
#include <pcl\gpu\containers\kernel_containers.h>

#include <iostream>
#include <pcl\gpu\utils\safe_call.hpp>
#include <vector>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "MSFRUtil.cu"

//////////////////////////////////////////////////////////////////////////
// shlf_add and warp_scan for int
//////////////////////////////////////////////////////////////////////////

//__device__ int step_b;
__device__ float dev_error;
//__device__ float dev_weight;
__device__ int dev_Ii_number;
__device__ int dev_Iij_number;
__device__ int dev_terms_number;

__global__ void kernelUpdatePartialAfromLandmark(
    pcl::gpu::PtrSz<float> A, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<int> activate_basis_index,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<int2> landmarkIndex, const pcl::gpu::PtrSz<float> reg,
    const int step_b) {
  __shared__ float matA[8][8];
  __shared__ float matB[8][8];
  const int tidc = threadIdx.x;
  const int tidr = threadIdx.y;
  const int bidc = blockIdx.x << 3;
  const int bidr = blockIdx.y << 3;
  const int Ar = tidr + bidr;
  const int Bc = tidc + bidc;
  const int M = activate_basis_index.size;
  int ArIdx;
  int BcIdx;
  if (Ar < M) {
    ArIdx = __ldg(&activate_basis_index[Ar]);
  }
  if (Bc < M) {
    BcIdx = __ldg(&activate_basis_index[Bc]);
  }
  // if (bidc <= bidr)
  {
    int n = 3 * landmarkIndex.size;
    int landmarks_num = landmarkIndex.size;
    int i, j;
    float results = 0.0f;
    float comp = 0.0f;

    for (int k = 0; k < n; k += 8) {
      int lId = (k + tidc) / 3;
      if (lId < landmarks_num) {
        int coor_id = (k + tidc) % 3;
        int vId = __ldg(&landmarkIndex[lId].y);
        j = 3 * vId + coor_id;
        if (Ar < M) {
          matA[tidr][tidc] = base[j * step_b + ArIdx];
        } else {
          matA[tidr][tidc] = 0.0f;
        }
      } else {
        matA[tidr][tidc] = 0.0f;
      }
      lId = (k + tidr) / 3;
      if (lId < landmarks_num) {
        int coor_id = (k + tidr) % 3;
        int vId = __ldg(&landmarkIndex[lId].y);
        j = 3 * vId + coor_id;
        if (Bc < M && __ldg(&lambda[vId]) != 0.0f) {
          matB[tidr][tidc] = base[j * step_b + BcIdx] * lambda[vId];
        } else {
          matB[tidr][tidc] = 0.0f;
        }
      } else {
        matB[tidr][tidc] = 0.0f;
      }

      __syncthreads();

      for (i = 0; i < 8; i++) {
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

    if (Ar < M && Bc < M) {
      A[(Ar)*M + Bc] += results;
    }
  }
}

__global__ void kernelUpdateAfromLandmark(
    pcl::gpu::PtrSz<float> A, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<int2> landmarkIndex, const pcl::gpu::PtrSz<float> reg,
    const int step_b) {
  __shared__ float matA[16][16];
  __shared__ float matB[16][16];
  const int tidc = threadIdx.x;
  const int tidr = threadIdx.y;
  const int bidc = blockIdx.x << 4;
  const int bidr = blockIdx.y << 4;
  const int Ar = tidr + bidr;
  const int Bc = tidc + bidc;
  // if (bidc <= bidr)
  {
    int n = 3 * landmarkIndex.size;
    int landmarks_num = landmarkIndex.size;
    int i, j;
    float results = 0.0f;
    float comp = 0.0f;

    for (int k = 0; k < n; k += 16) {
      int lId = (k + tidc) / 3;
      if (lId < landmarks_num) {
        int coor_id = (k + tidc) % 3;
        int vId = __ldg(&landmarkIndex[lId].y);
        j = 3 * vId + coor_id;
        if (Ar < step_b) {
          matA[tidr][tidc] = base[j * step_b + Ar];
        } else {
          matA[tidr][tidc] = 0.0f;
        }
      } else {
        matA[tidr][tidc] = 0.0f;
      }
      lId = (k + tidr) / 3;
      if (lId < landmarks_num) {
        int coor_id = (k + tidr) % 3;
        int vId = __ldg(&landmarkIndex[lId].y);
        j = 3 * vId + coor_id;
        if (Bc < step_b && __ldg(&lambda[vId]) != 0.0f) {
          matB[tidr][tidc] = base[j * step_b + Bc] * lambda[vId];
        } else {
          matB[tidr][tidc] = 0.0f;
        }
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

    if (Ar < step_b && Bc < step_b) {
      A[(Ar)*step_b + Bc] += results;
    }
  }
}

__global__ void kernelUpdateAfromLandmarkPoint2Line(
    pcl::gpu::PtrSz<float> A, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float> target_position_inv_sRT,
    const pcl::gpu::PtrSz<int2> landmarkIndex, const pcl::gpu::PtrSz<float> reg,
    const pcl::gpu::PtrSz<float> rotation,
    const pcl::gpu::PtrSz<float> translation, const int step_b) {
  __shared__ float3 camera_pos;
  __shared__ float matA[8][8];
  __shared__ float matB[8][8];
  const int tidc = threadIdx.x;
  const int tidr = threadIdx.y;
  const int bidc = blockIdx.x << 3;
  const int bidr = blockIdx.y << 3;
  const int Ar = tidr + bidr;
  const int Bc = tidc + bidc;
  if (tidc == 0 && tidr == 0) {
    camera_pos = M33TxV3(rotation, make_float3(-translation[0], -translation[1],
                                               -translation[2]) /
                                       translation[3]);
  }
  __syncthreads();
  // if (bidc <= bidr)
  {
    int n = 3 * landmarkIndex.size;
    int landmarks_num = landmarkIndex.size;
    int i, j;
    float results = 0.0f;
    float comp = 0.0f;

    for (int k = 0; k < n; k += 8) {
      int lId = (k + tidc) / 3;
      if (lId < landmarks_num) {
        int coor_id = (k + tidc) % 3;
        int vId = __ldg(&landmarkIndex[lId].y);
        j = 3 * vId;
        float lambda_i = __ldg(&lambda[vId]);
        if (Ar < step_b && lambda_i != 0.0f) {
          float3 delta = target_position_inv_sRT[vId] / lambda_i - camera_pos;
          delta /= length(delta);
          float3 v_i = make_float3(__ldg(&base[j * step_b + Ar]),
                                   __ldg(&base[(j + 1) * step_b + Ar]),
                                   __ldg(&base[(j + 2) * step_b + Ar]));
          float alpha = dot(delta, v_i);
          matA[tidr][tidc] =
              base[(j + coor_id) * step_b + Ar] - alpha * (&delta.x)[coor_id];
        } else {
          matA[tidr][tidc] = 0.0f;
        }
      } else {
        matA[tidr][tidc] = 0.0f;
      }
      lId = (k + tidr) / 3;
      if (lId < landmarks_num) {
        int coor_id = (k + tidr) % 3;
        int vId = __ldg(&landmarkIndex[lId].y);
        j = 3 * vId + coor_id;
        float lambda_i = __ldg(&lambda[vId]);
        if (Bc < step_b && lambda_i != 0.0f) {
          matB[tidr][tidc] = base[j * step_b + Bc] * lambda_i;
        } else {
          matB[tidr][tidc] = 0.0f;
        }
      } else {
        matB[tidr][tidc] = 0.0f;
      }

      __syncthreads();

      for (i = 0; i < 8; i++) {
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

    if (Ar < step_b && Bc < step_b) {
      A[(Ar)*step_b + Bc] += results;
    }
  }
}

__global__ void kernelUpdatePartialAfromLambda_II(
    pcl::gpu::PtrSz<float> A, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<int> activate_basis_index,
    const pcl::gpu::PtrSz<float> lambda, const pcl::gpu::PtrSz<float> reg,
    const float reg_lambda, const int step_b) {
  __shared__ float matA[8][8];
  __shared__ float matB[8][8];
  const int tidc = threadIdx.x;
  const int tidr = threadIdx.y;
  const int bidc = blockIdx.x << 3;
  const int bidr = blockIdx.y << 3;
  const int Ar = tidr + bidr;
  const int Bc = tidc + bidc;
  const int M = activate_basis_index.size;
  int ArIdx;
  int BcIdx;
  if (Ar < M) {
    ArIdx = __ldg(&activate_basis_index[Ar]);
  }
  if (Bc < M) {
    BcIdx = __ldg(&activate_basis_index[Bc]);
  }
  // if (bidc <= bidr)
  {
    int n = base.size / step_b;
    int i, j;
    float results = 0.0f;
    float comp = 0.0f;

    for (j = 0; j < n; j += 8) {
      if (Ar < M && tidc + j < n) {
        matA[tidr][tidc] = base[(tidc + j) * step_b + ArIdx];
      } else {
        matA[tidr][tidc] = 0.0f;
      }

      if (tidr + j < n && Bc < M && __ldg(&lambda[(tidr + j) / 3]) != 0.0f) {
        matB[tidr][tidc] =
            base[(tidr + j) * step_b + BcIdx] * lambda[(tidr + j) / 3];
      } else {
        matB[tidr][tidc] = 0.0f;
      }

      __syncthreads();

      for (i = 0; i < 8; i++) {
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

    if (Ar < M && Bc < M) {
      A[(Ar)*M + Bc] = results;
      /*if (bidc < bidr)
      {
      A[(tidc + bidc) * step_b + tidr + bidr] = A[(tidr + bidr) * step_b + tidc
      + bidc];
      }*/
      if (Ar == Bc) {
        A[(Ar)*M + Ar] += reg_lambda * reg[ArIdx];
      }
    }
  }
}

__global__ void kernelUpdateAfromLambda_II(pcl::gpu::PtrSz<float> A,
                                           const pcl::gpu::PtrSz<float> base,
                                           const pcl::gpu::PtrSz<float> lambda,
                                           const pcl::gpu::PtrSz<float> reg,
                                           const float reg_lambda,
                                           const int step_b) {
  __shared__ float matA[16][16];
  __shared__ float matB[16][16];
  const int tidc = threadIdx.x;
  const int tidr = threadIdx.y;
  const int bidc = blockIdx.x << 4;
  const int bidr = blockIdx.y << 4;
  // if (bidc <= bidr)
  {
    int n = base.size / step_b;
    int i, j;
    float results = 0.0f;
    float comp = 0.0f;

    for (j = 0; j < n; j += 16) {
      if (tidr + bidr < step_b && tidc + j < n) {
        matA[tidr][tidc] = base[(tidc + j) * step_b + tidr + bidr];
      } else {
        matA[tidr][tidc] = 0.0f;
      }

      if (tidr + j < n && tidc + bidc < step_b &&
          __ldg(&lambda[(tidr + j) / 3]) != 0.0f) {
        matB[tidr][tidc] =
            base[(tidr + j) * step_b + tidc + bidc] * lambda[(tidr + j) / 3];
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

    if (tidr + bidr < step_b && tidc + bidc < step_b) {
      A[(tidr + bidr) * step_b + tidc + bidc] = results;
      /*if (bidc < bidr)
      {
        A[(tidc + bidc) * step_b + tidr + bidr] = A[(tidr + bidr) * step_b +
      tidc + bidc];
      }*/
      if (tidr + bidr == tidc + bidc) {
        A[(tidr + bidr) * step_b + tidr + bidr] +=
            reg_lambda * reg[tidr + bidr];
      }
    }
  }
}

__global__ void kernelUpdateAfromLambdaTarget(
    pcl::gpu::PtrSz<float> A, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<float> lambda, const float reg_lambda) {
  int col = blockIdx.x;
  int row = blockIdx.y;

  if (row <= col) {
    __shared__ float partial_sum[32];

    int base_rows = base.size / gridDim.x;
    int threadDims = (base_rows + 1023) >> 10;
    float result = 0;
    {
      int beginId = threadDims * threadIdx.x;
      int endId = min(threadDims + beginId, base_rows);
      for (int i = beginId, offset = gridDim.x * beginId; i < endId;
           ++i, offset += gridDim.x) {
        result += base[row + offset] * base[col + offset] * lambda[i];
      }
    }

    block_sum(result, partial_sum);

    if (threadIdx.x == 31) {
      A[row * gridDim.x + col] = result;
      A[col * gridDim.x + row] = result;
      if (row == col) {
        A[row * gridDim.x + col] += reg_lambda;
      }
    }
  }
}

__global__ void kernelUnRTTargetPosition(
    pcl::gpu::PtrSz<float3> unRT_position, const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position_inv_sRT,
    const pcl::gpu::PtrSz<float3> position) {
  int vId = blockIdx.x * blockDim.x + threadIdx.x;
  if (vId < position.size) {
    unRT_position[vId] =
        target_position_inv_sRT[vId] - position[vId] * lambda[vId];
  }
}

__global__ void kernelUnRTTargetPositionLandmark(
    pcl::gpu::PtrSz<float3> unRT_position, const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position_inv_sRT,
    const pcl::gpu::PtrSz<float3> position,
    const pcl::gpu::PtrSz<int2> landmarkIndex) {
  int lId = blockIdx.x * blockDim.x + threadIdx.x;
  if (lId < landmarkIndex.size) {
    int vId = __ldg(&landmarkIndex[lId].y);
    unRT_position[vId] =
        target_position_inv_sRT[vId] - position[vId] * lambda[vId];
  }
}

__global__ void kernelUnRTTargetPositionLandmarkPoint2Line(
    pcl::gpu::PtrSz<float3> unRT_position, const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position_inv_sRT,
    const pcl::gpu::PtrSz<float3> position,
    const pcl::gpu::PtrSz<int2> landmarkIndex,
    const pcl::gpu::PtrSz<float> rotation,
    const pcl::gpu::PtrSz<float> translation) {
  __shared__ float3 camera_pos;
  int lId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x) {
    camera_pos = M33TxV3(rotation, make_float3(-translation[0], -translation[1],
                                               -translation[2]) /
                                       translation[3]);
  }
  __syncthreads();
  if (lId < landmarkIndex.size) {
    int vId = __ldg(&landmarkIndex[lId].y);
    float lambda_i = __ldg(&lambda[vId]);
    if (lambda_i != 0.0f) {
      float3 delta_t = target_position_inv_sRT[vId] / lambda_i - camera_pos;
      delta_t /= length(delta_t);
      float3 delta_p = camera_pos - position[vId];
      unRT_position[vId] =
          (delta_p - dot(delta_p, delta_t) * delta_t) * lambda_i;
    } else {
      unRT_position[vId] = {0.0f, 0.0f, 0.0f};
    }
  }
}

__global__ void kernelUpdatePartial_b_fromLambdaTarget(
    pcl::gpu::PtrSz<float> b, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<int> activate_basis_index,
    const pcl::gpu::PtrSz<float3> target_position_,
    const pcl::gpu::PtrSz<float> coeff, const pcl::gpu::PtrSz<float> reg,
    const float reg_lambda, const int step_b) {
  const float *target_position = &target_position_[0].x;
  __shared__ float partial_sum[32];

  int base_rows = base.size / step_b;
  int threadDims = (base_rows + 1023) >> 10;

  int row = __ldg(&activate_basis_index[blockIdx.x]);
  float result = 0;
  float comp = 0;
  {
    int beginId = threadDims * threadIdx.x;
    int endId = min(threadDims + beginId, base_rows);
    for (int i = beginId, offset = step_b * beginId + row; i < endId;
         ++i, offset += step_b) {
      float t;
      comp -= base[offset] * target_position[i];
      t = result - comp;
      comp = (t - result) + comp;
      result = t;
    }
  }
  block_sum(result, partial_sum);

  if (threadIdx.x == 31) {
    b[blockIdx.x] = result - reg_lambda * reg[row] * coeff[row];
  }
}

__global__ void kernelUpdate_b_fromSampledLambdaTarget(
    pcl::gpu::PtrSz<float> b, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<float3> target_position_,
    const pcl::gpu::PtrSz<float> coeff, const pcl::gpu::PtrSz<float> reg,
    const pcl::gpu::PtrSz<int> sampled_key, const float reg_lambda,
    const int step_b) {
  const float *target_position = &target_position_[0].x;
  __shared__ float partial_sum[32];

  const int base_rows = sampled_key.size;
  const int threadDims = (base_rows + 1023) >> 10;

  const int row = blockIdx.x;
  float result = 0;
  float comp = 0;
  {
    int beginId = threadDims * threadIdx.x;
    int endId = min(threadDims + beginId, base_rows);
    for (int i = beginId, offset = step_b * beginId + row; i < endId;
         ++i, offset += step_b) {
      float t;
      comp -= base[offset] * target_position[i];
      t = result - comp;
      comp = (t - result) + comp;
      result = t;
    }
  }
  block_sum(result, partial_sum);

  if (threadIdx.x == 31) {
    b[row] = result - reg_lambda * reg[row] * coeff[row];
  }
}

__global__ void kernelUpdate_b_fromLambdaTarget(
    pcl::gpu::PtrSz<float> b, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<float3> target_position_,
    const pcl::gpu::PtrSz<float> coeff, const pcl::gpu::PtrSz<float> reg,
    const float reg_lambda, const int step_b) {
  const float *target_position = &target_position_[0].x;
  __shared__ float partial_sum[32];

  int base_rows = base.size / step_b;
  int threadDims = (base_rows + 1023) >> 10;

  int row = blockIdx.x;
  float result = 0;
  float comp = 0;
  {
    int beginId = threadDims * threadIdx.x;
    int endId = min(threadDims + beginId, base_rows);
    for (int i = beginId, offset = step_b * beginId + row; i < endId;
         ++i, offset += step_b) {
      float t;
      comp -= base[offset] * target_position[i];
      t = result - comp;
      comp = (t - result) + comp;
      result = t;
    }
  }
  block_sum(result, partial_sum);

  if (threadIdx.x == 31) {
    b[row] = result - reg_lambda * reg[row] * coeff[row];
  }
}

__global__ void kernelUpdatePartial_b_fromLandmark(
    pcl::gpu::PtrSz<float> b, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<int> activate_basis_index,
    const pcl::gpu::PtrSz<float3> target_position_,
    const pcl::gpu::PtrSz<int2> landmarkIndex,
    const pcl::gpu::PtrSz<float> coeff, const pcl::gpu::PtrSz<float> reg,
    const int step_b) {
  const float *target_position = &target_position_[0].x;
  __shared__ float partial_sum[32];

  int n = landmarkIndex.size * 3;
  int threadDims = (n + 1023) >> 10;

  int row = __ldg(&activate_basis_index[blockIdx.x]);
  float result = 0;
  float comp = 0;
  {
    int beginId = threadDims * threadIdx.x;
    int endId = min(threadDims + beginId, n);
    for (int i = beginId; i < endId; ++i) {
      int lId = i / 3;
      int vId = __ldg(&landmarkIndex[lId].y);
      int coor_id = i % 3;
      int offset = 3 * vId + coor_id;
      float t;
      comp -= base[offset * step_b + row] * target_position[offset];
      t = result - comp;
      comp = (t - result) + comp;
      result = t;
    }
  }
  block_sum(result, partial_sum);

  if (threadIdx.x == 31) {
    b[blockIdx.x] += result;
  }
}

__global__ void kernelUpdate_b_fromLandmark(
    pcl::gpu::PtrSz<float> b, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<float3> target_position_,
    const pcl::gpu::PtrSz<int2> landmarkIndex,
    const pcl::gpu::PtrSz<float> coeff, const pcl::gpu::PtrSz<float> reg,
    const int step_b) {
  const float *target_position = &target_position_[0].x;
  __shared__ float partial_sum[32];

  int n = landmarkIndex.size * 3;
  int threadDims = (n + 1023) >> 10;

  int row = blockIdx.x;
  float result = 0;
  float comp = 0;
  {
    int beginId = threadDims * threadIdx.x;
    int endId = min(threadDims + beginId, n);
    for (int i = beginId; i < endId; ++i) {
      int lId = i / 3;
      int vId = __ldg(&landmarkIndex[lId].y);
      int coor_id = i % 3;
      int offset = 3 * vId + coor_id;
      float t;
      comp -= base[offset * step_b + row] * target_position[offset];
      t = result - comp;
      comp = (t - result) + comp;
      result = t;
    }
  }
  block_sum(result, partial_sum);

  if (threadIdx.x == 31) {
    b[row] += result;
  }
}

void cudaUpdateAbfromLambdaTarget(
    pcl::gpu::DeviceArray<float> A, pcl::gpu::DeviceArray<float> b,
    pcl::gpu::DeviceArray<float3> unRT_target_position,
    const pcl::gpu::DeviceArray<float> base,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position_inv_sRT,
    const pcl::gpu::DeviceArray<float3> position,
    const pcl::gpu::DeviceArray<float> coeff,
    const pcl::gpu::DeviceArray<float> reg, const float reg_lambda) {
  // dim3 block_A(1024);
  // dim3 grid_A(b.size(), b.size());
  dim3 block_A(16, 16);
  dim3 grid_A(pcl::gpu::divUp(b.size(), block_A.x),
              pcl::gpu::divUp(b.size(), block_A.y));
  // cudaSafeCall(cudaMemcpyToSymbol(dev_reg_lambda, &reg_lambda,
  // sizeof(float)));
  int b_size = b.size();
  // cudaSafeCall(cudaMemcpyToSymbol(step_b, &b_size, sizeof(int)));
  // kernelUpdateAfromLambdaTarget<<<grid_A, block_A >>>(A, base, lambda);
  kernelUpdateAfromLambda_II<<<grid_A, block_A>>>(A, base, lambda, reg,
                                                  reg_lambda, b_size);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  dim3 block(1024);
  dim3 grid_target_position(
      pcl::gpu::divUp(target_position_inv_sRT.size(), block.x));
  // pcl::gpu::DeviceArray<float3>
  // unRT_target_position(target_position_inv_sRT.size());
  kernelUnRTTargetPosition<<<grid_target_position, block>>>(
      unRT_target_position, lambda, target_position_inv_sRT, position);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif

  dim3 grid_b(b.size());
  kernelUpdate_b_fromLambdaTarget<<<grid_b, block>>>(
      b, base, unRT_target_position, coeff, reg, reg_lambda, b_size);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

void cudaUpdatePartialAbfromLambdaTarget(
    pcl::gpu::DeviceArray<float> A, pcl::gpu::DeviceArray<float> b,
    pcl::gpu::DeviceArray<float3> unRT_target_position,
    const pcl::gpu::DeviceArray<float> base,
    const pcl::gpu::DeviceArray<int> activate_base_index,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position_inv_sRT,
    const pcl::gpu::DeviceArray<float3> position,
    const pcl::gpu::DeviceArray<float> coeff,
    const pcl::gpu::DeviceArray<float> reg, const float reg_lambda) {
  // dim3 block_A(1024);
  // dim3 grid_A(b.size(), b.size());
  dim3 block_A(8, 8);
  dim3 grid_A(pcl::gpu::divUp(b.size(), block_A.x),
              pcl::gpu::divUp(b.size(), block_A.y));
  // cudaSafeCall(cudaMemcpyToSymbol(dev_reg_lambda, &reg_lambda,
  // sizeof(float)));
  int step_b = base.size() / 3 / position.size();
  // cudaSafeCall(cudaMemcpyToSymbol(step_b, &b_size, sizeof(int)));
  // kernelUpdateAfromLambdaTarget<<<grid_A, block_A >>>(A, base, lambda);
  kernelUpdatePartialAfromLambda_II<<<grid_A, block_A>>>(
      A, base, activate_base_index, lambda, reg, reg_lambda, step_b);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  dim3 block(1024);
  dim3 grid_target_position(
      pcl::gpu::divUp(target_position_inv_sRT.size(), block.x));
  // pcl::gpu::DeviceArray<float3>
  // unRT_target_position(target_position_inv_sRT.size());
  kernelUnRTTargetPosition<<<grid_target_position, block>>>(
      unRT_target_position, lambda, target_position_inv_sRT, position);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif

  dim3 grid_b(b.size());
  kernelUpdatePartial_b_fromLambdaTarget<<<grid_b, block>>>(
      b, base, activate_base_index, unRT_target_position, coeff, reg,
      reg_lambda, step_b);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelUpdateAbfromTemporalSmoothness(
    pcl::gpu::PtrSz<float> A, pcl::gpu::PtrSz<float> b,
    const pcl::gpu::PtrSz<float> present, const pcl::gpu::PtrSz<float> prev_1,
    const pcl::gpu::PtrSz<float> prev_2, const float smooth_lambda) {
  const int index = threadIdx.x;
  const int step_b = b.size;
  if (index < step_b) {
    A[index * step_b + index] += smooth_lambda;
    b[index] +=
        smooth_lambda * (2 * __ldg(&prev_1[index]) - __ldg(&prev_2[index]) -
                         __ldg(&present[index]));
  }
}

void cudaUpdateAbfromTemporalSmoothness(
    pcl::gpu::DeviceArray<float> A, pcl::gpu::DeviceArray<float> b,
    const pcl::gpu::DeviceArray<float> present,
    const pcl::gpu::DeviceArray<float> prev_1,
    const pcl::gpu::DeviceArray<float> prev_2, const float smooth_lambda) {
  float sqrt_lambda = sqrtf(smooth_lambda);

  dim3 block(b.size());
  dim3 grid(1);
  kernelUpdateAbfromTemporalSmoothness<<<grid, block>>>(A, b, present, prev_1,
                                                        prev_2, sqrt_lambda);
}

__global__ void kernelClamp(pcl::gpu::PtrSz<float> x) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < x.size) {
    float x_i = x[idx];
    if (x_i > 1.0f) {
      x[idx] = 1.0f;
    } else if (x_i < 0.f) {
      x[idx] = 0.f;
    }
  }
}

void cudaClamp(pcl::gpu::DeviceArray<float> x) {
  dim3 block(32);
  dim3 grid(pcl::gpu::divUp(x.size(), block.x));
  kernelClamp<<<grid, block>>>(x);
}

void cudaUpdateAbfromLandmarkTargetPoint2Line(
    pcl::gpu::DeviceArray<float> A, pcl::gpu::DeviceArray<float> b,
    pcl::gpu::DeviceArray<float3> unRT_target_position,
    const pcl::gpu::DeviceArray<float> base,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position_inv_sRT,
    const pcl::gpu::DeviceArray<float3> position,
    const pcl::gpu::DeviceArray<int2> landmarkIndex,
    const pcl::gpu::DeviceArray<float> coeff,
    const pcl::gpu::DeviceArray<float> reg,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> translation) {
  dim3 block_A(8, 8);
  dim3 grid_A(pcl::gpu::divUp(b.size(), block_A.x),
              pcl::gpu::divUp(b.size(), block_A.y));
  // cudaSafeCall(cudaMemcpyToSymbol(dev_reg_lambda, &reg_lambda,
  // sizeof(float)));
  int b_size = b.size();
  // cudaSafeCall(cudaMemcpyToSymbol(step_b, &b_size, sizeof(int)));
  // kernelUpdateAfromLambdaTarget<<<grid_A, block_A >>>(A, base, lambda);
  kernelUpdateAfromLandmarkPoint2Line<<<grid_A, block_A>>>(
      A, base, lambda, target_position_inv_sRT, landmarkIndex, reg, rotation,
      translation, b_size);
  dim3 block(1024);
  dim3 grid_target_position(pcl::gpu::divUp(landmarkIndex.size(), block.x));
  // pcl::gpu::DeviceArray<float3>
  // unRT_target_position(target_position_inv_sRT.size());
  kernelUnRTTargetPositionLandmarkPoint2Line<<<grid_target_position, block>>>(
      unRT_target_position, lambda, target_position_inv_sRT, position,
      landmarkIndex, rotation, translation);
  dim3 grid_b(b.size());
  kernelUpdate_b_fromLandmark<<<grid_b, block>>>(
      b, base, unRT_target_position, landmarkIndex, coeff, reg, b_size);
}

__global__ void kernelAddDelta(pcl::gpu::PtrSz<float> v,
                               const pcl::gpu::PtrSz<float> delta) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < v.size) {
    v[id] += delta[id];
  }
}

void cudaAddDelta(pcl::gpu::DeviceArray<float> v,
                  const pcl::gpu::DeviceArray<float> delta) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(v.size(), block.x));
  kernelAddDelta<<<grid, block>>>(v, delta);
}

__global__ void kernelAddDeltaPartial(
    pcl::gpu::PtrSz<float> v, const pcl::gpu::PtrSz<float> delta,
    const pcl::gpu::PtrSz<int> activate_basis_index) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < delta.size) {
    v[__ldg(&activate_basis_index[id])] += __ldg(&delta[id]);
  }
}

void cudaAddDeltaPartial(
    pcl::gpu::DeviceArray<float> v, const pcl::gpu::DeviceArray<float> delta,
    const pcl::gpu::DeviceArray<int> activate_basis_index) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(delta.size(), block.x));
  kernelAddDeltaPartial<<<grid, block>>>(v, delta, activate_basis_index);
}

__global__ void kernelUpdateM_intermediateforRT(
    pcl::gpu::PtrSz<float> M_intermediate, const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position,
    const pcl::gpu::PtrSz<float> mu_target_mu) {
  __shared__ float3 mu, target_mu;
  if (threadIdx.x == 0) {
    mu.x = mu_target_mu[0];
    mu.y = mu_target_mu[1];
    mu.z = mu_target_mu[2];
    target_mu.x = mu_target_mu[3];
    target_mu.y = mu_target_mu[4];
    target_mu.z = mu_target_mu[5];
  }
  __syncthreads();
  __shared__ float partial_sum[32];
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float *M_i = &M_intermediate[blockIdx.x * 9];
  float3 M[3] = {0};
  if (id < position.size) {
    if (lambda[id] > 0.0f) {
      float3 target_position_Mu = target_position[id] - lambda[id] * target_mu;
      float3 position_Mu = position[id] - mu;
      M[0] = position_Mu.x * target_position_Mu;
      M[1] = position_Mu.y * target_position_Mu;
      M[2] = position_Mu.z * target_position_Mu;
    }
  }
  block_sum(M[0].x, partial_sum);
  block_sum(M[0].y, partial_sum);
  block_sum(M[0].z, partial_sum);
  block_sum(M[1].x, partial_sum);
  block_sum(M[1].y, partial_sum);
  block_sum(M[1].z, partial_sum);
  block_sum(M[2].x, partial_sum);
  block_sum(M[2].y, partial_sum);
  block_sum(M[2].z, partial_sum);
  if (threadIdx.x == 31) {
    M_i[0] = M[0].x;
    M_i[1] = M[0].y;
    M_i[2] = M[0].z;
    M_i[3] = M[1].x;
    M_i[4] = M[1].y;
    M_i[5] = M[1].z;
    M_i[6] = M[2].x;
    M_i[7] = M[2].y;
    M_i[8] = M[2].z;
  }
}

__global__ void kernelUpdateScale(pcl::gpu::PtrSz<float> T,
                                  const pcl::gpu::PtrSz<float> lambda,
                                  const pcl::gpu::PtrSz<float3> target_position,
                                  const pcl::gpu::PtrSz<float3> position) {
  __shared__ float partial_sum[32];
  __shared__ float3 mu, target_mu;
  int threadDim = (position.size + 1023) >> 10;
  int beginId = threadIdx.x * threadDim;
  int endId = min(beginId + threadDim, (int)position.size);

  float mu_x = 0, mu_y = 0, mu_z = 0;
  float target_mu_x = 0, target_mu_y = 0, target_mu_z = 0;
  float lambda_i = 0;
  for (int i = beginId; i < endId; ++i) {
    if (lambda[i] > 0.0f) {
      mu_x += position[i].x * lambda[i];
      mu_y += position[i].y * lambda[i];
      mu_z += position[i].z * lambda[i];
      target_mu_x += target_position[i].x;
      target_mu_y += target_position[i].y;
      target_mu_z += target_position[i].z;
      lambda_i += lambda[i];
    }
  }
  block_sum(mu_x, partial_sum);
  block_sum(mu_y, partial_sum);
  block_sum(mu_z, partial_sum);
  block_sum(target_mu_x, partial_sum);
  block_sum(target_mu_y, partial_sum);
  block_sum(target_mu_z, partial_sum);
  block_sum(lambda_i, partial_sum);
  if (threadIdx.x == 31) {
    mu.x = mu_x;
    mu.y = mu_y;
    mu.z = mu_z;
    target_mu.x = target_mu_x;
    target_mu.y = target_mu_y;
    target_mu.z = target_mu_z;
    mu /= lambda_i;
    target_mu /= lambda_i;
  }
  __syncthreads();

  float b_i, a_i = 0;
  for (int i = beginId; i < endId; ++i) {
    if (lambda[i] > 0.0f) {
      b_i = norm2(target_position[i] - lambda[i] * target_mu);
      a_i = norm2(lambda[i] * (position[i] - mu));
    }
  }
  block_sum(a_i, partial_sum);
  block_sum(b_i, partial_sum);

  if (threadIdx.x == 31) {
    T[3] = sqrtf(b_i / a_i);
  }
}

__global__ void kernelUpdateMforRT(
    pcl::gpu::PtrSz<float> M, const pcl::gpu::PtrSz<float> M_intermediate) {
  __shared__ float partial_sum[32];
  int M_intermediate_size = M_intermediate.size / 9;
  float Mii = 0.0f;
  if (threadIdx.x < M_intermediate_size) {
    Mii = M_intermediate[threadIdx.x * 9 + blockIdx.x];
  }
  block_sum(Mii, partial_sum);
  if (threadIdx.x == 31) {
    M[blockIdx.x] = Mii;
  }
}

void cudaUpdateScale(pcl::gpu::DeviceArray<float> T,
                     const pcl::gpu::DeviceArray<float> lambda,
                     const pcl::gpu::DeviceArray<float3> target_position,
                     const pcl::gpu::DeviceArray<float3> position) {
  kernelUpdateScale<<<1, 1024>>>(T, lambda, target_position, position);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelUpdateMu(pcl::gpu::PtrSz<float> mu_target_mu,
                               const pcl::gpu::PtrSz<float> lambda,
                               const pcl::gpu::PtrSz<float3> target_position,
                               const pcl::gpu::PtrSz<float3> position) {
  __shared__ float partial_sum[32];
  __shared__ float3 mu, target_mu;
  int threadDim = (position.size + 1023) >> 10;
  int beginId = threadIdx.x * threadDim;
  int endId = min(beginId + threadDim, (int)position.size);

  float mu_x = 0, mu_y = 0, mu_z = 0;
  float target_mu_x = 0, target_mu_y = 0, target_mu_z = 0;
  float lambda_i = 0;
  for (int i = beginId; i < endId; ++i) {
    if (lambda[i] > 0.0f) {
      mu_x += position[i].x * lambda[i];
      mu_y += position[i].y * lambda[i];
      mu_z += position[i].z * lambda[i];
      target_mu_x += target_position[i].x;
      target_mu_y += target_position[i].y;
      target_mu_z += target_position[i].z;
      lambda_i += lambda[i];
    }
  }
  block_sum(mu_x, partial_sum);
  block_sum(mu_y, partial_sum);
  block_sum(mu_z, partial_sum);
  block_sum(target_mu_x, partial_sum);
  block_sum(target_mu_y, partial_sum);
  block_sum(target_mu_z, partial_sum);
  block_sum(lambda_i, partial_sum);
  if (threadIdx.x == 31) {
    mu.x = mu_x;
    mu.y = mu_y;
    mu.z = mu_z;
    target_mu.x = target_mu_x;
    target_mu.y = target_mu_y;
    target_mu.z = target_mu_z;
    mu /= lambda_i;
    target_mu /= lambda_i;
    mu_target_mu[0] = mu.x;
    mu_target_mu[1] = mu.y;
    mu_target_mu[2] = mu.z;
    mu_target_mu[3] = target_mu.x;
    mu_target_mu[4] = target_mu.y;
    mu_target_mu[5] = target_mu.z;
  }
}

void cudaUpdateMforRWithoutICP(
    pcl::gpu::DeviceArray<float> M, const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position,
    const pcl::gpu::DeviceArray<float3> position) {
  pcl::gpu::DeviceArray<float> mu_target_mu(6);
  kernelUpdateMu<<<1, 1024>>>(mu_target_mu, lambda, target_position, position);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif

  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position.size(), block.x));
  pcl::gpu::DeviceArray<float> M_intermediate(grid.x * 9);
  kernelUpdateM_intermediateforRT<<<grid, block>>>(
      M_intermediate, lambda, target_position, position, mu_target_mu);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  kernelUpdateMforRT<<<9, 1024>>>(M, M_intermediate);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelUpdateSRTwithNormal(
    pcl::gpu::PtrSz<float> JTJ, pcl::gpu::PtrSz<float> Jr,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position_sRT,
    const pcl::gpu::PtrSz<float3> normal_R,
    const pcl::gpu::PtrSz<float> translation) {
  __shared__ float partial_sum[32];  // shared_JTJ[36], shared_Jr[6];
  __shared__ float3 trans;
  __shared__ float scale_inv;
  int tId = threadIdx.x;
  if (tId == 0) {
    trans = *((float3 *)(&translation[0]));
    scale_inv = 1.0f / translation[3];
  }
  __syncthreads();
  int threadDim = (position_sRT.size + 1023) >> 10;
  float J_j[7];  // threadJTJ[36] = { 0.0f }, threadJr[6] = { 0.0f };
  float r;
  for (int i = 0, j = tId; i < threadDim; ++i, j += 1024) {
    float lambda_j = __ldg(&lambda[j]);
    if (j < position_sRT.size && lambda_j > 0.0f) {
      float weight = sqrtf(lambda_j);
      float3 nt = normal_R[j];
      float3 wv = position_sRT[j];
      float3 tv = target_position[j] / lambda_j;

      J_j[0] = weight * (-nt.y * wv.z + nt.z * wv.y);
      J_j[1] = weight * (nt.x * wv.z - nt.z * wv.x);
      J_j[2] = weight * (-nt.x * wv.y + nt.y * wv.x);
      // A_values[A_index] = weight*(-nt.y*tv.z + nt.z*tv.y);
      // A_values[A_index + 1] = weight*(nt.x*tv.z - nt.z*tv.x);
      // A_values[A_index + 2] = weight*(-nt.x*tv.y + nt.y*tv.x);
      J_j[3] = weight * nt.x;
      J_j[4] = weight * nt.y;
      J_j[5] = weight * nt.z;
      J_j[6] = weight * dot(nt, wv - trans) * scale_inv;
      r = weight * dot(nt, tv - wv);
    } else {
      J_j[0] = 0.0f;
      J_j[1] = 0.0f;
      J_j[2] = 0.0f;
      J_j[3] = 0.0f;
      J_j[4] = 0.0f;
      J_j[5] = 0.0f;
      J_j[6] = 0.0f;
      r = 0.0f;
    }
    // for (int u = 0; u < 6; ++u)
    //{
    //  for (int v = 0; v < 6; ++v)
    //  {
    //    threadJTJ[6 * u + v] += J_j[u] * J_j[v];
    //  }
    //  threadJr[u] += J_j[u] * r;
    //}
    int cnt = 0;
    for (int u = 0; u < 7; ++u) {
      for (int v = 0; v < 7; ++v) {
        float Jr_uv = J_j[u] * J_j[v];
        block_sum(Jr_uv, partial_sum);
        if (tId == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_j[u] * r;
      block_sum(Jr_u, partial_sum);
      if (tId == 31) {
        Jr[u] += Jr_u;
      }
    }
  }
}

__global__ void kernelUpdateSRTwithLandmark(
    pcl::gpu::PtrSz<float> JTJ, pcl::gpu::PtrSz<float> Jr,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position_sRT,
    const pcl::gpu::PtrSz<float3> normal_R,
    const pcl::gpu::PtrSz<int2> landmarkIndex,
    const pcl::gpu::PtrSz<float> translation) {
  __shared__ float partial_sum[32];
  __shared__ float3 trans;
  __shared__ float scale_inv;
  if (threadIdx.x == 0) {
    trans = *((float3 *)(&translation[0]));
    scale_inv = 1.0f / translation[3];
  }
  __syncthreads();
  int lId = threadIdx.x + blockIdx.x * blockDim.x;
  {
    int vId;
    float lambda_i;
    float3 target_pos_i;
    float3 pos_i;
    if (lId < landmarkIndex.size) {
      vId = __ldg(&landmarkIndex[lId].y);
      lambda_i = sqrtf(__ldg(&lambda[vId]));
      target_pos_i = target_position[vId] / lambda_i;
      pos_i = lambda_i * position_sRT[vId];
    }
    float J_i[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, r = 0.0f;
    if (lambda_i > 0.0f) {
      J_i[0] = 0.0f;
      J_i[1] = pos_i.z;
      J_i[2] = -pos_i.y;
      J_i[3] = lambda_i;
      J_i[4] = 0.0f;
      J_i[5] = 0.0f;
      J_i[6] = (pos_i.x - lambda_i * trans.x) * scale_inv;
      r = target_pos_i.x - pos_i.x;
    }
    int cnt = 0;
#pragma unroll
    for (int u = 0; u < 7; ++u) {
#pragma unroll
      for (int v = 0; v < 7; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
    if (lambda_i > 0.0f) {
      J_i[0] = -pos_i.z;
      J_i[1] = 0.0f;
      J_i[2] = pos_i.x;
      J_i[3] = 0.0f;
      J_i[4] = lambda_i;
      J_i[5] = 0.0f;
      J_i[6] = (pos_i.y - lambda_i * trans.y) * scale_inv;
      r = target_pos_i.y - pos_i.y;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 7; ++u) {
#pragma unroll
      for (int v = 0; v < 7; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
    if (lambda_i > 0.0f) {
      J_i[0] = pos_i.y;
      J_i[1] = -pos_i.x;
      J_i[2] = 0.0f;
      J_i[3] = 0.0f;
      J_i[4] = 0.0f;
      J_i[5] = lambda_i;
      J_i[6] = (pos_i.z - lambda_i * trans.z) * scale_inv;
      r = target_pos_i.z - pos_i.z;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 7; ++u) {
#pragma unroll
      for (int v = 0; v < 7; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
  }
}

__global__ void kernelUpdateRTwithLandmark(
    pcl::gpu::PtrSz<float> JTJ, pcl::gpu::PtrSz<float> Jr,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position_sRT,
    const pcl::gpu::PtrSz<float3> normal_R,
    const pcl::gpu::PtrSz<int2> landmarkIndex) {
  __shared__ float partial_sum[32];
  int lId = threadIdx.x + blockIdx.x * blockDim.x;
  {
    float lambda_i;
    float3 target_pos_i, pos_i;
    if (lId < landmarkIndex.size) {
      int vId = __ldg(&landmarkIndex[lId].y);
      lambda_i = __ldg(&lambda[vId]);
      target_pos_i = target_position[vId];
      pos_i = position_sRT[vId];
    }

    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      lambda_i = sqrtf(lambda_i);
      target_pos_i = target_pos_i / lambda_i;
    } else {
      lambda_i = 0.0f;
    }

    pos_i *= lambda_i;
    float J_i[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, r = 0.0f;
    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      J_i[0] = 0.0f;
      J_i[1] = pos_i.z;
      J_i[2] = -pos_i.y;
      J_i[3] = lambda_i;
      J_i[4] = 0.0f;
      J_i[5] = 0.0f;
      r = target_pos_i.x - pos_i.x;
    }
    int cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      J_i[0] = -pos_i.z;
      J_i[1] = 0.0f;
      J_i[2] = pos_i.x;
      J_i[3] = 0.0f;
      J_i[4] = lambda_i;
      J_i[5] = 0.0f;
      r = target_pos_i.y - pos_i.y;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      J_i[0] = pos_i.y;
      J_i[1] = -pos_i.x;
      J_i[2] = 0.0f;
      J_i[3] = 0.0f;
      J_i[4] = 0.0f;
      J_i[5] = lambda_i;
      r = target_pos_i.z - pos_i.z;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
  }
}

__global__ void kernelUpdateRTwithLandmarkPoint2Line(
    pcl::gpu::PtrSz<float> JTJ, pcl::gpu::PtrSz<float> Jr,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position_sRT,
    const pcl::gpu::PtrSz<float3> normal_R,
    const pcl::gpu::PtrSz<int2> landmarkIndex) {
  __shared__ float partial_sum[32];
  int lId = threadIdx.x + blockIdx.x * blockDim.x;
  {
    float lambda_i = 0.0f;
    float3 target_pos_i, pos_i;
    if (lId < landmarkIndex.size) {
      int vId = __ldg(&landmarkIndex[lId].y);
      lambda_i = __ldg(&lambda[vId]);
      target_pos_i = target_position[vId];
      pos_i = position_sRT[vId];
    }
    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      lambda_i = sqrtf(lambda_i);
    } else {
      lambda_i = 0.0f;
    }
    pos_i *= lambda_i;
    float3 lv = target_pos_i / length(target_pos_i);
    float3 cross_lv_pos_i = cross(lv, pos_i);
    float alpha = dot(lv, pos_i);
    float J_i[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, r = 0.0f;
    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      J_i[0] = cross_lv_pos_i.x * lv.x;
      J_i[1] = cross_lv_pos_i.y * lv.x + pos_i.z;
      J_i[2] = cross_lv_pos_i.z * lv.x - pos_i.y;
      J_i[3] = lambda_i * (1 - lv.x * lv.x);
      J_i[4] = -lambda_i * lv.x * lv.y;
      J_i[5] = -lambda_i * lv.x * lv.z;
      r = alpha * lv.x - pos_i.x;
    }
    int cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      J_i[0] = cross_lv_pos_i.x * lv.y - pos_i.z;
      J_i[1] = cross_lv_pos_i.y * lv.y;
      J_i[2] = cross_lv_pos_i.z * lv.y + pos_i.x;
      J_i[3] = -lambda_i * lv.y * lv.x;
      J_i[4] = lambda_i * (1 - lv.y * lv.y);
      J_i[5] = -lambda_i * lv.y * lv.z;
      r = alpha * lv.y - pos_i.y;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
    if (lambda_i > 0.0f && lId < landmarkIndex.size) {
      J_i[0] = cross_lv_pos_i.x * lv.z + pos_i.y;
      J_i[1] = cross_lv_pos_i.y * lv.z - pos_i.x;
      J_i[2] = cross_lv_pos_i.z * lv.z;
      J_i[3] = -lambda_i * lv.z * lv.x;
      J_i[4] = -lambda_i * lv.z * lv.y;
      J_i[5] = lambda_i * (1 - lv.z * lv.z);
      r = alpha * lv.z - pos_i.z;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u;
      }
    }
  }
}

void cudaUpdateRTPoint2Line(pcl::gpu::DeviceArray<float> JTJ,
                            pcl::gpu::DeviceArray<float> Jr,
                            const pcl::gpu::DeviceArray<float> lambda,
                            const pcl::gpu::DeviceArray<float3> target_position,
                            const pcl::gpu::DeviceArray<float3> position_sRT,
                            const pcl::gpu::DeviceArray<float3> normal_R,
                            const pcl::gpu::PtrSz<int2> landmarkIndex) {
  kernelUpdateRTwithLandmarkPoint2Line<<<1, 1024>>>(
      JTJ, Jr, lambda, target_position, position_sRT, normal_R, landmarkIndex);
}

__global__ void kernelUpdateRTwithLandmarkPoint2Pixel(
    pcl::gpu::PtrSz<float> JTJ, pcl::gpu::PtrSz<float> Jr,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position_sRT,
    const pcl::gpu::PtrSz<float3> normal_R,
    const pcl::gpu::PtrSz<int2> landmarkIndex,
    const pcl::gpu::PtrSz<float> translation) {
  __shared__ float partial_sum[32];
  __shared__ float3 trans;
  __shared__ float scale_inv;
  if (threadIdx.x == 0) {
    trans = *((float3 *)(&translation[0]));
    scale_inv = 1.0f / translation[3];
  }
  __syncthreads();
  int lId = threadIdx.x + blockIdx.x * blockDim.x;
  {
    int vId;
    float lambda_i = 0.0f, lambda_i2;
    float3 target_pos_i;
    float3 pos_i;
    if (lId < landmarkIndex.size) {
      vId = __ldg(&landmarkIndex[lId].y);
      lambda_i2 = __ldg(&lambda[vId]);
      lambda_i = sqrtf(lambda_i2);
      target_pos_i = target_position[vId] / lambda_i;
      pos_i = lambda_i * position_sRT[vId];
    }
    float inv_z = 1 / pos_i.z;
    float J_i[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, r = 0.0f;
    if (lambda_i > 0.0f) {
      J_i[0] = 0.0f - pos_i.x * pos_i.y * inv_z * inv_z;  // 0.0f;
      J_i[1] = 1.0f + pos_i.x * pos_i.x * inv_z * inv_z;  // pos_i.z;
      J_i[2] = -pos_i.y * inv_z;                          //-pos_i.y;
      J_i[3] = lambda_i * inv_z;                          // lambda_i;
      J_i[4] = 0.0f;
      J_i[5] = -lambda_i * pos_i.x * inv_z * inv_z;
      r = target_pos_i.x / target_pos_i.z - pos_i.x * inv_z;
    }
    int cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv * lambda_i2;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u * lambda_i2;
      }
    }
    if (lambda_i > 0.0f) {
      J_i[0] = -1.0 - pos_i.y * pos_i.y * inv_z * inv_z;  //-pos_i.z;
      J_i[1] = pos_i.x * pos_i.y * inv_z * inv_z;         // 0.0f;
      J_i[2] = pos_i.x * inv_z;                           // pos_i.x;
      J_i[3] = 0.0f;
      J_i[4] = lambda_i * inv_z;  // lambda_i;
      J_i[5] = -lambda_i * pos_i.y * inv_z * inv_z;
      r = target_pos_i.y / target_pos_i.z - pos_i.y * inv_z;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
#pragma unroll
      for (int v = 0; v < 6; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv * lambda_i2;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u * lambda_i2;
      }
    }
    //    if (lambda_i > 0.0f) {
    //      J_i[0] = pos_i.y;
    //      J_i[1] = -pos_i.x;
    //      J_i[2] = 0.0f;
    //      J_i[3] = 0.0f;
    //      J_i[4] = 0.0f;
    //      J_i[5] = lambda_i;
    //      J_i[6] = (pos_i.z - lambda_i * trans.z) * scale_inv;
    //      r = target_pos_i.z - pos_i.z;
    //    }
    //    cnt = 0;
    //#pragma unroll
    //    for (int u = 0; u < 7; ++u) {
    //#pragma unroll
    //      for (int v = 0; v < 7; ++v) {
    //        float Jr_uv = J_i[u] * J_i[v];
    //        block_sum(Jr_uv, partial_sum);
    //        if (threadIdx.x == 31) {
    //          JTJ[cnt] += Jr_uv;
    //        }
    //        ++cnt;
    //      }
    //      float Jr_u = J_i[u] * r;
    //      block_sum(Jr_u, partial_sum);
    //      if (threadIdx.x == 31) {
    //        Jr[u] += Jr_u;
    //      }
    //    }
  }
}

void cudaUpdateRTPoint2Pixel(
    pcl::gpu::DeviceArray<float> JTJ, pcl::gpu::DeviceArray<float> Jr,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<int2> landmarkIndex,
    pcl::gpu::DeviceArray<float> translation) {
  kernelUpdateRTwithLandmarkPoint2Pixel<<<1, 1024>>>(
      JTJ, Jr, lambda, target_position, position_sRT, normal_R, landmarkIndex,
      translation);
}

__global__ void kernelUpdateSRTwithLandmarkPoint2Pixel(
    pcl::gpu::PtrSz<float> JTJ, pcl::gpu::PtrSz<float> Jr,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position_sRT,
    const pcl::gpu::PtrSz<float3> normal_R,
    const pcl::gpu::PtrSz<int2> landmarkIndex,
    const pcl::gpu::PtrSz<float> translation) {
  __shared__ float partial_sum[32];
  __shared__ float3 trans;
  __shared__ float scale_inv;
  if (threadIdx.x == 0) {
    trans = *((float3 *)(&translation[0]));
    scale_inv = 1.0f / translation[3];
  }
  __syncthreads();
  int lId = threadIdx.x + blockIdx.x * blockDim.x;
  {
    int vId;
    float lambda_i = 0.0f, lambda_i2;
    float3 target_pos_i;
    float3 pos_i;
    if (lId < landmarkIndex.size) {
      vId = __ldg(&landmarkIndex[lId].y);
      lambda_i2 = __ldg(&lambda[vId]);
      lambda_i = sqrtf(lambda_i2);
      target_pos_i = target_position[vId] / lambda_i;
      pos_i = lambda_i * position_sRT[vId];
    }
    float inv_z = 1 / pos_i.z;
    float J_i[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, r = 0.0f;
    if (lambda_i > 0.0f) {
      J_i[0] = 0.0f - pos_i.x * pos_i.y * inv_z * inv_z;  // 0.0f;
      J_i[1] = 1.0f + pos_i.x * pos_i.x * inv_z * inv_z;  // pos_i.z;
      J_i[2] = -pos_i.y * inv_z;                          //-pos_i.y;
      J_i[3] = lambda_i * inv_z;                          // lambda_i;
      J_i[4] = 0.0f;
      J_i[5] = -lambda_i * pos_i.x * inv_z * inv_z;
      J_i[6] = ((pos_i.x - lambda_i * trans.x) -
                (pos_i.z - lambda_i * trans.z) * pos_i.x * inv_z) *
               scale_inv * inv_z;
      r = target_pos_i.x / target_pos_i.z - pos_i.x * inv_z;
    }
    int cnt = 0;
#pragma unroll
    for (int u = 0; u < 7; ++u) {
#pragma unroll
      for (int v = 0; v < 7; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv * lambda_i2;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u * lambda_i2;
      }
    }
    if (lambda_i > 0.0f) {
      J_i[0] = -1.0 - pos_i.y * pos_i.y * inv_z * inv_z;  //-pos_i.z;
      J_i[1] = pos_i.x * pos_i.y * inv_z * inv_z;         // 0.0f;
      J_i[2] = pos_i.x * inv_z;                           // pos_i.x;
      J_i[3] = 0.0f;
      J_i[4] = lambda_i * inv_z;  // lambda_i;
      J_i[5] = -lambda_i * pos_i.y * inv_z * inv_z;
      J_i[6] = ((pos_i.y - lambda_i * trans.y) -
                (pos_i.z - lambda_i * trans.z) * pos_i.x * inv_z) *
               scale_inv * inv_z;
      r = target_pos_i.y / target_pos_i.z - pos_i.y * inv_z;
    }
    cnt = 0;
#pragma unroll
    for (int u = 0; u < 7; ++u) {
#pragma unroll
      for (int v = 0; v < 7; ++v) {
        float Jr_uv = J_i[u] * J_i[v];
        block_sum(Jr_uv, partial_sum);
        if (threadIdx.x == 31) {
          JTJ[cnt] += Jr_uv * lambda_i2;
        }
        ++cnt;
      }
      float Jr_u = J_i[u] * r;
      block_sum(Jr_u, partial_sum);
      if (threadIdx.x == 31) {
        Jr[u] += Jr_u * lambda_i2;
      }
    }
    //    if (lambda_i > 0.0f) {
    //      J_i[0] = pos_i.y;
    //      J_i[1] = -pos_i.x;
    //      J_i[2] = 0.0f;
    //      J_i[3] = 0.0f;
    //      J_i[4] = 0.0f;
    //      J_i[5] = lambda_i;
    //      J_i[6] = (pos_i.z - lambda_i * trans.z) * scale_inv;
    //      r = target_pos_i.z - pos_i.z;
    //    }
    //    cnt = 0;
    //#pragma unroll
    //    for (int u = 0; u < 7; ++u) {
    //#pragma unroll
    //      for (int v = 0; v < 7; ++v) {
    //        float Jr_uv = J_i[u] * J_i[v];
    //        block_sum(Jr_uv, partial_sum);
    //        if (threadIdx.x == 31) {
    //          JTJ[cnt] += Jr_uv;
    //        }
    //        ++cnt;
    //      }
    //      float Jr_u = J_i[u] * r;
    //      block_sum(Jr_u, partial_sum);
    //      if (threadIdx.x == 31) {
    //        Jr[u] += Jr_u;
    //      }
    //    }
  }
}

void cudaUpdateSRTPoint2Pixel(
    pcl::gpu::DeviceArray<float> JTJ, pcl::gpu::DeviceArray<float> Jr,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<int2> landmarkIndex,
    pcl::gpu::DeviceArray<float> translation) {
  kernelUpdateSRTwithLandmarkPoint2Pixel<<<1, 1024>>>(
      JTJ, Jr, lambda, target_position, position_sRT, normal_R, landmarkIndex,
      translation);
}

__device__ __forceinline__ float mask_func(float x) {
  // return 1.0 / (x + 0.05f);

  return 1.0f / (80 * x * x + 1.0f);
}

__global__ void kernelUpdateDfromM(pcl::gpu::PtrSz<float> D,
                                   const pcl::gpu::PtrSz<float> M, const int n,
                                   const float lambda) {
  const int rowIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int colIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (rowIdx < n && colIdx * n < D.size) {
    float mask_r_c = __ldg(&M[colIdx * 3 * n + rowIdx]);
    float offset = 0;
    // if (lambda > 1.5f) {
    //  offset = 0.0f;
    //} else {
    //  offset = 1.0f / (mask_r_c + 0.05f);
    //}
    /*if (rowIdx != 22 && rowIdx != 23 && rowIdx != 24) {
     mask_r_c = mask_r_c > 0.4f ? (mask_r_c - 0.4f) * 10.0f + 0.4f :
     mask_r_c;
   }*/
    D[colIdx * n + rowIdx] = lambda * mask_func(mask_r_c) + offset;
  }
}

__device__ __forceinline__ float mask_func_Corr(float x) {
  // return 1.0 / (x + 0.05f);

  return 1.0f / (80 * x * x + 0.05f);
  // return 1.0f / 40.0f;
}

__global__ void kernelUpdateDfromM_Corr(pcl::gpu::PtrSz<float> D,
                                        const pcl::gpu::PtrSz<float> M,
                                        const int n, const float lambda) {
  const int rowIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int colIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (rowIdx < n && colIdx * n < D.size) {
    float mask_r_c = __ldg(&M[colIdx * 3 * n + rowIdx]);
    float offset = 0;
    // if (lambda > 0.015f) {
    //  offset = 0.0f;
    //} else {
    //  offset = 0.01f / (mask_r_c + 0.05f);
    //}
    // if (rowIdx != 22 && rowIdx != 23 && rowIdx != 24) {
    //  mask_r_c = mask_r_c > 0.4f ? (mask_r_c - 0.4f) * 10.0f + 0.4f :
    //  mask_r_c;
    //}
    D[colIdx * n + rowIdx] = lambda * mask_func_Corr(mask_r_c) + offset;
  }
}

void cudaUpdateDfromM(pcl::gpu::DeviceArray<float> D,
                      const pcl::gpu::DeviceArray<float> M, const int n,
                      const float lambda, const cudaStream_t stream) {
  dim3 block(16, 64);
  dim3 grid(pcl::gpu::divUp(n, block.x),
            pcl::gpu::divUp(D.size() / n, block.y));
  kernelUpdateDfromM<<<grid, block, 0, stream>>>(D, M, n, lambda);
}

void cudaUpdateDfromM_Corr(pcl::gpu::DeviceArray<float> D,
                           const pcl::gpu::DeviceArray<float> M, const int n,
                           const float lambda, const cudaStream_t stream) {
  dim3 block(16, 64);
  dim3 grid(pcl::gpu::divUp(n, block.x),
            pcl::gpu::divUp(D.size() / n, block.y));
  kernelUpdateDfromM_Corr<<<grid, block, 0, stream>>>(D, M, n, lambda);
}

__global__ void kernelUpdateContourLandmarkIndex(
    pcl::gpu::PtrSz<int2> contour_landmark_index,
    const pcl::gpu::PtrSz<float2> landmark_uv,
    const pcl::gpu::PtrSz<float3> position_sRT,
    const pcl::gpu::PtrSz<int> stripIndex,
    const pcl::gpu::PtrSz<int2> stripBeginEnd,
    const msfr::intrinsics camera_intr) {
  __shared__ int shared_value[32];
  __shared__ float shared_key[32];
  const int strip_num = stripBeginEnd.size;

  float2 pos;
  for (int i = 0; i < strip_num; ++i) {
    __shared__ int2 begin_end;
    __shared__ int strip_vert_num;
    if (threadIdx.x == 0) {
      begin_end = stripBeginEnd[i];
      strip_vert_num = begin_end.y - begin_end.x + 1;
    }
    __syncthreads();

    int value;
    pos = make_float2(-FLT_MAX, -FLT_MAX);
    if (threadIdx.x < strip_vert_num) {
      const int vId = __ldg(&stripIndex[threadIdx.x + begin_end.x]);
      value = vId;
      float3 v = position_sRT[vId];
      pos = getProjectPos(camera_intr, v);
    }
    if (i < strip_num / 2) {
      if (threadIdx.x < strip_vert_num) {
        pos.x = -pos.x;
      }
      block_max(value, pos.x, shared_value, shared_key);
    } else {
      block_max(value, pos.x, shared_value, shared_key);
    }
    if (threadIdx.x == 0) {
      contour_landmark_index[i].y = value;
    }
  }
}

void cudaUpdateContourLandmarkIndex(
    pcl::gpu::DeviceArray<int2> contour_landmark_index,
    const pcl::gpu::DeviceArray<float2> landmark_uv,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<int> stripIndex,
    const pcl::gpu::DeviceArray<int2> stripBeginEnd,
    const msfr::intrinsics camera_intr) {
  dim3 block(1024);
  dim3 grid(1);
  kernelUpdateContourLandmarkIndex<<<grid, block>>>(
      contour_landmark_index, landmark_uv, position_sRT, stripIndex,
      stripBeginEnd, camera_intr);
}

__global__ void kernelSmoothExpBase(
    pcl::gpu::PtrSz<float> smooth_base, const pcl::gpu::PtrSz<float> base,
    const pcl::gpu::PtrSz<float3> mean_shape,
    const pcl::gpu::PtrSz<int3> tri_list,
    const pcl::gpu::PtrSz<int2> fvLookUpTable,
    const pcl::gpu::PtrSz<int1> fBegin, const pcl::gpu::PtrSz<int1> fEnd,
    const pcl::gpu::PtrSz<unsigned short> is_boundary, const int dim_exp) {
  const int dimId = threadIdx.x + blockIdx.x * blockDim.x;
  const int vId = threadIdx.y + blockIdx.y * blockDim.y;

  float weight = 0.0f;
  float3 delta = {0.0f, 0.0f, 0.0f};
  float3 vI = mean_shape[vId];
  if (is_boundary[vId] == 0) {
    delta += 2.0f * (make_float3(base[dim_exp * 3 * vId + dimId],
                                 base[dim_exp * (3 * vId + 1) + dimId],
                                 base[dim_exp * (3 * vId + 2) + dimId]) +
                     vI);
    weight += 2.0f;
    for (int i = fBegin[vId].x; i < fEnd[vId].x; ++i) {
      const int findex = fvLookUpTable[i].x;
      const int v[3] = {tri_list[findex].x, tri_list[findex].y,
                        tri_list[findex].z};
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        if (is_boundary[v[j]] == 0 && v[j] != vId) {
          delta += make_float3(base[dim_exp * 3 * v[j] + dimId],
                               base[dim_exp * (3 * v[j] + 1) + dimId],
                               base[dim_exp * (3 * v[j] + 2) + dimId]) +
                   mean_shape[v[j]];
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
        verts[j] =
            make_float3(base[dim_exp * 3 * v[reIndex[j]] + dimId],
                        base[dim_exp * (3 * v[reIndex[j]] + 1) + dimId],
                        base[dim_exp * (3 * v[reIndex[j]] + 2) + dimId]) +
            mean_shape[v[reIndex[j]]];
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
    delta += make_float3(base[dim_exp * 3 * vId + dimId],
                         base[dim_exp * (3 * vId + 1) + dimId],
                         base[dim_exp * (3 * vId + 2) + dimId]) +
             vI;
    weight += 1.0f;
  }

  delta /= weight;
  delta -= vI;
  smooth_base[dim_exp * 3 * vId + dimId] = delta.x;
  smooth_base[dim_exp * (3 * vId + 1) + dimId] = delta.y;
  smooth_base[dim_exp * (3 * vId + 2) + dimId] = delta.z;
}

void cudaSmoothExpBase(pcl::gpu::DeviceArray<float> smooth_base,
                       const pcl::gpu::DeviceArray<float> base,
                       const pcl::gpu::DeviceArray<float3> mean_shape,
                       const pcl::gpu::DeviceArray<int3> tri_list,
                       const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                       const pcl::gpu::DeviceArray<int1> fBegin,
                       const pcl::gpu::DeviceArray<int1> fEnd,
                       const pcl::gpu::DeviceArray<unsigned short> is_boundary,
                       const int dim_exp, const int n_verts,
                       const cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid(pcl::gpu::divUp(dim_exp, block.x),
            pcl::gpu::divUp(n_verts, block.x));
  kernelSmoothExpBase<<<grid, block, 0, stream>>>(
      smooth_base, base, mean_shape, tri_list, fvLookUpTable, fBegin, fEnd,
      is_boundary, dim_exp);
}

__global__ void kernelPlusAandB(pcl::gpu::PtrSz<float> AandB,
                                const pcl::gpu::PtrSz<float> A,
                                const pcl::gpu::PtrSz<float> B) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < A.size) {
    AandB[idx] = __ldg(&A[idx]) + __ldg(&B[idx]);
  }
}

void cudaPlusAandB(pcl::gpu::DeviceArray<float> AandB,
                   const pcl::gpu::DeviceArray<float> A,
                   const pcl::gpu::DeviceArray<float> B,
                   const cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(B.size(), block.x));
  kernelPlusAandB<<<grid, block, 0, stream>>>(AandB, A, B);
}

__global__ void kernelMinusAandB(pcl::gpu::PtrSz<float> AandB,
                                 const pcl::gpu::PtrSz<float> A,
                                 const pcl::gpu::PtrSz<float> B) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < A.size) {
    AandB[idx] = __ldg(&A[idx]) - __ldg(&B[idx]);
  }
}

void cudaMinusAandB(pcl::gpu::DeviceArray<float> AandB,
                    const pcl::gpu::DeviceArray<float> A,
                    const pcl::gpu::DeviceArray<float> B,
                    const cudaStream_t stream) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(B.size(), block.x));
  kernelMinusAandB<<<grid, block, 0, stream>>>(AandB, A, B);
}

__global__ void kernelTransposeA(pcl::gpu::PtrSz<float> AT,
                                 const pcl::gpu::PtrSz<float> A, const int n,
                                 const int ldn, const int m,
                                 const int row_offset) {
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (rowIdx < n && colIdx < m) {
    AT[colIdx * ldn + rowIdx + row_offset] = __ldg(&A[rowIdx * m + colIdx]);
  }
}

void cudaTransposeA(pcl::gpu::DeviceArray<float> AT,
                    const pcl::gpu::DeviceArray<float> A, const int n,
                    const int ldn, const int m, const int row_offset) {
  dim3 block(16, 16);
  dim3 grid(pcl::gpu::divUp(n, block.x), pcl::gpu::divUp(m, block.y));
  kernelTransposeA<<<grid, block>>>(AT, A, n, ldn, m, row_offset);
}

__global__ void kernelTransposeA_decay(pcl::gpu::PtrSz<float> AT,
                                       const pcl::gpu::PtrSz<float> A,
                                       const int n, const int ldn, const int m,
                                       const int row_offset,
                                       const float decay) {
  int rowIdx = blockIdx.x;
  int colIdx = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ float decay_row;
  if (threadIdx.y == 0) {
    // decay_row = sqrtf(expf(decay * (n - 1 - rowIdx)) * (1 - expf(decay)) / (1
    // - expf(decay * n)));
    decay_row = sqrtf(1.0f / n);
  }
  __syncthreads();
  if (rowIdx < n && colIdx < m) {
    AT[colIdx * ldn + rowIdx + row_offset] =
        decay_row * __ldg(&A[rowIdx * m + colIdx]);
  }
}

void cudaTransposeA_decay(pcl::gpu::DeviceArray<float> AT,
                          const pcl::gpu::DeviceArray<float> A, const int n,
                          const int ldn, const int m, const int row_offset,
                          const float decay, const cudaStream_t stream) {
  dim3 block(1, 64);
  dim3 grid(pcl::gpu::divUp(n, block.x), pcl::gpu::divUp(m, block.y));
  kernelTransposeA_decay<<<grid, block, 0, stream>>>(AT, A, n, ldn, m,
                                                     row_offset, decay);
}

__global__ void kernelSampleExpBase(pcl::gpu::PtrSz<float> sampled_base,
                                    const pcl::gpu::PtrSz<float> base,
                                    const pcl::gpu::PtrSz<int> sampled_key,
                                    const int step_base) {
  const int rowIdx = blockIdx.x;
  const int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (rowIdx < sampled_key.size && colIdx < step_base) {
    const int vId = __ldg(&sampled_key[rowIdx]);
    int sampled_offset = rowIdx * 3 * step_base + colIdx;
    int offset = vId * 3 * step_base + colIdx;
    sampled_base[sampled_offset] = base[offset];
    sampled_offset += step_base;
    offset += step_base;
    sampled_base[sampled_offset] = base[offset];
    sampled_offset += step_base;
    offset += step_base;
    sampled_base[sampled_offset] = base[offset];
  }
}

void cudaSampleExpBase(pcl::gpu::DeviceArray<float> sampled_base,
                       const pcl::gpu::DeviceArray<float> base,
                       const pcl::gpu::DeviceArray<int> sampled_key) {
  const int step_base = sampled_base.size() / sampled_key.size() / 3;
  dim3 block(32);
  dim3 grid(sampled_key.size(), pcl::gpu::divUp(step_base, block.x));
  kernelSampleExpBase<<<grid, block>>>(sampled_base, base, sampled_key,
                                       step_base);
}

__global__ void kernelPushLandmarkTargetVertices(
    pcl::gpu::PtrSz<float3> target_position,
    pcl::gpu::PtrSz<float> lambda_position,
    const pcl::gpu::PtrSz<int2> landmark_id,
    const pcl::gpu::PtrSz<float3> input_target_position, const float lambda,
    const bool is_use_P2L) {
  int lId = threadIdx.x;
  int vId = landmark_id[lId].y;
  int landmarkId = landmark_id[lId].x;
  if (lId < input_target_position.size) {
    float3 input_pos = input_target_position[lId];
    if (input_pos.z == -1.0f && is_use_P2L) {
      input_pos = -input_pos;
    }
    if (input_pos.z > 0.0f) {
      // if (landmarkId == 93 || (landmarkId >= 99 && landmarkId <= 105) ||
      //    landmarkId == 109) {
      //  target_position[vId] += 60.0f * lambda * input_target_position[lId];
      //  lambda_position[vId] += 60.0f * lambda;
      //} else if (landmarkId == 84 || landmarkId == 89) {
      //  target_position[vId] += 30.0f * lambda * input_target_position[lId];
      //  lambda_position[vId] += 30.0f * lambda;
      //} else if (landmarkId >= 57 && landmarkId <= 80) {
      //  target_position[vId] += 20 * lambda * input_target_position[lId];
      //  lambda_position[vId] += 20 * lambda;
      //} else
      // if (landmarkId == 18) {
      //  target_position[vId] += 60.0f * lambda * input_target_position[lId];
      //  lambda_position[vId] += 60.0f * lambda;
      //} else
      {
        target_position[vId] += lambda * input_target_position[lId];
        lambda_position[vId] += lambda;
      }
    }
  }
}

void cudaPushLandmarkTargetVertices(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position,
    const float lambda, const bool is_use_P2L) {
  // cudaSafeCall(cudaMemcpyToSymbol(dev_reg_lambda, &lambda, sizeof(float)));
  kernelPushLandmarkTargetVertices<<<1, landmark_id.size()>>>(
      target_position, lambda_position, landmark_id, input_target_position,
      lambda, is_use_P2L);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelPushNoseLandmarkTargetVertices(
    pcl::gpu::PtrSz<float3> target_position,
    pcl::gpu::PtrSz<float> lambda_position,
    const pcl::gpu::PtrSz<int2> landmark_id,
    const pcl::gpu::PtrSz<float3> input_target_position, const float lambda) {
  int lId = threadIdx.x;

  if (lId < input_target_position.size) {
    auto input_pos = input_target_position[lId];
    if (input_pos.z == -1.0f) {
      input_pos = -input_pos;
    }
    int vId = landmark_id[lId].y;
    int landmarkId = landmark_id[lId].x;
    if ((landmarkId >= 27 && landmarkId <= 30) || landmarkId == 33) {
      target_position[vId] += 30.0f * lambda * input_pos;
      lambda_position[vId] += 30.0f * lambda;
    } else if (landmarkId == 36 || (39 <= landmarkId && landmarkId <= 41) ||
               landmarkId == 42 ||
               (45 <= landmarkId &&
                landmarkId <= 47)) {  // fixme add eye landmarks to pose
      target_position[vId] += 30.0f * lambda * input_pos;
      lambda_position[vId] += 30.0f * lambda;
    }
  }
}

void cudaPushNoseLandmarkTargetVertices(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position,
    const float lambda) {
  kernelPushNoseLandmarkTargetVertices<<<1, landmark_id.size()>>>(
      target_position, lambda_position, landmark_id, input_target_position,
      lambda);
}

__global__ void kernelPushExpLandmarkTargetVertices(
    pcl::gpu::PtrSz<float3> target_position,
    pcl::gpu::PtrSz<float> lambda_position,
    const pcl::gpu::PtrSz<unsigned short> is_front,
    const pcl::gpu::PtrSz<int2> landmark_id,
    const pcl::gpu::PtrSz<float3> input_target_position, const float lambda,
    const float eye_lambda, const float eyebrow_lambda) {
  int lId = threadIdx.x;
  float3 input_pos = input_target_position[lId];
  if (lId < input_target_position.size && input_pos.z != 0.0f) {
    int vId = landmark_id[lId].y;
    // if (input_pos.z == -1.0f)
    //{
    //  input_pos = -input_pos;
    //}
    if (input_pos.z != 0.0f) {
      int landmarkId = landmark_id[lId].x;
      if (landmarkId == 93 || landmarkId == 99 || landmarkId == 105 ||
          landmarkId == 109) {
        target_position[vId] += 180 * lambda * input_pos;  // todo mouth
        lambda_position[vId] += 180 * lambda;              // todo mouth
      } else if (landmarkId >= 57 && landmarkId <= 80) {
        target_position[vId] +=
            eye_lambda * lambda * input_pos;  // todo eye_lambda
        lambda_position[vId] += eye_lambda * lambda;
      } else if ((landmarkId >= 106 && landmarkId <= 108) ||
                 (landmarkId >= 110 && landmarkId <= 114)) {
        target_position[vId] += 120 * lambda * input_pos;  // todo inner mouth
        lambda_position[vId] += 120 * lambda;
      } else if (landmarkId >= 93 && landmarkId <= 112) {
        target_position[vId] += 180 * lambda * input_pos;
        lambda_position[vId] += 180 * lambda;
      } else if (landmarkId >= 37 && landmarkId <= 56) {
        target_position[vId] += eyebrow_lambda * lambda * input_pos;
        lambda_position[vId] += eyebrow_lambda * lambda;
      } else {
        target_position[vId] += 20 * lambda * input_pos;
        lambda_position[vId] += 20 * lambda;
      }
    }
  }
}

void cudaPushExpLandmarkTargetVertices(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position,
    const float lambda, const float eye_lambda, const float eyebrow_lambda) {
  kernelPushExpLandmarkTargetVertices<<<1, landmark_id.size()>>>(
      target_position, lambda_position, is_front, landmark_id,
      input_target_position, lambda, eye_lambda, eyebrow_lambda);
}

__global__ void kernelPushExpLandmarkTargetVerticesFromUV(
    pcl::gpu::PtrSz<float3> target_position,
    pcl::gpu::PtrSz<float> lambda_position,
    const pcl::gpu::PtrSz<int2> landmark_id,
    const pcl::gpu::PtrSz<float2> input_target_uv,
    const msfr::intrinsics camera, const float lambda, const float eye_lambda,
    const float eyebrow_lambda) {
  int lId = threadIdx.x;
  if (lId < input_target_uv.size) {
    float2 input_uv = input_target_uv[lId];
    int vId = landmark_id[lId].y;
    // if (input_pos.z == -1.0f) {
    //  input_pos = -input_pos;
    //}
    if (input_uv.x > 0.0f) {
      float3 input_pos = {input_uv.x, input_uv.y, 1.0f};
      input_pos = unProjectedFromIndex(camera, input_pos);
      int landmarkId = landmark_id[lId].x;
      if (landmarkId == 48 || landmarkId == 54 || landmarkId == 60 ||
          landmarkId == 64) {
        target_position[vId] += lambda * input_pos;  // todo mouth
        lambda_position[vId] += lambda;              // todo mouth
      } else if (landmarkId >= 36 && landmarkId <= 47) {
        target_position[vId] +=
            eye_lambda * lambda * input_pos;  // todo eye_lambda
        lambda_position[vId] += eye_lambda * lambda;
      } else if ((landmarkId >= 61 && landmarkId <= 63) ||
                 (landmarkId >= 65 && landmarkId <= 67)) {
        target_position[vId] += lambda * input_pos;  // todo inner mouth
        lambda_position[vId] += lambda;
      } else if (landmarkId >= 48 && landmarkId <= 67) {
        target_position[vId] += lambda * input_pos;
        lambda_position[vId] += lambda;
      } else if (landmarkId >= 17 && landmarkId <= 26) {
        target_position[vId] += eyebrow_lambda * lambda * input_pos;
        lambda_position[vId] += eyebrow_lambda * lambda;
      } else if (landmarkId == 31 || landmarkId == 35) {
        target_position[vId] += lambda * input_pos;
        lambda_position[vId] += lambda;
      } else {
        target_position[vId] += lambda * input_pos;
        lambda_position[vId] += lambda;
      }
    }
  }
}

void cudaPushExpLandmarkTargetVerticesFromUV(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float2> input_target_uv,
    const msfr::intrinsics camera, const float lambda, const float eye_lambda,
    const float eyebrow_lambda) {
  kernelPushExpLandmarkTargetVerticesFromUV<<<1, landmark_id.size()>>>(
      target_position, lambda_position, landmark_id, input_target_uv, camera,
      lambda, eye_lambda, eyebrow_lambda);
}

__global__ void kernelPushSymVertices(pcl::gpu::PtrSz<float3> target_position,
                                      pcl::gpu::PtrSz<float> lambda_position,
                                      const pcl::gpu::PtrSz<int> symlist,
                                      const float lambda) {
  const int vId = threadIdx.x + blockIdx.x * blockDim.x;
  if (vId < (symlist.size >> 1)) {
    // if (lambda_position[vId] == 0.0f)
    {
      const int sym_vId = symlist[vId];
      if (sym_vId != -1) {
        float lambda_i = lambda_position[vId];
        float sym_lambda_i = lambda_position[sym_vId];
        float3 sym_pos = {0.0f, 0.0f, 0.0f};
        float3 pos = {0.0f, 0.0f, 0.0f};
        if (sym_lambda_i != 0)
          sym_pos = target_position[sym_vId] / lambda_position[sym_vId];
        if (lambda_i != 0) pos = target_position[vId] / lambda_position[vId];
        lambda_position[vId] += lambda;
        pos.x = -pos.x;
        target_position[vId] += sym_pos * lambda;

        lambda_position[sym_vId] += lambda;
        sym_pos.x = -sym_pos.x;
        target_position[sym_vId] += pos * lambda;
      }
    }
  }
}

void cudaPushSymVertices(pcl::gpu::DeviceArray<float3> target_position,
                         pcl::gpu::DeviceArray<float> lambda_position,
                         const pcl::gpu::DeviceArray<int> symlist,
                         const float lambda) {
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(lambda_position.size() / 2, block.x));
  kernelPushSymVertices<<<grid, block>>>(target_position, lambda_position,
                                         symlist, lambda);
}

__global__ void kernelPushPresentVertices(
    pcl::gpu::PtrSz<float3> target_position,
    pcl::gpu::PtrSz<float> lambda_position,
    const pcl::gpu::PtrSz<float3> position, const float lambda) {
  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  if (vId < target_position.size) {
    float3 input_pos = position[vId];
    target_position[vId] += lambda * input_pos;
    lambda_position[vId] += lambda;
  }
}

void cudaPushPresentVertices(pcl::gpu::DeviceArray<float3> target_position,
                             pcl::gpu::DeviceArray<float> lambda_position,
                             const pcl::gpu::DeviceArray<float3> position,
                             const float lambda) {
  // cudaSafeCall(cudaMemcpyToSymbol(dev_reg_lambda, &lambda, sizeof(float)));
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position.size(), block.x));
  kernelPushPresentVertices<<<grid, block>>>(target_position, lambda_position,
                                             position, lambda);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void
kernelUpdateTranslation(  // pcl::gpu::PtrSz<float4> T_intermediate,
    pcl::gpu::PtrSz<float> translation, const pcl::gpu::PtrSz<float> rotation_,
    const pcl::gpu::PtrSz<float> lambda,
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position) {
  __shared__ float partial_sum[32];
  __shared__ float rotation[9];
  if (threadIdx.x < 9) {
    rotation[threadIdx.x] = rotation_[threadIdx.x] * translation[3];
  }
  __syncthreads();

  float3 T = make_float3(0.0f, 0.0f, 0.0f);
  float lambda_i = 0;
  int vId = threadIdx.x;
  int position_n = position.size;
  for (; vId < position_n; vId += 1024) {
    if (lambda[vId] > 0.0f) {
      T += target_position[vId] - M33xV3(rotation, position[vId]) * lambda[vId];
      lambda_i += lambda[vId];
    }
  }
  block_sum(T.x, partial_sum);
  block_sum(T.y, partial_sum);
  block_sum(T.z, partial_sum);
  block_sum(lambda_i, partial_sum);

  if (threadIdx.x == 31) {
    translation[0] = T.x / lambda_i;
    translation[1] = T.y / lambda_i;
    translation[2] = T.z / lambda_i;
  }
}

__global__ void kernelSumTranslation(
    pcl::gpu::PtrSz<float> T, const pcl::gpu::PtrSz<float4> T_intermediate) {
  __shared__ float partial_sum[32];
  int threadDim = (T_intermediate.size + 1023) >> 10;
  int beginId = threadIdx.x * threadDim;
  int endId = min(beginId + threadDim, (int)T_intermediate.size);
  float4 thread_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int i = beginId; i < endId; ++i) {
    thread_sum += T_intermediate[i];
  }
  block_sum(thread_sum.x, partial_sum);
  block_sum(thread_sum.y, partial_sum);
  block_sum(thread_sum.z, partial_sum);
  block_sum(thread_sum.w, partial_sum);
  if (threadIdx.x == 31) {
    T[0] = thread_sum.x / thread_sum.w;
    T[1] = thread_sum.y / thread_sum.w;
    T[2] = thread_sum.z / thread_sum.w;
  }
}

void cudaUpdateTranslation(pcl::gpu::DeviceArray<float> translation,
                           const pcl::gpu::DeviceArray<float> rotation,
                           const pcl::gpu::DeviceArray<float> lambda,
                           const pcl::gpu::DeviceArray<float3> target_position,
                           const pcl::gpu::DeviceArray<float3> position) {
  // dim3 block(1024);
  // dim3 grid(pcl::gpu::divUp(position.size(), block.x));
  kernelUpdateTranslation<<<1, 1024>>>(translation, rotation, lambda,
                                       target_position, position);
  // pcl::gpu::DeviceArray<float4> T_intermediate(grid.x);
  // kernelUpdateTranslation<<<grid, block>>>(T_intermediate, translation,
  // rotation,
  //  lambda, target_position, position);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  // kernelSumTranslation<<<1, 1024>>>(translation, T_intermediate);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelUpdateErr(pcl::gpu::PtrSz<float> err,
                                const pcl::gpu::PtrSz<float> rotation,
                                const pcl::gpu::PtrSz<float> translation,
                                const pcl::gpu::PtrSz<float> lambda,
                                const pcl::gpu::PtrSz<float3> target_position,
                                const pcl::gpu::PtrSz<float3> position) {
  __shared__ float3 T;
  __shared__ float scale;
  __shared__ float partial_sum[32];
  if (threadIdx.x == 0) {
    T.x = translation[0];
    T.y = translation[1];
    T.z = translation[2];
    scale = translation[3];
  }
  __syncthreads();
  int threadDim = (position.size + 1023) >> 10;
  int beginId = threadDim * threadIdx.x;
  int endId = min(beginId + threadDim, (int)position.size);
  float err_i = 0;
  float lambda_i = 0;
  for (int i = beginId; i < endId; ++i) {
    if (lambda[i] > 0.0f) {
      float3 d = target_position[i] -
                 (M33xV3(rotation, position[i]) * scale + T) * lambda[i];
      err_i += norm2(d) / lambda[i];
      lambda_i += lambda[i];
    }
  }
  block_sum(err_i, partial_sum);
  block_sum(lambda_i, partial_sum);
  if (threadIdx.x == 31) {
    err[0] = err_i / lambda_i;
  }
}

void cudaUpdateErr(pcl::gpu::DeviceArray<float> err,
                   const pcl::gpu::DeviceArray<float> rotation,
                   const pcl::gpu::DeviceArray<float> translation,
                   const pcl::gpu::DeviceArray<float> lambda,
                   const pcl::gpu::DeviceArray<float3> target_position,
                   const pcl::gpu::DeviceArray<float3> position) {
  kernelUpdateErr<<<1, 1024>>>(err, rotation, translation, lambda,
                               target_position, position);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelUpdateLandmarkErr(
    const pcl::gpu::PtrSz<float3> position,
    const pcl::gpu::PtrSz<unsigned short> is_front,
    const pcl::gpu::PtrSz<float> rotation_,
    const pcl::gpu::PtrSz<float> translation_,
    const pcl::gpu::PtrSz<int2> landmark_id,
    const pcl::gpu::PtrSz<float3> input_target_position)
// pcl::gpu::PtrSz<float3> temp_pos)
{
  __shared__ float partial_sum[32];
  __shared__ float3 translation;
  __shared__ float rotation[9];
  int id = threadIdx.x;
  if (id == 0) {
    translation.x = translation_[0];
    translation.y = translation_[1];
    translation.z = translation_[2];
  }
  if (id < 9) {
    rotation[id] = rotation_[id] * translation_[3];
  }
  __syncthreads();
  float err_i = 0;
  if (id < landmark_id.size) {
    if (input_target_position[id].z > 0.0f) {
      int vId = landmark_id[id].y;
      // temp_pos[id] = M33xV3(rotation, position[vId]) + translation;
      if (is_front[vId] == 1) {
        err_i += norm2(M33xV3(rotation, position[vId]) + translation -
                       input_target_position[id]);
      }
    }
  }
  block_sum(err_i, partial_sum);
  if (id == 31) {
    dev_error = err_i;
  }
}

void cudaUpdateLandmarkErr(
    float &err, const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float3> position,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position) {
  err = 0;
  // pcl::gpu::DeviceArray<float3> pos(input_target_position.size());
  cudaSafeCall(cudaMemcpyToSymbol(dev_error, &err, sizeof(float)));
  kernelUpdateLandmarkErr<<<1, 1024>>>(position, is_front, rotation,
                                       translation, landmark_id,
                                       input_target_position);
  cudaSafeCall(cudaMemcpyFromSymbol(&err, dev_error, sizeof(float)));

  // std::vector<float3> host_pos, host_input;
  // pos.download(host_pos);
  // input_target_position.download(host_input);
}

__global__ void kernelUpdateICPErr(
    const pcl::gpu::PtrSz<float3> target_position,
    const pcl::gpu::PtrSz<float3> position_sRT) {
  __shared__ float partial_sum[32];
  int id = threadIdx.x;
  int threadDim = (position_sRT.size + 1023) >> 10;
  int beginId = threadIdx.x * threadDim;
  int endId = min(beginId + threadDim, (int)position_sRT.size);
  float err_i = 0;
  float sum_cnt = 0;
  for (int i = beginId; i < endId; ++i) {
    if (target_position[i].z > 0.0f) {
      err_i += length(position_sRT[i] - target_position[i]);
      ++sum_cnt;
    }
  }
  block_sum(err_i, partial_sum);
  block_sum(sum_cnt, partial_sum);
  if (id == 31) {
    if (sum_cnt != 0) {
      dev_error = err_i / sum_cnt;
    }
  }
}

void cudaUpdateICPErr(float &err,
                      const pcl::gpu::DeviceArray<float3> target_position,
                      const pcl::gpu::DeviceArray<float3> position_sRT) {
  err = 0;
  // pcl::gpu::DeviceArray<float3> pos(input_target_position.size());
  cudaSafeCall(cudaMemcpyToSymbol(dev_error, &err, sizeof(float)));
  kernelUpdateICPErr<<<1, 1024>>>(target_position, position_sRT);
  cudaSafeCall(cudaMemcpyFromSymbol(&err, dev_error, sizeof(float)));
  cudaSafeCall(cudaStreamSynchronize(0));
}

__device__ __forceinline__ float interp(float v, float y0, float x0, float y1,
                                        float x1) {
  if (v < x0)
    return y0;
  else if (v > x1)
    return y1;
  else
    return (v - x0) * (y1 - y0) / (x1 - x0) + y0;
}

__device__ __forceinline__ float jet_base(float v) {
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

__device__ __forceinline__ float3 jet_colormap(float v) {
  float r = jet_base(v * 2.0f - 1.5f);
  float g = jet_base(v * 2.0f - 1.0f);
  float b = jet_base(v * 2.0f - 0.5f);
  return make_float3(r, g, b);
}

__global__ void kernelUpdateExpA(pcl::gpu::PtrSz<float> A,
                                 const pcl::gpu::PtrSz<float> kf_coefficients,
                                 const int step, const int n) {
  __shared__ float matA[16][16];
  __shared__ float matB[16][16];
  const int tidc = threadIdx.x;
  const int tidr = threadIdx.y;
  const int bidc = blockIdx.x << 4;
  const int bidr = blockIdx.y << 4;
  // if (bidc <= bidr)
  {
    int i, j;
    float results = 0;
    float comp = 0;

    for (j = 0; j < n; j += 16) {
      if (tidr + bidr < step && tidc + j < n) {
        matA[tidr][tidc] = kf_coefficients[(tidc + j) * step + tidr + bidr];
      } else {
        matA[tidr][tidc] = 0;
      }

      if (tidr + j < n && tidc + bidc < step) {
        matB[tidr][tidc] = kf_coefficients[(tidr + j) * step + tidc + bidc];
      } else {
        matB[tidr][tidc] = 0;
      }

      __syncthreads();

      for (i = 0; i < 16; i++) {
        float t;
        // results += matA[tidr][i] * matB[i][tidc];
        comp -= matA[tidr][i] * matB[i][tidc];
        t = results - comp;
        comp = (t - results) + comp;
        results = t;
      }

      __syncthreads();
    }

    if (tidr + bidr < n && tidc + bidc < n) {
      A[(tidr + bidr) * step + tidc + bidc] = results;
    }
  }
}

__global__ void kernelUpdateExpb(
    pcl::gpu::PtrSz<float> kf_b_values,
    const pcl::gpu::PtrSz<float> key_frames_coefficients,
    const pcl::gpu::PtrSz<float3> key_frames_position) {}

__global__ void kernelUpdateCTC(pcl::gpu::PtrSz<float> CTC,
                                const pcl::gpu::PtrSz<float> exp_coefficent,
                                const pcl::gpu::PtrSz<int> activated_index,
                                const int activated_index_num) {
  if (threadIdx.x < activated_index_num && blockIdx.x < activated_index_num) {
    const int row_id = __ldg(&activated_index[blockIdx.x]);
    const int col_id = __ldg(&activated_index[threadIdx.x]);
    __shared__ float exp_row;
    if (threadIdx.x == 0) {
      exp_row = __ldg(&exp_coefficent[row_id]);
    }
    __syncthreads();
    const int step = exp_coefficent.size;
    const float exp_col = __ldg(&exp_coefficent[col_id]);
    CTC[col_id * step + row_id] += exp_row * exp_col;
  }
}

void cudaUpdateExpBase(
    pcl::gpu::DeviceArray<float> exp_base,
    pcl::gpu::DeviceArray<float> kf_A_values,
    pcl::gpu::DeviceArray<float> kf_b_values,
    const pcl::gpu::DeviceArray<float3> key_frames_position,
    const pcl::gpu::DeviceArray<float> key_frames_coefficients,
    const int key_frames_num, const int dim_exp) {
  {
    dim3 block(16, 16);
    dim3 grid(pcl::gpu::divUp(dim_exp, 16), pcl::gpu::divUp(dim_exp, 16));
    kernelUpdateExpA<<<grid, block>>>(kf_A_values, key_frames_coefficients,
                                      dim_exp, key_frames_num);
  }
  {
    dim3 block(256);
    dim3 grid(dim_exp, pcl::gpu::divUp(key_frames_position.size(), 256));
    kernelUpdateExpb<<<grid, block>>>(kf_b_values, key_frames_coefficients,
                                      key_frames_position);
  }
}

#endif  // USE_CUDA