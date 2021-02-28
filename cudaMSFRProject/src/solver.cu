/*!
* \file solver.cu
* \date 2018/10/12 15:51
*
* \author sireer
* Contact: sireerw@gmail.com
*
* \brief
*
* TODO: long description
*
* \note
*/

#pragma once
#include "Common.h"
#ifdef USE_CUDA
#include "MSFRUtil.cu"
#include "PclUtil.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <pcl\gpu\utils\safe_call.hpp>
#include <pcl\gpu\utils\cutil_math.h>
#include <pcl\gpu\containers\device_array.h>
#include <pcl\gpu\containers\kernel_containers.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>



#define M 4096
#define BIN_WIDTH 36
#define BIN_WIDTH2 6
#define BIN_LENGTH    32



__device__ float dev_reg_lambda;
__device__ int dev_Ii_size;
__device__ int dev_Iij_size;
__device__ int dev_row_num;
__device__ int dev_col_num;
__device__ int dev_nonzero_Iij;
__device__ int dev_ATA_rowptr_size;





__global__ void
kernelExtractNewWeightFromWeightMap(pcl::gpu::PtrSz<float> new_weight,
  const pcl::gpu::PtrStepSz<float3> weight_map,
  const pcl::gpu::PtrStepSz<float> tri_map,
  const pcl::gpu::PtrSz<int3> tri_list)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (col < tri_map.cols && row < tri_map.rows)
  {
    int3 triIdx = tri_list[__float2int_rd(tri_map(row, col) + 0.5)];
    float3 weight = weight_map(row, col);
    new_weight[triIdx.x] += weight.x;
    new_weight[triIdx.y] += weight.y;
    new_weight[triIdx.z] += weight.z;
  }
}

void cudaExtractNewWeightFromWeightMap(pcl::gpu::DeviceArray<float> new_weight,
  const pcl::gpu::DeviceArray2D<float3> weight_map,
  const pcl::gpu::DeviceArray2D<float> tri_map,
  const pcl::gpu::DeviceArray<int3> tri_list)
{
  clearCudaMem(new_weight);
  dim3 block(16, 16);
  dim3 grid(pcl::gpu::divUp(tri_map.cols(), block.x),
    pcl::gpu::divUp(tri_map.rows(), block.y));
  kernelExtractNewWeightFromWeightMap<<<grid, block>>>(new_weight, 
    weight_map, tri_map, tri_list);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void
kernelUpdateProjectionICPFromDepthImage(pcl::gpu::PtrSz<float3> projected_position,
  const pcl::gpu::PtrSz<float3> position_sRT,
  const pcl::gpu::PtrSz<unsigned short> is_front,
  const pcl::gpu::PtrStepSz<float> depth_image,
  const msfr::intrinsics camera_intr)
{
  __shared__ msfr::intrinsics camera;
  __shared__ int  width, height;
  int id = threadIdx.x;
  if (id == 0)
  {
    camera = camera_intr;
    width = depth_image.cols;
    height = depth_image.rows;
  }
  __syncthreads();

  int vId = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (vId < position_sRT.size)
  {
    projected_position[vId] = make_float3(-1.0f, -1.0f, -1.0f);
    if (is_front[vId] == 1)
    {

      float3 pos = position_sRT[vId];
      int2 uv = getProjectIndex(camera, pos);
      if (uv.x > 0 && uv.x < width && uv.y>0 && uv.y < height)
      {
        float depth = depth_image(uv.y, uv.x);
        if (depth > 0.0f && fabs(depth - pos.z) < 1e-2f) /// set sqrt(5)cm as the threshold
        {
          projected_position[vId] = unProjectedFromIndex(camera,
            make_float3(__int2float_rn(uv.x) + 0.5f, __int2float_rn(uv.y) + 0.5f, depth));
        }
      }
    }
  }
}

void cudaUpdateProjectionICPFromDepthImage(
  pcl::gpu::DeviceArray<float3> projected_position,
  const pcl::gpu::DeviceArray<float3> position_sRT,
  const pcl::gpu::DeviceArray<unsigned short> is_front,
  const pcl::gpu::DeviceArray2D<float> depth_image,
  const msfr::intrinsics camera_intr)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position_sRT.size(), block.x));
  kernelUpdateProjectionICPFromDepthImage<<<grid, block>>>(projected_position,
    position_sRT, is_front, depth_image, camera_intr);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  //std::vector<float3> host_position, host_target_position;
  //temp.download(host_position);
  //projected_position.download(host_target_position);
}




__global__ void kernelDownloadDepthMap(pcl::gpu::PtrSz<float3> dst,
  cudaTextureObject_t src, const int width, const int height)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < dst.size)
  {
    int row = id / width;
    int col = id - width * row;
    dst[id].x = tex2D<float>(src, col, row);
    dst[id].y = row;
    dst[id].z = col;
  }
}

__global__ void
kernelUpdateNormalICPFromDepthImage(pcl::gpu::PtrSz<float3> projected_position,
  const pcl::gpu::PtrSz<float3> position_sRT,
  const pcl::gpu::PtrSz<float3> normal_R,
  const pcl::gpu::PtrSz<unsigned short> is_front,
  const pcl::gpu::PtrStepSz<float> depth_image,
  const msfr::intrinsics camera_intr,
  const float threshold)
{
  __shared__ msfr::intrinsics camera;

  int id = threadIdx.x;
  if (id == 0)
  {
    camera = camera_intr;
  }
  __syncthreads();
  id += blockDim.x * blockIdx.x;
  if (id < position_sRT.size)
  {
    projected_position[id] = make_float3(-1.0f, -1.0f, -1.0f);
    float step = threshold; /// set sqrt(5)cm as the threshold
    float3 pos = position_sRT[id];
    float3 n = normal_R[id];
    n = n / length(n);
    
    int2 uv = getProjectIndex(camera, pos);
    float depth;
    if (is_front[id] == 1 && n.z < 0.0f)
    {
      float3 pos_f, pos_b;

      int2 uv_f, uv_b;
      unsigned short is_legal = 1; /// 1 represents legal
      if (depth_image(uv.y, uv.x) < pos.z)
      {
        pos_f = pos + step * n;
        pos_b = pos;
        uv_f = getProjectIndex(camera, pos_f);
        uv_b = uv;
        depth = depth_image(uv_f.y, uv_f.x);
        if (depth == 0.0f || depth < pos_f.z)
        {
          is_legal = 0;
        }
      }
      else
      {
        pos_b = pos - step * n;
        pos_f = pos;
        uv_f = uv;
        uv_b = getProjectIndex(camera, pos_b);
        depth = depth_image(uv_b.y, uv_b.x);
        if (depth == 0.0f || depth >= pos_b.z)
        {
          is_legal = 0;
        }
      }
      for (int i = 0; i < 5; ++i)
      {
        float3 mid = (pos_b + pos_f) / 2;
        uv = getProjectIndex(camera, pos);
        depth = depth_image(uv.y, uv.x);
        if (depth == 0.0f)
        {
          is_legal = 0;
        }
        if (depth < mid.z)
        {
          pos_b = mid;
          uv_b = uv;
        }
        else
        {
          pos_f = mid;
          uv_f = uv;
        }
      }
      if (is_legal == 1)
      {
        projected_position[id] = unProjectedFromIndex(camera, make_float3(uv.x + 0.5f, uv.y + 0.5f, depth));
      }
    }
  }
}


__global__ void
kernelUpdateNormalICPFromDepthImage(pcl::gpu::PtrSz<float3> projected_position,
  const pcl::gpu::PtrSz<float3> position_sRT,
  const pcl::gpu::PtrSz<float3> normal_R,
  const pcl::gpu::PtrSz<unsigned short> is_front,
  const cudaTextureObject_t depth_image,
  const msfr::intrinsics camera_intr,
  const float threshold)
{
  __shared__ msfr::intrinsics camera;

  int id = threadIdx.x;
  if (id == 0)
  {
    camera = camera_intr;
  }
  __syncthreads();
  id += blockDim.x * blockIdx.x;
  if (id < position_sRT.size)
  {
    projected_position[id] = make_float3(-1.0f, -1.0f, -1.0f);
    float step = threshold; /// set sqrt(5)cm as the threshold
    float3 pos = position_sRT[id];
    float3 n = normal_R[id];
    n = n / length(n);

    int2 uv = getProjectIndex(camera, pos);
    float depth;
    if (is_front[id] == 1 && n.z < 0.0f)
    {
      float3 pos_f, pos_b;

      int2 uv_f, uv_b;
      unsigned short is_legal = 1; /// 1 represents legal
      if (tex2D<float>(depth_image, uv.x, uv.y) < pos.z)
      {
        pos_f = pos + step * n;
        pos_b = pos;
        uv_f = getProjectIndex(camera, pos_f);
        uv_b = uv;
        depth = tex2D<float>(depth_image, uv_f.x, uv_f.y);
        if (depth == 0.0f || depth < pos_f.z)
        {
          is_legal = 0;
        }
      }
      else
      {
        pos_b = pos - step * n;
        pos_f = pos;
        uv_f = uv;
        uv_b = getProjectIndex(camera, pos_b);
        depth = tex2D<float>(depth_image, uv_b.x, uv_b.y);
        if (depth == 0.0f || depth >= pos_b.z)
        {
          is_legal = 0;
        }
      }
      for (int i = 0; i < 5; ++i)
      {
        float3 mid = (pos_b + pos_f) / 2;
        uv = getProjectIndex(camera, pos);
        depth = tex2D<float>(depth_image, uv.x, uv.y);
        if (depth == 0.0f)
        {
          is_legal = 0;
        }
        if (depth < mid.z)
        {
          pos_b = mid;
          uv_b = uv;
        }
        else
        {
          pos_f = mid;
          uv_f = uv;
        }
      }
      if (is_legal == 1)
      {
        projected_position[id] = unProjectedFromIndex(camera, make_float3(uv.x + 0.5f, uv.y + 0.5f, depth));
      }
    }
  }
}

void cudaUpdateNormalICPFromDepthImage(pcl::gpu::DeviceArray<float3> projected_position, 
  const pcl::gpu::DeviceArray<float3> position_sRT,
  const pcl::gpu::DeviceArray<float3> normal_R,
  const pcl::gpu::DeviceArray<unsigned short> is_front,
  const cudaTextureObject_t depth_image, 
  const msfr::intrinsics camera_intr,
  const float threshold)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position_sRT.size(), block.x));
  kernelUpdateNormalICPFromDepthImage<<<grid, block>>>(projected_position,
    position_sRT, normal_R, is_front, depth_image, camera_intr, threshold);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

void cudaUpdateNormalICPFromDepthImage(pcl::gpu::DeviceArray<float3> projected_position,
  const pcl::gpu::DeviceArray<float3> position_sRT, 
  const pcl::gpu::DeviceArray<float3> normal_R,
  const pcl::gpu::DeviceArray<unsigned short> is_front, 
  const pcl::gpu::DeviceArray2D<float> depth_image, 
  const msfr::intrinsics camera_intr,
  const float threshold)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position_sRT.size(), block.x));
  kernelUpdateNormalICPFromDepthImage<<<grid, block>>>(projected_position,
    position_sRT, normal_R, is_front, depth_image, camera_intr, threshold);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}


__global__ void kernelUpdateClosestPointfromDepthImage(pcl::gpu::PtrSz<float3> projected_position,
  const pcl::gpu::PtrSz<float3> position_RT,
  const pcl::gpu::PtrSz<unsigned short> is_front,
  const cudaTextureObject_t depth_image,
  const msfr::intrinsics camera_intr,
  const int width_, const int height_,
  const float threshold_sq)
{
  __shared__ msfr::intrinsics camera;
  __shared__ int width, height;
  if (threadIdx.x == 0)
  {
    camera = camera_intr;
    width = width_;
    height = height_;
  }
  __syncthreads();
  int vid = blockDim.x * blockIdx.x + threadIdx.x;
  if (vid < position_RT.size)
  {
    
    float min_dis2 = threshold_sq; /// 
    float3 nearest_pos = make_float3(-1.0f, -1.0f, -1.0f);
    if (is_front[vid] == 1)
    {
      float3 pos_i = position_RT[vid];
      int2 uv = getProjectIndex(camera, pos_i);
      int min_x = max(uv.x - 10, 1);
      int max_x = min(uv.x + 11, width - 1);
      int min_y = max(uv.y - 10, 1);
      int max_y = min(uv.y + 11, height - 1);
      for (int i = min_x; i < max_x; ++i)
      {
        for (int j = min_y; j < max_y; ++j)
        {
          float depth = tex2D<float>(depth_image, i, j);
          if (depth > 0.0f)
          {
            float3 pos_ij = unProjectedFromIndex(camera,
              make_float3(__float2int_rn(i) + 0.5f, __float2int_rn(j) + 0.5f,
                depth));
            float dis2_ij = norm2(pos_i - pos_ij);
            if (dis2_ij < min_dis2)
            {
              min_dis2 = dis2_ij;
              nearest_pos = pos_ij;
            }
          }
        }
      }
    }
    projected_position[vid] = nearest_pos;
  }
}

void cudaUpdateClosestPointfromDepthImage(pcl::gpu::DeviceArray<float3> projected_position,
  const pcl::gpu::DeviceArray<float3> position_RT,
  const pcl::gpu::DeviceArray<unsigned short> is_front,
  const cudaTextureObject_t depth_image,
  const msfr::intrinsics camera_intr,
  const int width, const int height,
  const float threshold)
{
  const float threshold_sq = threshold * threshold;
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position_RT.size(), block.x));
  kernelUpdateClosestPointfromDepthImage<<<grid, block>>>(projected_position,
    position_RT, is_front, depth_image, camera_intr, width, height, threshold_sq);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}


__global__ void kernelUpdateClosestPointfromDepthImageNew(
    pcl::gpu::PtrSz<float3> projected_position,
    const pcl::gpu::PtrSz<float3> position_RT,
    const cudaTextureObject_t depth_image, const msfr::intrinsics camera_intr,
    const int width_, const int height_, const float threshold_sq) {
  __shared__ msfr::intrinsics camera;
  __shared__ int width, height;
  if (threadIdx.x == 0) {
    camera = camera_intr;
    width = width_;
    height = height_;
  }
  __syncthreads();
  int vid = blockDim.x * blockIdx.x + threadIdx.x;
  if (vid < position_RT.size) {
    float min_dis2 = threshold_sq;  ///
    float3 nearest_pos = make_float3(-1.0f, -1.0f, -1.0f);
    {
      float3 pos_i = position_RT[vid];
      int2 uv = getProjectIndex(camera, pos_i);
      int min_x = max(uv.x - 10, 1);
      int max_x = min(uv.x + 11, width - 1);
      int min_y = max(uv.y - 10, 1);
      int max_y = min(uv.y + 11, height - 1);
      for (int i = min_x; i < max_x; ++i) {
        for (int j = min_y; j < max_y; ++j) {
          float depth = tex2D<float>(depth_image, i, j);
          if (depth > 0.0f) {
            float3 pos_ij = unProjectedFromIndex(
                camera, make_float3(__float2int_rn(i) + 0.5f,
                                    __float2int_rn(j) + 0.5f, depth));
            float dis2_ij = norm2(pos_i - pos_ij);
            if (dis2_ij < min_dis2) {
              min_dis2 = dis2_ij;
              nearest_pos = pos_ij;
            }
          }
        }
      }
    }
    projected_position[vid] = nearest_pos;
  }
}

void cudaUpdateClosestPointfromDepthImageNew(
    pcl::gpu::DeviceArray<float3> projected_position,
    const pcl::gpu::DeviceArray<float3> position_RT,
    const cudaTextureObject_t depth_image, const msfr::intrinsics camera_intr,
    const int width, const int height, const float threshold) {
  const float threshold_sq = threshold * threshold;
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position_RT.size(), block.x));
  kernelUpdateClosestPointfromDepthImageNew<<<grid, block>>>(
      projected_position, position_RT, depth_image, camera_intr,
      width, height, threshold_sq);
#if CUDA_GET_LAST_ERROR_AND_SYNC == 1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}


__global__ void kernelUpdateClosestPointfromDepthImageWithNormalConstraint(
	pcl::gpu::PtrSz<float3> projected_position,	
	const pcl::gpu::PtrSz<float3> position_RT,
	const pcl::gpu::PtrSz<float3> normal_R,
	const pcl::gpu::PtrSz<unsigned short> is_front,
	const cudaTextureObject_t depth_image,
	const msfr::intrinsics camera_intr,
	const int width_, const int height_,
	const float threshold_sq)
{
	__shared__ msfr::intrinsics camera;
	__shared__ int width, height;
	if (threadIdx.x == 0)
	{
		camera = camera_intr;
		width = width_;
		height = height_;
	}
	__syncthreads();
	int vid = blockDim.x * blockIdx.x + threadIdx.x;
	if (vid < position_RT.size)
	{

		float min_dis2 = threshold_sq; /// 
		float3 nearest_pos = make_float3(-1.0f, -1.0f, -1.0f);
		float3 pos_i = position_RT[vid];
		float3 normal_i = normal_R[vid];
		if (is_front[vid] == 1 && normal_i.z < -0.2f)
		{
			int2 uv = getProjectIndex(camera, pos_i);
			int min_x = max(uv.x - 10, 1);
			int max_x = min(uv.x + 11, width - 1);
			int min_y = max(uv.y - 10, 1);
			int max_y = min(uv.y + 11, height - 1);
			for (int i = min_x; i < max_x; ++i)
			{
				for (int j = min_y; j < max_y; ++j)
				{
					float depth = tex2D<float>(depth_image, i, j);
					if (depth > 0.0f)
					{
						float3 pos_ij = unProjectedFromIndex(camera,
							make_float3(__float2int_rn(i) + 0.5f, __float2int_rn(j) + 0.5f,
								depth));
						float dis2_ij = norm2(pos_i - pos_ij);
						float cos_i = abs(dot(pos_i - pos_ij, normal_i) / sqrtf(dis2_ij));
						if (dis2_ij < min_dis2 && cos_i < 0.5f)
						{
							min_dis2 = dis2_ij;
							nearest_pos = pos_ij;
						}
					}
				}
			}
		}
		projected_position[vid] = nearest_pos;
	}
}

void cudaUpdateClosestPointfromDepthImageWithNormalConstraint(
	pcl::gpu::DeviceArray<float3> projected_position,
	const pcl::gpu::DeviceArray<float3> position_RT,
	const pcl::gpu::DeviceArray<float3> normal_R, 
	const pcl::gpu::DeviceArray<unsigned short> is_front,
	const cudaTextureObject_t depth_image,
	const msfr::intrinsics camera_intr,
	const int width, const int height,
	const float threshold)
{
	const float threshold_sq = threshold * threshold;
	dim3 block(1024);
	dim3 grid(pcl::gpu::divUp(position_RT.size(), block.x));
	kernelUpdateClosestPointfromDepthImageWithNormalConstraint<<<grid, block>>>(projected_position,
		position_RT, normal_R, is_front, depth_image,
		camera_intr, width, height, threshold_sq);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
	// device synchronize
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__device__ __forceinline__ bool is_in_0_1(float a, float delta = 0.0f) {
  return a >= -delta && a <= 1.0f + delta;
}

__global__ void kernelRenderMesh(pcl::gpu::PtrStepSz<float> canvas,
  const pcl::gpu::PtrSz<float3> position,
  const pcl::gpu::PtrSz<float> rotation_,
  const pcl::gpu::PtrSz<float> translation_,
  const msfr::intrinsics camera_)
{
  __shared__ msfr::intrinsics camera;
  __shared__ float rotation[9];
  __shared__ float3 translation;
  __shared__ int width, height;
  int id = threadIdx.x;
  if (id == 1)
  {
    camera = camera_;
    translation.x = translation_[0];
    translation.y = translation_[1];
    translation.z = translation_[2];
    width = canvas.cols;
    height = canvas.rows;
  }
  if (id < 9)
  {
    rotation[id] = rotation_[id] * translation_[3];
  }
  __syncthreads();
  id += blockDim.x*blockIdx.x;
  if (id < position.size)
  {
    float3 pos = M33xV3(rotation, position[id]) + translation;
    int2 uv = getProjectIndex(camera, pos);
    if (uv.x>=0 && uv.x< width && uv.y>=0 && uv.y< height)
    canvas(uv.y, uv.x) += pos.z;
  }
}

void cudaRenderMesh(pcl::gpu::DeviceArray2D<float> canvas, 
  const pcl::gpu::DeviceArray<float3> position, 
  const pcl::gpu::DeviceArray<float> rotation, 
  const pcl::gpu::DeviceArray<float> translation, 
  const msfr::intrinsics & camera)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position.size(), block.x));
  clearCudaMem(canvas);
  kernelRenderMesh<<<grid, block>>>(canvas, position,
    rotation, translation, camera);
}

__global__ void kernelUpdateInvSRTTargetPosition(pcl::gpu::PtrSz<float3> target_position_inv_sRT,
  const pcl::gpu::PtrSz<float3> target_position,
  const pcl::gpu::PtrSz<float> lambda_position,
  const pcl::gpu::PtrSz<float> rotation_,
  const pcl::gpu::PtrSz<float> translation_)
{
  __shared__ float rotation[9];
  __shared__ float3 translation;
  int id = threadIdx.x;
  if (id == 0)
  {
    translation.x = translation_[0];
    translation.y = translation_[1];
    translation.z = translation_[2];
  }
  if (id < 9)
  {
    rotation[id] = rotation_[id] / translation_[3];
  }
  __syncthreads();
  id += blockDim.x * blockIdx.x;
  if (id < target_position.size)
  {
    float lambda_i = __ldg(&lambda_position[id]);
    if (lambda_i > 0.0f)
    {
      target_position_inv_sRT[id] = M33TxV3(rotation, target_position[id]
        - lambda_i * translation);
    }
    else
    {
      target_position_inv_sRT[id] = make_float3(0.0f, 0.0f, 0.0f);
    }
  }
}

void cudaUpdateInvSRTTargetPosition(pcl::gpu::DeviceArray<float3> target_position_inv_sRT,
  const pcl::gpu::DeviceArray<float3> target_position,
  const pcl::gpu::DeviceArray<float> lambda_position,
  const pcl::gpu::DeviceArray<float> rotation,
  const pcl::gpu::DeviceArray<float> translation)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(target_position.size(), block.x));
  kernelUpdateInvSRTTargetPosition<<<grid, block>>>(target_position_inv_sRT, target_position,
    lambda_position, rotation, translation);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}


__global__ void kernelUpdateInvSRTProjectionPosition(pcl::gpu::PtrSz<float3> projection_position_inv_sRT,
  const pcl::gpu::PtrSz<float3> projection_position,
  const pcl::gpu::PtrSz<float> rotation_,
  const pcl::gpu::PtrSz<float> translation_)
{
  __shared__ float rotation[9];
  __shared__ float3 translation;
  int id = threadIdx.x;
  if (id == 0)
  {
    translation.x = translation_[0];
    translation.y = translation_[1];
    translation.z = translation_[2];
  }
  if (id < 9)
  {
    rotation[id] = rotation_[id] / translation_[3];
  }
  __syncthreads();
  id += blockDim.x * blockIdx.x;
  if (id < projection_position_inv_sRT.size)
  {
    if (projection_position[id].z > 0.0f)
    {
      projection_position_inv_sRT[id] = M33TxV3(rotation, projection_position[id]
        - translation);
    }
    else
    {
      projection_position_inv_sRT[id] = make_float3(0.0f, 0.0f, 0.0f);
    }
  }
}

void cudaUpdateInvSRTProjectionPosition(pcl::gpu::DeviceArray<float3> projection_position_inv_sRT,
  const pcl::gpu::DeviceArray<float3> projection_position,
  const pcl::gpu::DeviceArray<float> rotation,
  const pcl::gpu::DeviceArray<float> translation)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(projection_position.size(), block.x));
  kernelUpdateInvSRTProjectionPosition<<<grid, block>>>(projection_position_inv_sRT, projection_position,
    rotation, translation);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelSRTPositionNormal(pcl::gpu::PtrSz<float3> position_RT,
  pcl::gpu::PtrSz<float3> normal_R,
  const pcl::gpu::PtrSz<float3> position,
  const pcl::gpu::PtrSz<float3> normal,
  const pcl::gpu::PtrSz<float> rotation_,
  const pcl::gpu::PtrSz<float> translation_)
{
  __shared__ float rotation[9], rotation_scale[9];
  __shared__ float3 translation;
  __shared__ float scale;
  int id = threadIdx.x;
  if (id == 0)
  {
    translation.x = translation_[0];
    translation.y = translation_[1];
    translation.z = translation_[2];
    scale = translation_[3];
  }
  if (id < 9)
  {
    rotation[id] = rotation_[id];
    rotation_scale[id] = rotation[id] * scale;
  }
  __syncthreads();
  id += blockDim.x * blockIdx.x;
  if (id < position.size)
  {
    position_RT[id] = M33xV3(rotation_scale, position[id]) + translation;
    normal_R[id] = M33xV3(rotation, normal[id]);
  }
}

void cudaUpdateSRTPositionNormal(pcl::gpu::DeviceArray<float3> position_RT,
  pcl::gpu::DeviceArray<float3> normal_R,
  const pcl::gpu::DeviceArray<float3> position,
  const pcl::gpu::DeviceArray<float3> normal,
  const pcl::gpu::DeviceArray<float> rotation,
  const pcl::gpu::DeviceArray<float> translation)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(position.size(), block.x));
  kernelSRTPositionNormal<<<grid, block>>>(position_RT, normal_R,
    position, normal, rotation, translation);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  //std::vector<float3> host_position_RT, host_position;
  //std::vector<float> host_rotation, host_translation;
  //position_RT.download(host_position_RT);
  //position.download(host_position);
  //rotation.download(host_rotation);
  //translation.download(host_translation);
}




// Dense Matrix Compute: The grid number equal to the size of x
__device__ __forceinline__ void computeAx_i(float & xi,
  const pcl::gpu::PtrSz<float> &x,
  const pcl::gpu::PtrSz<float> &A)
{
  int threadDim = (gridDim.x + 31) >> 5; /// x.size equals to gridDim.x
  int beginId = threadDim * threadIdx.x;
  int endId = min(gridDim.x, beginId + threadDim);
  xi = 0;
  for (int i = beginId, offset = blockIdx.x*x.size; i < endId; ++i)
  {
    xi += A[offset + i] * x[i];
  }
  __syncthreads();
  xi = warp_scan(xi);
}

__global__ void prepare_r_p(pcl::gpu::PtrSz<float> r,
  pcl::gpu::PtrSz<float> p,
  const pcl::gpu::PtrSz<float> x,
  const pcl::gpu::PtrSz<float> A,
  const pcl::gpu::PtrSz<float> b)
{
  float xi;
  computeAx_i(xi, x, A);
  if (threadIdx.x == 31)
  {
    r[blockIdx.x] = b[blockIdx.x] - xi;
    p[blockIdx.x] = r[blockIdx.x];
  }
}

__global__ void kernelComputeAx(pcl::gpu::PtrSz<float> Ax,
  const pcl::gpu::PtrSz<float> A,
  const pcl::gpu::PtrSz<float> x)
{
  float xi;
  computeAx_i(xi, x, A);
  if (threadIdx.x == 31)
  {
    Ax[blockIdx.x] = xi;
  }
}

__device__ __forceinline__
float warp_up_scan8(float data)
{
  data += __shfl_up(data, 1);
  data += __shfl_up(data, 2);
  data += __shfl_up(data, 4);
  return data;
}

__device__ __forceinline__
void compute_uTvi(float & uTv, const pcl::gpu::PtrSz<float> & u,
  const pcl::gpu::PtrSz<float> & v)
{
  __shared__ float partial_sum[8];
  uTv = 0;
  if (threadIdx.x < u.size)
  {
    uTv = u[threadIdx.x] * v[threadIdx.x];
  }
  __syncthreads();
  uTv = warp_scan(uTv);
  if ((threadIdx.x & 31) == 31)
  {
    partial_sum[threadIdx.x >> 5] = uTv;
  }
  __syncthreads();
  if (threadIdx.x < 8)
  {
    uTv = partial_sum[threadIdx.x];
    uTv = warp_up_scan8(uTv);
  }
}

__global__ void kernelCGIter(pcl::gpu::PtrSz<float> x,
  pcl::gpu::PtrSz<float> r,
  pcl::gpu::PtrSz<float> p,
  const pcl::gpu::PtrSz<float> Ap)
{
  __shared__ float alpha, beta;
  float rTri, pTApi;
  compute_uTvi(rTri, r, r);
  compute_uTvi(pTApi, p, Ap);
  if (threadIdx.x == 7)
  {
    alpha = rTri / pTApi;
  }

  __syncthreads();

  if (threadIdx.x < x.size)
  {
    x[threadIdx.x] += alpha*p[threadIdx.x];
    r[threadIdx.x] -= alpha*Ap[threadIdx.x];
  }
  float new_rTri;
  compute_uTvi(new_rTri, r, r);
  if (threadIdx.x == 7)
  {
    beta = new_rTri / rTri;
  }

  __syncthreads();
  if (threadIdx.x < x.size)
  {
    p[threadIdx.x] = beta * p[threadIdx.x] + r[threadIdx.x];
  }
}

void cudaCGSolver(pcl::gpu::DeviceArray<float> x,
  const pcl::gpu::DeviceArray<float> A,
  const pcl::gpu::DeviceArray<float> b,
  int nIters)
{
  pcl::gpu::DeviceArray<float> p(b.size()), r(b.size()), Ap(b.size());
  dim3 block(32);
  dim3 grid(x.size());

  prepare_r_p<<<grid, block>>>(r, p, x, A, b);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
  
  for (int i = 0; i < nIters; i++)
  {
    kernelComputeAx<<<grid, block>>>(Ap, A, p);

#if CUDA_GET_LAST_ERROR_AND_SYNC==1
    // device synchronize
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaStreamSynchronize(0));
#endif
    kernelCGIter<<<1, 256>>>(x, r ,p, Ap);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
    // device synchronize
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaStreamSynchronize(0));
#endif
  }
}

#define USE_PRECONDITION 0 // fixme why? why?!!
#define PCG_EVAL_RESIDUAL 0

__global__ void prepare_r_p_z(pcl::gpu::PtrSz<float> r,
  pcl::gpu::PtrSz<float> p,
  pcl::gpu::PtrSz<float> z,
  const pcl::gpu::PtrSz<float> x,
  const pcl::gpu::PtrSz<float> A,
  const pcl::gpu::PtrSz<float> b,
  pcl::gpu::PtrSz<float> j) {
  float xi;
  computeAx_i(xi, x, A);
  if(threadIdx.x==31) {
    auto& rTmp = r[blockIdx.x];
    rTmp = b[blockIdx.x] - xi;
    p[blockIdx.x] = rTmp;
#if USE_PRECONDITION
    j[blockIdx.x] = A[blockIdx.x*(x.size + 1)];
    z[blockIdx.x] = rTmp / j[blockIdx.x];
#else
    z[blockIdx.x] = rTmp;
#endif
  }
}

__global__ void kernelPCGIter(pcl::gpu::PtrSz<float> x,
  pcl::gpu::PtrSz<float> r,
  pcl::gpu::PtrSz<float> p,
  pcl::gpu::PtrSz<float> z,
  const pcl::gpu::PtrSz<float> j,
  const pcl::gpu::PtrSz<float> Ap) {
  __shared__ float alpha, beta;
  float zTri, pTApi;
  compute_uTvi(zTri, z, r);
  compute_uTvi(pTApi, p, Ap);
  if (threadIdx.x == 7)
  {
    alpha = zTri / pTApi;
  }

  __syncthreads();

  if (threadIdx.x < x.size) {
    auto& ri = r[threadIdx.x];
    x[threadIdx.x] += alpha * p[threadIdx.x];
    ri -= alpha * Ap[threadIdx.x];
#if USE_PRECONDITION
    z[threadIdx.x] = ri / j[threadIdx.x];
#else
    z[threadIdx.x] = ri;
#endif
  }
  float new_zTri;
  compute_uTvi(new_zTri, z, r);
  if (threadIdx.x == 7)
  {
    beta = new_zTri / zTri;
  }

  __syncthreads();
  if (threadIdx.x < x.size)
  {
    p[threadIdx.x] = beta * p[threadIdx.x] + z[threadIdx.x];
  } 
}


void cudaPCGSolver(pcl::gpu::DeviceArray<float> x,
  const pcl::gpu::DeviceArray<float> A,
  const pcl::gpu::DeviceArray<float> b,
  int nIters,
  cudaStream_t stream) {
  pcl::gpu::DeviceArray<float> p(b.size()), r(b.size()), Ap(b.size()),z(b.size()), j(x.size());
  dim3 block(32);
  dim3 grid(x.size());
  prepare_r_p_z << <grid, block, 0, stream >> > (r, p, z, x, A, b, j);
  for (int k = 0; k < nIters; k++) {
    kernelComputeAx << <grid, block, 0, stream >> > (Ap, A, p);
    kernelPCGIter<<<1,256,0,stream>>>(x, r, p, z, j, Ap);
  }
}

__global__ void kernelShootingSolve(pcl::gpu::PtrSz<float> x,
  pcl::gpu::PtrSz<float> S,
  const pcl::gpu::PtrSz<float> ATA,
  const float lambda)
{
  const int x_size = x.size;
  const int tId = threadIdx.x;
  __shared__ float delta_x_i;
  if (tId < x_size)
  {
    for (int i = 0; i < 100; ++i)
    {
      for (int j = 0; j < x_size; ++j)
      {
        if (tId == j)
        {
          float ATA_jj = __ldg(&ATA[j * x_size + j]);
          float x_i = __ldg(&x[j]);
          float prev_x_i = x_i;
          float S_0 = -2 * (__ldg(&S[j]) + x_i * ATA_jj);
          if (S_0 > lambda)
          {
            x_i = (lambda - S_0) / (2 * ATA_jj);
          }
          else if (S_0 < -lambda)
          {
            x_i = -(lambda + S_0) / (2 * ATA_jj);
          }
          else
          {
            x_i = 0.0f;
          }
          if (x_i < 0.0f)
          {
            x_i = 0.0f;
          }
          if (x_i > 1.f)
          {
            x_i = 1.0f;
          }
          delta_x_i = x_i - prev_x_i;
          x[j] = x_i;
        }
        __syncthreads();
        S[tId] -= delta_x_i * __ldg(&ATA[tId * x_size + j]);
      }
    }
  }
}

void cudaShootingSolve(pcl::gpu::DeviceArray<float> x,
  pcl::gpu::DeviceArray<float> S,
  const pcl::gpu::DeviceArray<float> ATA,
  const float lambda)
{
 /* cudaSafeCall(cudaStreamSynchronize(0));
  std::vector<float> host_x, host_S, host_ATA;
  x.download(host_x); S.download(host_S); ATA.download(host_ATA);
  for (int i = 0; i < 10; ++i)
  {
    for (int j = 0; j < host_x.size(); ++j)
    {
      float x_i = host_x[j];
      float ATA_jj = host_ATA[j*x.size() + j];
      float prev_x_i = x_i;
      float S_0 = - 2 * (host_S[j] + x_i * ATA_jj);
      if (S_0 > lambda)
      {
        x_i = (lambda - S_0) / (2 * ATA_jj);
      }
      else if (S_0 < -lambda)
      {
        x_i = -(lambda + S_0) / (2 * ATA_jj);
      }
      else
      {
        x_i = 0.0f;
      }
      if (x_i < 0.0f)
      {
        x_i = 0.0f;
      }
      if (x_i > 1.f)
      {
        x_i = 1.0f;
      }
      host_x[j] = x_i;
      float delta_x_i = x_i - prev_x_i;
      for (int k = 0; k < x.size(); ++k)
      {
        host_S[k] -= delta_x_i * host_ATA[k*x.size() + j];
      }
    }
  }*/
  kernelShootingSolve<<<1, x.size()>>> (x, S, ATA, lambda);
  std::cout << lambda << std::endl;
}

__global__ void kernalSolveRMxb(pcl::gpu::PtrSz<float> B,
  const int mB, const int ldB,
  const pcl::gpu::PtrSz<float> R,
  const int nR, const int ldR,
  const pcl::gpu::PtrSz<float> Mask,
  const int R_colIdx)
{
  extern __shared__ float R_col[];
  const int B_colIdx = threadIdx.x + blockDim.x * blockIdx.x;
  if (threadIdx.x < nR)
  {
    R_col[threadIdx.x] = __ldg(&R[ldR*R_colIdx + threadIdx.x]);
  }
  __syncthreads();
  
  if (B_colIdx < mB)
  {
    const float Mask_ii = __ldg(&Mask[nR * B_colIdx + R_colIdx]);
    float * B_col = &B[ldB * B_colIdx];

    if (Mask_ii > 0.0f)
    {
      float b_ii = B_col[R_colIdx] /= R_col[R_colIdx];
      for (int i = 0; i < R_colIdx; ++i)
      {
        B_col[i] -= b_ii * R_col[i];
      }
    }
  }
}

void cudaSolveRMxb(pcl::gpu::DeviceArray<float> B,
  const int mB, const int ldB,
  const pcl::gpu::DeviceArray<float> R,
  const int nR, const int ldR, 
  const pcl::gpu::DeviceArray<float> Mask,
  const cudaStream_t stream)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(mB, block.x));
  for (int i = 0; i < nR; ++i)
  {
    //std::cout << i << ": " << nR << std::endl;
    kernalSolveRMxb<<<grid, block, nR*sizeof(float), stream>>>(B, mB, ldB, R, nR, ldR, Mask, i);
    cudaSafeCall(cudaStreamSynchronize(stream));
  }
}

__global__ void kernelSORUpdateA(pcl::gpu::PtrSz<float> A,
  const pcl::gpu::PtrSz<float> A_,
  const int n,
  const float omega)
{
  const int rowIdx = threadIdx.x + blockDim.x * blockIdx.x;
  const int colIdx = threadIdx.y + blockDim.y * blockIdx.y;
  if (rowIdx < n && colIdx < A_.size)
  {
    if (rowIdx == colIdx)
    {
      A[colIdx * n + rowIdx] = __ldg(&A_[colIdx * n + rowIdx]);
    }
    else
    {
      A[colIdx * n + rowIdx] = __ldg(&A_[colIdx * n + rowIdx]) * omega;
    }
  }
}

__global__ void kernelSORUpdateb(pcl::gpu::PtrSz<float> b,
  const pcl::gpu::PtrSz<float> b_,
  const pcl::gpu::PtrSz<float> D,
  const pcl::gpu::PtrSz<float> B,
  const int n,
  const float omega)
{
  const int rowIdx = threadIdx.x + blockDim.x * blockIdx.x;
  const int colIdx = threadIdx.y + blockDim.y * blockIdx.y;
  const int D_colIdx = colIdx / 3;
  if (rowIdx < n && colIdx*n<b_.size)
  {
    const int index = colIdx * n + rowIdx;
    b[index] = (__ldg(&b_[index]) + __ldg(&D[D_colIdx * n + rowIdx]) * __ldg(&B[index])) * omega;
  }
}

void cudaSORUpdateAb(pcl::gpu::DeviceArray<float> A,
  pcl::gpu::DeviceArray<float> b,
  const pcl::gpu::DeviceArray<float> A_,
  const pcl::gpu::DeviceArray<float> b_,
  const pcl::gpu::DeviceArray<float> D,
  const pcl::gpu::DeviceArray<float> B,
  const int n,
  const float omega,
  const cudaStream_t stream)
{
  {
    dim3 block(8, 8);
    dim3 grid(pcl::gpu::divUp(n, block.x), pcl::gpu::divUp(n, block.y));
    kernelSORUpdateA<<<grid, block, 0, stream>>>(A, A_, n, omega);
    //cudaSafeCall(cudaStreamSynchronize(0));
  }
  {
    dim3 block(8, 8);
    dim3 grid(pcl::gpu::divUp(n, block.x), pcl::gpu::divUp(b.size()/n, block.y));
    kernelSORUpdateb<<<grid, block, 0, stream>>>(b, b_, D, B, n, omega);
    //cudaSafeCall(cudaStreamSynchronize(0));
  }
}

__global__ void kernelSORIter(pcl::gpu::PtrSz<float> X_,
  pcl::gpu::PtrSz<float> X_assist,
  const pcl::gpu::PtrSz<float> X,
  const pcl::gpu::PtrSz<float> b,
  const pcl::gpu::PtrSz<float> A,
  const pcl::gpu::PtrSz<float> D,
  const pcl::gpu::PtrSz<float> error,
  const int n,
  const float omega)
  //pcl::gpu::PtrSz<float> temp_diag)
{
  extern __shared__ float A_row[];
  float diag;
  const int colIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int X_offset = colIdx * n;
  const int D_colIdx_offset = colIdx / 3 * n;
  const int col_max = X.size / n;
  float result, comp;
  float error_i = 1.0f;
  if (colIdx < col_max)
  {
    error_i = sqrtf(__ldg(&error[colIdx]) / n);
  }
  for (int i = 0; i < n; ++i)
  {
    
    if (threadIdx.x < n)
    {
      A_row[threadIdx.x] = __ldg(&A[i*n + threadIdx.x]);
    }
    __syncthreads();
    if (colIdx < col_max)
    {
      if (error_i > 1e-7f)
      {
        diag = __ldg(&D[D_colIdx_offset + i]) + A_row[i];
        result = 0.0f;
        comp = 0.0f;
        for (int j = i + 1; j < n; ++j)
        {
          float t;
          //result -= A_row[j] * __ldg(&X[X_offset + j]);
          comp -= -A_row[j] * __ldg(&X[X_offset + j]);
          t = result - comp;
          comp = (t - result) + comp;
          result = t;
        }
        for (int j = 0; j < i; ++j)
        {
          float t;
          comp -= -A_row[j] * X_[X_offset + j];
          t = result - comp;
          comp = (t - result) + comp;
          result = t;
        }
        {
          float t;
          //result += __ldg(&b[X_offset + i]);// +X_assist[X_offset + i];
          comp -= __ldg(&b[X_offset + i]);
          t = result - comp;
          comp = (t - result) + comp;
          result = t;
          result /= diag;
          result = result + (1.0f - omega) * __ldg(&X[X_offset + i]);
          //result += (1.0f - omega) * __ldg(&X[X_offset + i]);
          //comp -= (1.0f - omega) * diag * __ldg(&X[X_offset + i]);
          //t = result - comp;
          //comp = (t - result) + comp;
          //result = t;

        }
        //for (int j = i + 1; j < n; ++j)
        //{
        //  X_assist[X_offset + j] -= A_row[j] * result;
        //}
        X_[X_offset + i] = result;
        //temp_diag[X_offset + i] = diag;
      }
      else
      {
        X_[X_offset + i] = __ldg(&X[X_offset + i]);
      }
    } 
    __syncthreads();
  }
}


void cudaSORIter(pcl::gpu::DeviceArray<float> & X,
  pcl::gpu::DeviceArray<float> & X_,
  pcl::gpu::DeviceArray<float> X_assist,
  const pcl::gpu::DeviceArray<float> A,
  const pcl::gpu::DeviceArray<float> b,
  const pcl::gpu::DeviceArray<float> D,
  const pcl::gpu::DeviceArray<float> error,
  const int n, const float omega,
  const cudaStream_t stream)
{
  {
    //std::vector<float> host_X, host_X_, host_diag, host_D;
    //X.download(host_X);
    //clearCudaMemAsync(X_assist, stream);
    //pcl::gpu::DeviceArray<float> diag(X.size());
    dim3 block(1024);
    dim3 grid(pcl::gpu::divUp(X_.size() / n, block.x));
    kernelSORIter<<<grid, block, n * sizeof(float), stream>>>(X_, X_assist, X, b, A, D, error, n, omega);
    cudaSafeCall(cudaStreamSynchronize(stream));
		X_.swap(X);
    //
    //X.download(host_X_);
    ////diag.download(host_diag);
    //D.download(host_D);
    //for (int i = 0; i < n; ++i)
    //{
    //  std::cout << i << " " << host_D[32352 * n + i] << std::endl;
    //}
    //std::cout << 1;
  }
}

__device__ __forceinline__ float square(float x)
{
  return x*x;
}

__global__ void kernelSORComputeError(pcl::gpu::PtrSz<float> error,
  const pcl::gpu::PtrSz<float> A,
  const pcl::gpu::PtrSz<float> X,
  const pcl::gpu::PtrSz<float> b,
  const pcl::gpu::PtrSz<float> D,
  const int n)
{
  extern __shared__ float A_row[];
  const int colIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int b_offset = colIdx * n;
  const int D_colIdx_offset = colIdx / 3 * n;
  const int col_max = X.size / n;
  float result, comp, error_i = 0.0f;
  for (int i = 0; i < n; ++i)
  {
    if (threadIdx.x < n)
    {
      A_row[threadIdx.x] = __ldg(&A[i*n + threadIdx.x]);
    }
    __syncthreads();
    if (colIdx < col_max)
    {
      float diag = __ldg(&D[D_colIdx_offset + i]);
      result = 0.0f;
      comp = 0.0f;
      for (int j = 0; j < n; ++j)
      {
        float t;
        //result += A_row[j] * __ldg(&X[b_offset + j]);
        comp -= A_row[j] * __ldg(&X[b_offset + j]);
        t = result - comp;
        comp = (t - result) + comp;
        result = t;
      }
      float t;
      //result += diag * __ldg(&X[b_offset + i]);
      comp -= diag * __ldg(&X[b_offset + i]);
      t = result - comp;
      comp = (t - result) + comp;
      result = t;
      error_i += square(result - __ldg(&b[b_offset + i]));
    }
    __syncthreads();
  }
  if (colIdx < col_max)
  {
    error[colIdx] = error_i;
  }
}

void cudaSORComputeError(pcl::gpu::DeviceArray<float> error,
  const pcl::gpu::DeviceArray<float> A,
  const pcl::gpu::DeviceArray<float> X,
  const pcl::gpu::DeviceArray<float> b,
  const pcl::gpu::DeviceArray<float> D,
  const int n,
  const cudaStream_t stream)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(error.size(), block.x));
  kernelSORComputeError<<<grid, block, n * sizeof(float), stream>>>(error, A, X, b, D, n);
  cudaSafeCall(cudaStreamSynchronize(stream));
  //std::vector<float> host_error;
  //error.download(host_error);
  //float sum = 0.0f;
  //for (auto &iter : host_error)
  //{
  //  sum += iter;
  //}
  //std::cout << sqrtf(host_error[32352 * 3]/n) << std::endl;
}



#endif // USE_CUDA
