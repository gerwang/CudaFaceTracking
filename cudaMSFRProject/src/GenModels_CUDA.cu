#include "Common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl\gpu\utils\safe_call.hpp>
#include <pcl\gpu\containers\device_array.h>

__global__ void kernelUpdateFaceModel(pcl::gpu::PtrSz<float> verticesInfo,
  const pcl::gpu::PtrSz<float3> x, const pcl::gpu::PtrSz<float3> color,
  const pcl::gpu::PtrSz<float3> normal, const pcl::gpu::PtrSz<int3> tri_list)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int fId = index;
  if (index < tri_list.size)
  {
    const int * vidx = &tri_list[index].x;
    index = index * 3;
    for (int i = 0; i < 3; i++)
    {
      int id = (index + i) * 14; // 14 is number of floats per vertex, magic number!
      int vId = __ldg(&vidx[i]);
      //float3 position = x[vId];
      //float4 albedo = make_float4(color[vId].x, color[vId].y, color[vId].z, 1.0f);
      //float3 n = normal[vId];
      *((float3*)&verticesInfo[id]) = x[vId];// position.x; verticesInfo[id + 1] = position.y; verticesInfo[id + 2] = position.z;
      *((float3*)&verticesInfo[id + 3]) = color[vId];
      *((float3*)&verticesInfo[id + 7]) = normal[vId];
      float3 weight{ 0.0f, 0.0f, 0.0f };
      *(reinterpret_cast<float*>(&weight) + i) = 1.0f;
      *reinterpret_cast<float3*>(&verticesInfo[id + 10]) = weight;
      verticesInfo[id + 13] = fId + 1; // plus 1 to distinguish with not rendered 0
      //verticesInfo[id + 3] = albedo.x; verticesInfo[id + 4] = albedo.y; verticesInfo[id + 5] = albedo.z;
      //verticesInfo[id + 6] = 1.0f;

      //verticesInfo[id+7] = n.x; verticesInfo[id + 8] = n.y; verticesInfo[id + 9] = n.z;
    }
  }
}

void cudaUpdateMesh(pcl::gpu::DeviceArray<float> verticesInfo,
  const pcl::gpu::DeviceArray<float3> x, 
  const pcl::gpu::DeviceArray<float3> color,
  const pcl::gpu::DeviceArray<float3> normal, 
  const pcl::gpu::DeviceArray<int3> tri_list)
{
  dim3 block(1024);
  dim3 grid(pcl::gpu::divUp(tri_list.size() * 3, block.x));
  kernelUpdateFaceModel<<<grid, block>>>(verticesInfo, x, color, normal, tri_list);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__
void kernelUpdateFaceModel2(pcl::gpu::PtrSz<float> verticesInfo,
	const pcl::gpu::PtrSz<float3> x,
	const pcl::gpu::PtrSz<float3> color,
	const pcl::gpu::PtrSz<float3> normal,
	const pcl::gpu::PtrSz<int2> fvLookUpTable,
	const pcl::gpu::PtrSz<int1> fbegin,
	const pcl::gpu::PtrSz<int1> fend,
	const pcl::gpu::PtrSz<int> fv_idx)
{
	const int vId = blockDim.x * blockIdx.x + threadIdx.x;
	if (vId < x.size)
	{
		float3 pos_i = x[vId];
		float3 color_i = color[vId];
		float3 normal_i = normal[vId];
		for (int i = fbegin[vId].x; i < fend[vId].x; ++i)
		{
			const int findex = __ldg(&fvLookUpTable[i].x);
			const int fv_idx_i = __ldg(&fv_idx[i]);
			int id = (findex * 3 + fv_idx_i) * 14;
			*((float3*)&verticesInfo[id]) = pos_i;// position.x; verticesInfo[id + 1] = position.y; verticesInfo[id + 2] = position.z;
			*((float3*)&verticesInfo[id + 3]) = color_i;
			*((float3*)&verticesInfo[id + 7]) = normal_i;
			float3 weight{ 0.0f, 0.0f, 0.0f };
			*(reinterpret_cast<float*>(&weight) + fv_idx_i) = 1.0f;
			*reinterpret_cast<float3*>(&verticesInfo[id + 10]) = weight;
			verticesInfo[id + 13] = findex + 1; // plus 1 to distinguish with not rendered 0
		}
	}
}

void cudaUpdateMesh2(pcl::gpu::DeviceArray<float> verticesInfo,
	const pcl::gpu::DeviceArray<float3> x,
	const pcl::gpu::DeviceArray<float3> color,
	const pcl::gpu::DeviceArray<float3> normal,
	const pcl::gpu::DeviceArray<int2> fvLookUpTable,
	const pcl::gpu::DeviceArray<int1> fbegin,
	const pcl::gpu::DeviceArray<int1> fend,
	const pcl::gpu::DeviceArray<int> fv_idx)
{
	dim3 block(1024);
	dim3 grid(pcl::gpu::divUp(x.size(), block.x));
	kernelUpdateFaceModel2 << <grid, block >> >(verticesInfo, x, color, normal, fvLookUpTable,
		fbegin, fend, fv_idx);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaStreamSynchronize(0));
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
	// device synchronize
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaStreamSynchronize(0));
#endif
}
#endif // USE_CUDA