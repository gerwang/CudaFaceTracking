#pragma once
#include <pcl\gpu\containers\device_array.h>
#include <pcl\gpu\utils\safe_call.hpp>
#include <cuda_runtime.h>

template<typename T>
inline void clearCudaMem(pcl::gpu::DeviceArray<T> mem) {
  cudaSafeCall(cudaMemset(mem.ptr(), 0,
    mem.size() * mem.elem_size));
}

template<typename T>
inline void clearCudaMem(pcl::gpu::DeviceArray2D<T> mem) {
  cudaSafeCall(cudaMemset(mem.ptr(), 0,
    mem.rows() * mem.cols() * mem.elem_size));
}

template<typename T>
inline void clearCudaMemAsync(pcl::gpu::DeviceArray<T> mem, cudaStream_t stream) {
  cudaSafeCall(cudaMemsetAsync(mem.ptr(), 0,
    mem.size() * mem.elem_size, stream));
}

template<typename T>
inline void clearCudaMemAsync(pcl::gpu::DeviceArray2D<T> mem) {
  cudaSafeCall(cudaMemsetAsync(mem.ptr(), 0,
    mem.rows() * mem.cols() * mem.elem_size, stream));
}