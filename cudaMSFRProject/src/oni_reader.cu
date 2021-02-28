#include <cuda_runtime.h>
#include <iostream>
#include <pcl/gpu/containers/device_array.h>
#include "pcl/gpu/utils/safe_call.hpp"
#include "pcl/gpu/utils/limits.hpp"
#include "pcl/gpu/utils/cutil_math.h"
#include <device_launch_parameters.h>


/** \brief Cross-stream extrinsics: encode the topology describing how the different devices are connected. */
typedef struct rs2_extrinsics
{
  float rotation[9];    /**< Column-major 3x3 rotation matrix */
  float translation[3]; /**< Three-element translation vector, in meters */
} rs2_extrinsics;

/** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
typedef enum rs2_distortion
{
  RS2_DISTORTION_NONE, /**< Rectilinear images. No distortion compensation required. */
  RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
  RS2_DISTORTION_INVERSE_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
  RS2_DISTORTION_FTHETA, /**< F-Theta fish-eye distortion model */
  RS2_DISTORTION_BROWN_CONRADY, /**< Unmodified Brown-Conrady distortion model */
  RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_distortion;

/** \brief Video stream intrinsics */
typedef struct rs2_intrinsics
{
  int           width;     /**< Width of the image in pixels */
  int           height;    /**< Height of the image in pixels */
  float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
  float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
  float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
  float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
  rs2_distortion model;    /**< Distortion model of the image */
  float         coeffs[5]; /**< Distortion coefficients, order: k1, k2, p1, p2, k3 */
} rs2_intrinsics;


#define HALF_WIN_SIZE 5
#define DIST_LIMIT 0.10f

#define SIGMA_SPACE 4.5
#define SIGMA_COLOR 30
#define SIGMA_SPACE2_INV_HALF (0.5/(SIGMA_SPACE*SIGMA_SPACE))
#define SIGMA_COLOR2_INV_HALF (0.5/(SIGMA_COLOR*SIGMA_COLOR))
#define BILATERAL_FILTER_RADIUS 6
#define BILATERAL_FILTER_DIAMETER (2*BILATERAL_FILTER_RADIUS + 1)

__global__ void kernelConvertUshort2Float(pcl::gpu::PtrStepSz<float> output,
  const pcl::gpu::PtrStepSz<unsigned short> input,
  const float depth_scale)
{
  int xid = blockIdx.x*blockDim.x + threadIdx.x;
  int yid = blockIdx.y*blockDim.y + threadIdx.y;
  if (xid < output.cols && yid < output.rows)
  {
    output(yid, xid) = (float)input(yid, xid) * depth_scale;
    //output(yid, xid) /= 1000.0f; /// some inputs need to multiply 1.25 to get the correct depth
  }
}

void convertUshort2Float(pcl::gpu::DeviceArray2D<float> output,
  const pcl::gpu::DeviceArray2D<unsigned short> input,
  const float depth_scale)
{
  dim3 block(16, 16);
  dim3 grid(pcl::gpu::divUp(output.cols(), block.x),
    pcl::gpu::divUp(output.rows(), block.y));
  kernelConvertUshort2Float<<<grid, block>>>(output, input, depth_scale);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}

__global__ void kernelConvertUchar32Float3(pcl::gpu::PtrStepSz<float4> output,
  const pcl::gpu::PtrStepSz<uchar3> input)
{
  int xid = blockIdx.x*blockDim.x + threadIdx.x;
  int yid = blockIdx.y*blockDim.y + threadIdx.y;
  if (xid < output.cols && yid < output.rows)
  {
    output(yid, xid).x = (float)input(yid, xid).z / 255.0f;
    output(yid, xid).y = (float)input(yid, xid).y / 255.0f;
    output(yid, xid).z = (float)input(yid, xid).x / 255.0f;
    output(yid, xid).w = 1.0f;
  }
}


void convertUchar2Float(pcl::gpu::DeviceArray2D<float4> output,
  const pcl::gpu::DeviceArray2D<uchar3> input)
{
  dim3 block(16, 16);
  dim3 grid(pcl::gpu::divUp(output.cols(), block.x),
    pcl::gpu::divUp(output.rows(), block.y));
  kernelConvertUchar32Float3<<<grid, block>>>(output, input);
#if CUDA_GET_LAST_ERROR_AND_SYNC==1
  // device synchronize
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaStreamSynchronize(0));
#endif
}


__global__ void kernelConvertUint2Uchar(pcl::gpu::PtrStepSz<unsigned short> output,
  const pcl::gpu::PtrStepSz<unsigned int> input)
{
  int xid = blockIdx.x*blockDim.x + threadIdx.x;
  int yid = blockIdx.y*blockDim.y + threadIdx.y;
  if (xid < output.cols && yid < output.rows)
  {
    output(yid, xid) = (unsigned short)input(yid, xid);
  }
}

void convertUint2Uchar(pcl::gpu::DeviceArray2D<unsigned short> output,
  const pcl::gpu::DeviceArray2D<unsigned int> input)
{
  dim3 block(16, 16);
  dim3 grid(pcl::gpu::divUp(output.cols(), block.x),
    pcl::gpu::divUp(output.rows(), block.y));
  kernelConvertUint2Uchar<<<grid, block>>>(output, input);
}

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
__device__ __forceinline__ void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics * intrin, const float point[3])
{
  //assert(intrin->model != RS2_DISTORTION_INVERSE_BROWN_CONRADY); // Cannot project to an inverse-distorted image

  float x = point[0] / point[2], y = point[1] / point[2];

  if (intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY)
  {

    float r2 = x*x + y*y;
    float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2*r2 + intrin->coeffs[4] * r2*r2*r2;
    x *= f;
    y *= f;
    float dx = x + 2 * intrin->coeffs[2] * x*y + intrin->coeffs[3] * (r2 + 2 * x*x);
    float dy = y + 2 * intrin->coeffs[3] * x*y + intrin->coeffs[2] * (r2 + 2 * y*y);
    x = dx;
    y = dy;
  }
  if (intrin->model == RS2_DISTORTION_FTHETA)
  {
    float r = sqrtf(x*x + y*y);
    float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r* tan(intrin->coeffs[0] / 2.0f)));
    x *= rd / r;
    y *= rd / r;
  }

  pixel[0] = x * intrin->fx + intrin->ppx;
  pixel[1] = y * intrin->fy + intrin->ppy;
}


__device__ __forceinline__ void rs2_transform_point_to_point(float to_point[3], const struct rs2_extrinsics * extrin, const float from_point[3])
{
  to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] + extrin->rotation[6] * from_point[2] + extrin->translation[0];
  to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] + extrin->rotation[7] * from_point[2] + extrin->translation[1];
  to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] + extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

__global__ void kernel_set_to_max(pcl::gpu::PtrStepSz<unsigned int> input)
{
  int xid = blockIdx.x*blockDim.x + threadIdx.x;
  int yid = blockIdx.y*blockDim.y + threadIdx.y;
  if (xid < input.cols && yid < input.rows)
  {
    input(yid, xid) = 65535U;
  }
}
__global__ void kernel_set_max_to_zero(pcl::gpu::PtrStepSz<unsigned int> input)
{
  int xid = blockIdx.x*blockDim.x + threadIdx.x;
  int yid = blockIdx.y*blockDim.y + threadIdx.y;
  if (xid < input.cols && yid < input.rows)
  {
    if (input(yid, xid) == 65535)
    {
      input(yid, xid) = 0;
    }
  }
}
