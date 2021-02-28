#pragma once
#include "Common.h"
#ifdef USE_CUDA
/*!
 * \file solver.h
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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>

#include "HostUtil.hpp"
#include "ObjMesh.h"
#include "PclUtil.h"
#include "pcl\gpu\utils\safe_call.hpp"

//
// template<typename T>
// inline void clearCudaMem(pcl::gpu::DeviceArray<T> mem) {
//  cudaSafeCall(cudaMemset(mem.ptr(), 0,
//    mem.size() * mem.elem_size));
//}

#pragma region CUDA_FUNCTION_DECLARATION
void cudaExtractNewWeightFromWeightMap(
    pcl::gpu::DeviceArray<float> new_weight,
    const pcl::gpu::DeviceArray2D<float3> weight_map,
    const pcl::gpu::DeviceArray2D<float> tri_map,
    const pcl::gpu::DeviceArray<int3> tri_list);

void cudaUpdateProjectionICPFromDepthImage(
    pcl::gpu::DeviceArray<float3> projected_position,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const pcl::gpu::DeviceArray2D<float> depth_image,
    const msfr::intrinsics camera_intr);

void cudaUpdateNormalICPFromDepthImage(
    pcl::gpu::DeviceArray<float3> projected_position,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const cudaTextureObject_t depth_image, const msfr::intrinsics camera_intr,
    const float threshold);

void cudaUpdateNormalICPFromDepthImage(
    pcl::gpu::DeviceArray<float3> projected_position,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const pcl::gpu::DeviceArray2D<float> depth_image,
    const msfr::intrinsics camera_intr, const float threshold);

void cudaUpdateClosestPointfromDepthImage(
    pcl::gpu::DeviceArray<float3> projected_position,
    const pcl::gpu::DeviceArray<float3> position_RT,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const cudaTextureObject_t depth_image, const msfr::intrinsics camera_intr,
    const int width, const int height, const float threshold);

void cudaUpdateClosestPointfromDepthImageNew(
    pcl::gpu::DeviceArray<float3> projected_position,
    const pcl::gpu::DeviceArray<float3> position_RT,
    const cudaTextureObject_t depth_image, const msfr::intrinsics camera_intr,
    const int width, const int height, const float threshold);

void cudaUpdateClosestPointfromDepthImageWithNormalConstraint(
    pcl::gpu::DeviceArray<float3> projected_position,
    const pcl::gpu::DeviceArray<float3> position_RT,
    const pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const cudaTextureObject_t depth_image, const msfr::intrinsics camera_intr,
    const int width, const int height, const float threshold);

void cudaRenderMesh(pcl::gpu::DeviceArray2D<float> canvas,
                    const pcl::gpu::DeviceArray<float3> position,
                    const pcl::gpu::DeviceArray<float> rotation,
                    const pcl::gpu::DeviceArray<float> translation,
                    const msfr::intrinsics &camera);

void cudaUpdateInvSRTTargetPosition(
    pcl::gpu::DeviceArray<float3> target_position_inv_sRT,
    const pcl::gpu::DeviceArray<float3> target_position,
    const pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> translation);

void cudaUpdateInvSRTProjectionPosition(
    pcl::gpu::DeviceArray<float3> projection_position_inv_sRT,
    const pcl::gpu::DeviceArray<float3> projection_position,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> translation);

void cudaUpdateSRTPositionNormal(
    pcl::gpu::DeviceArray<float3> position_RT,

    pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<float3> position,
    const pcl::gpu::DeviceArray<float3> normal,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> translation);

// solve Ax=b problem using a CG method.
void cudaCGSolver(pcl::gpu::DeviceArray<float> x,
                  const pcl::gpu::DeviceArray<float> A,
                  const pcl::gpu::DeviceArray<float> b, int nIters = 5);

void cudaPCGSolver(pcl::gpu::DeviceArray<float> x,
                   const pcl::gpu::DeviceArray<float> A,
                   const pcl::gpu::DeviceArray<float> b, int nIters,
                   cudaStream_t stream);

void cudaShootingSolve(pcl::gpu::DeviceArray<float> x,
                       pcl::gpu::DeviceArray<float> S,
                       const pcl::gpu::DeviceArray<float> ATA,
                       const float lambda);

void cudaSolveRMxb(pcl::gpu::DeviceArray<float> B, const int mB, const int ldB,
                   const pcl::gpu::DeviceArray<float> R, const int nR,
                   const int ldR, const pcl::gpu::DeviceArray<float> Mask,
                   const cudaStream_t stream);

void cudaSORUpdateAb(pcl::gpu::DeviceArray<float> A,
                     pcl::gpu::DeviceArray<float> b,
                     const pcl::gpu::DeviceArray<float> A_,
                     const pcl::gpu::DeviceArray<float> b_,
                     const pcl::gpu::DeviceArray<float> D,
                     const pcl::gpu::DeviceArray<float> B, const int n,
                     const float omega, const cudaStream_t stream);

void cudaSORIter(pcl::gpu::DeviceArray<float> &X,
                 pcl::gpu::DeviceArray<float> &X_,
                 pcl::gpu::DeviceArray<float> X_assist,
                 const pcl::gpu::DeviceArray<float> A,
                 const pcl::gpu::DeviceArray<float> b,
                 const pcl::gpu::DeviceArray<float> D,
                 const pcl::gpu::DeviceArray<float> error, const int n,
                 const float omega, const cudaStream_t stream);

void cudaSORComputeError(pcl::gpu::DeviceArray<float> error,
                         const pcl::gpu::DeviceArray<float> A,
                         const pcl::gpu::DeviceArray<float> X,
                         const pcl::gpu::DeviceArray<float> b,
                         const pcl::gpu::DeviceArray<float> D, const int n,
                         const cudaStream_t stream);
#pragma endregion CUDA_FUNCTION_DECLARATION

class Solver {
 public:
  void solveDenseLeastSquare(pcl::gpu::DeviceArray<float> x,
                             const pcl::gpu::DeviceArray<float> A,
                             const pcl::gpu::DeviceArray<float> b) {
    cudaCGSolver(x, A, b);
  }
};

// This mesh should have the topography as any mesh in the 3DMM.
class meshToBeSolved : public cudaObjMesh {
 public:
  meshToBeSolved() : cudaObjMesh() { clear_lambda(); }

  meshToBeSolved(const cudaObjMesh &obj)
      : cudaObjMesh(obj),
        cuda_position_weight_(obj.n_verts_),
        cuda_position_new_weight_(obj.n_verts_),
        cuda_optimal_weight_(obj.n_verts_),
        lambda_position_(obj.n_verts_),
        lambda_landmark_(obj.n_verts_),
        target_position_(obj.n_verts_),
        target_landmark_(obj.n_verts_),
        target_position_inv_sRT(obj.n_verts_),
        target_landmark_inv_sRT(obj.n_verts_),
        projection_position_(obj.n_verts_),
        projection_position_inv_sRT(obj.n_verts_),
        cuda_position_sRT(obj.n_verts_) {
    clear_lambda();
  }

 public:
  // weights are for the position in the canonical depth
  pcl::gpu::DeviceArray<float> cuda_position_weight_, cuda_position_new_weight_,
      cuda_optimal_weight_;

  // solve the coefficients
  pcl::gpu::DeviceArray<float> lambda_position_, lambda_landmark_;
  pcl::gpu::DeviceArray<float3> target_position_, target_landmark_;

  pcl::gpu::DeviceArray<float3> target_position_inv_sRT,
      target_landmark_inv_sRT;

  pcl::gpu::DeviceArray<float3> projection_position_;
  pcl::gpu::DeviceArray<float3> projection_position_inv_sRT;
  pcl::gpu::DeviceArray<float3> cuda_position_sRT;
  pcl::gpu::DeviceArray<float3> cuda_normal_R;

  bool is_lambda_clear_;
  void clear_lambda() {
    clearCudaMem(lambda_position_);
    clearCudaMem(target_position_);
    clearCudaMem(target_position_inv_sRT);
    clearCudaMem(lambda_landmark_);
    clearCudaMem(target_landmark_);
    clearCudaMem(target_landmark_inv_sRT);
    is_lambda_clear_ = true;
  }

  virtual void init() {
    cudaObjMesh::init();
    estimate_geoposition();
    geoposition_.transposeInPlace();
    geoposition_.transposeInPlace();
    cuda_position_weight_.create(n_verts_);
    cuda_position_new_weight_.create(n_verts_);
    cuda_optimal_weight_.create(n_verts_);
    lambda_position_.create(n_verts_);
    target_position_.create(n_verts_);
    target_position_inv_sRT.create(n_verts_);
    lambda_landmark_.create(n_verts_);
    target_landmark_.create(n_verts_);
    target_landmark_inv_sRT.create(n_verts_);
    projection_position_.create(n_verts_);
    projection_position_inv_sRT.create(n_verts_);
    cuda_position_sRT.create(n_verts_);
    cuda_normal_R.create(n_verts_);
  }

  virtual const pcl::gpu::DeviceArray<float3> &position() const {
    return cuda_position_;
  }

  virtual const pcl::gpu::DeviceArray<float3> &normal() const {
    return cuda_normal_;
  }

  void updateInvSRTTargetPosition(
      const pcl::gpu::DeviceArray<float> rotation,
      const pcl::gpu::DeviceArray<float> translation) {
    cudaUpdateInvSRTTargetPosition(target_position_inv_sRT, target_position_,
                                   lambda_position_, rotation, translation);
    cudaUpdateInvSRTTargetPosition(target_landmark_inv_sRT, target_landmark_,
                                   lambda_landmark_, rotation, translation);
  }

  void updateInvSRTTargetOnlyPosition(
      const pcl::gpu::DeviceArray<float> rotation,
      const pcl::gpu::DeviceArray<float> translation) {
    cudaUpdateInvSRTTargetPosition(target_position_inv_sRT, target_position_,
                                   lambda_position_, rotation, translation);
  }

  void updateInvSRTTargetOnlyLandmark(
      const pcl::gpu::DeviceArray<float> rotation,
      const pcl::gpu::DeviceArray<float> translation) {
    cudaUpdateInvSRTTargetPosition(target_landmark_inv_sRT, target_landmark_,
                                   lambda_landmark_, rotation, translation);
  }

  void updateInvSRTProjectionPosition(
      const pcl::gpu::DeviceArray<float> rotation,
      const pcl::gpu::DeviceArray<float> translation) {
    cudaUpdateInvSRTProjectionPosition(projection_position_inv_sRT,
                                       projection_position_, rotation,
                                       translation);
  }

  void updateSRTPositionNormal(const pcl::gpu::DeviceArray<float> rotation,
                               const pcl::gpu::DeviceArray<float> translation) {
    cudaUpdateSRTPositionNormal(cuda_position_sRT, cuda_normal_R, position(),
                                normal(), rotation, translation);
  }



  void getProjectionICP(const pcl::gpu::DeviceArray2D<float> depth_map,
                        const msfr::intrinsics &camera) {
    cudaUpdateProjectionICPFromDepthImage(projection_position_,
                                          cuda_position_sRT, cuda_is_front_now,
                                          depth_map, camera);
  }

  void getNormalICP(const pcl::gpu::DeviceArray2D<float> depth_map,
                    const msfr::intrinsics &camera) {
    cudaUpdateNormalICPFromDepthImage(projection_position_, cuda_position_sRT,
                                      cuda_normal_R, cuda_is_front_now,
                                      depth_map, camera, 1e-2f);
  }

  void renderImage(const pcl::gpu::DeviceArray<float> rotation,
                   const pcl::gpu::DeviceArray<float> translation,
                   const pcl::gpu::DeviceArray2D<float> canvas,
                   const msfr::intrinsics &camera) {
    cudaRenderMesh(canvas, position(), rotation, translation, camera);
  }
};

class cudaShootingSolver {
 public:
  void solve(pcl::gpu::DeviceArray<float> x, pcl::gpu::DeviceArray<float> S,
             const pcl::gpu::DeviceArray<float> ATA, const float lambda) {
    cudaShootingSolve(x, S, ATA, lambda);
  }
};

#endif  // USE_CUDA