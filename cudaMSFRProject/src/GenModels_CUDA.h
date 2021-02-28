/*!
 * \file GenModels_CUDA.cpp
 * \date 2018/10/09 13:50
 *
 * \author sireer
 * Contact: sireerw@gmail.com
 *
 * \brief This codes were written by Zhibo Wang. The original codes are from
 * Ming Lu.This is a CUDA version which will be faster.
 *
 * TODO: This codes is not finished yet.
 *
 * \note
 */
#pragma once
#include "Common.h"
#ifdef USE_CUDA
#include "ObjMesh.h"
#include "VertexFormat.h"
#include "solver.hpp"

#include <map>
#include <vector>

#include <GL\glew.h>
#include <Eigen\Eigen>

#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <pcl\gpu\utils\safe_call.hpp>
#include "cuda_gl_interop.h"
//#include <GL\freeglut.h>

struct Model {
  unsigned int vao;
  std::vector<unsigned int> vbos;
  std::vector<cudaGraphicsResource*> cudaResourcesBuf;
  size_t size;
  int width, height;
  Model(){};
};

void cudaUpdateMesh(pcl::gpu::DeviceArray<float> verticesInfo,
                    const pcl::gpu::DeviceArray<float3> x,
                    const pcl::gpu::DeviceArray<float3> color,
                    const pcl::gpu::DeviceArray<float3> normal,
                    const pcl::gpu::DeviceArray<int3> tri_list);


void cudaUpdateMesh2(pcl::gpu::DeviceArray<float> verticesInfo,
                     const pcl::gpu::DeviceArray<float3> x,
                     const pcl::gpu::DeviceArray<float3> color,
                     const pcl::gpu::DeviceArray<float3> normal,
                     const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                     const pcl::gpu::DeviceArray<int1> fbegin,
                     const pcl::gpu::DeviceArray<int1> fend,
                     const pcl::gpu::DeviceArray<int> fv_idx);

class cudaGenModels {
 public:
  cudaGenModels() {
    // initialize cudaResourceDesc
    memset(&res_desc_, 0, sizeof(cudaResourceDesc));
    res_desc_.resType = cudaResourceTypeArray;

    // initialize cudaTextureDesc
    memset(&tex_desc_, 0, sizeof(cudaTextureDesc));
    tex_desc_.addressMode[0] = cudaAddressModeBorder;
    tex_desc_.addressMode[1] = cudaAddressModeBorder;
    tex_desc_.filterMode = cudaFilterModePoint;
    tex_desc_.readMode = cudaReadModeElementType;
    tex_desc_.normalizedCoords = 0;
  }
  ~cudaGenModels();


  void createMesh(const std::string& gameModelName,
                  const cudaObjMesh& ref_model);
  void updateMesh(const std::string& gameModelName,
                  const cudaObjMesh& ref_model);
  void updateMesh(const std::string& gameModelName,
                  const pcl::gpu::DeviceArray<float3> position,
                  const pcl::gpu::DeviceArray<float3> color,
                  const pcl::gpu::DeviceArray<float3> normal,
                  const pcl::gpu::DeviceArray<int3> tri_list);

  void updateMesh2(const std::string& gameModelName,
                   const pcl::gpu::DeviceArray<float3> position,
                   const pcl::gpu::DeviceArray<float3> color,
                   const pcl::gpu::DeviceArray<float3> normal,
                   const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                   const pcl::gpu::DeviceArray<int1> fbegin,
                   const pcl::gpu::DeviceArray<int1> fend,
                   const pcl::gpu::DeviceArray<int> fv_idx);

  void createImage(const std::string& imageName, const int width,
                   const int height);
  void getImage(const std::string& imageName, const int id,
                pcl::gpu::DeviceArray<float>& dst);
  void getImage(const std::string& imageName, const int id,
                pcl::gpu::DeviceArray2D<float>& dst);

  template <typename T>
  void getImage(const std::string& imageName, const int id,
                typename pcl::gpu::DeviceArray2D<T>& dst) {
    cudaArray* imageAdd;
    cudaSafeCall(cudaGraphicsMapResources(
        1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
    cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(
        &imageAdd, GameModelList[imageName].cudaResourcesBuf[id], 0, 0));
    cudaSafeCall(
        cudaMemcpyFromArray(reinterpret_cast<void*>(dst.ptr()), imageAdd, 0, 0,
                            GameModelList[imageName].size * dst.elem_size,
                            cudaMemcpyDeviceToDevice));
    cudaSafeCall(cudaGraphicsUnmapResources(
        1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
  }

  template <typename T>
  void setImage(const std::string& imageName, const int id,
                typename pcl::gpu::DeviceArray<T> src) {
    cudaArray* imageAdd;
    cudaGraphicsMapResources(1, &GameModelList[imageName].cudaResourcesBuf[id],
                             0);
    cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(
        &imageAdd, GameModelList[imageName].cudaResourcesBuf[id], 0, 0));
    cudaSafeCall(
        cudaMemcpyToArray(imageAdd, 0, 0, reinterpret_cast<void*>(src.ptr()),
                          GameModelList[imageName].size * src.elem_size,
                          cudaMemcpyDeviceToDevice));
    cudaSafeCall(cudaGraphicsUnmapResources(
        1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
  }

  template <typename T>
  void setImage(const std::string& imageName, const int id,
                typename pcl::gpu::DeviceArray2D<T> src) {
    cudaArray* imageAdd;
    cudaGraphicsMapResources(1, &GameModelList[imageName].cudaResourcesBuf[id],
                             0);
    cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(
        &imageAdd, GameModelList[imageName].cudaResourcesBuf[id], 0, 0));
    cudaSafeCall(
        cudaMemcpyToArray(imageAdd, 0, 0, reinterpret_cast<void*>(src.ptr()),
                          GameModelList[imageName].size * src.elem_size,
                          cudaMemcpyDeviceToDevice));

    cudaSafeCall(cudaGraphicsUnmapResources(
        1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
  }

  void DeleteModel(const std::string& gameModelName);
  unsigned int GetModel(const std::string& gameModelName);
  const unsigned int GenHandler(const std::string& gameModelName,
                                const int handlerId) {
    return GameModelList[gameModelName].vbos[handlerId];
  }

  std::map<const std::string, int> nTriFrontList;

 public:
  std::map<std::string, Model> GameModelList;
  friend class SafeMapTexObj;

 private:
  cudaResourceDesc res_desc_;
  cudaTextureDesc tex_desc_;

  cudaTextureObject_t MapTexObj(const std::string& imageName, const int id,
                                cudaTextureObject_t& texture_obj_) {
    cudaArray_t array;
    cudaSafeCall(cudaGraphicsMapResources(
                     1, &GameModelList[imageName].cudaResourcesBuf[id]),
                 0);
    cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(
        &array, GameModelList[imageName].cudaResourcesBuf[id], 0, 0));
    res_desc_.res.array.array = array;
    cudaSafeCall(
        cudaCreateTextureObject(&texture_obj_, &res_desc_, &tex_desc_, NULL));
    return texture_obj_;
  }
  void UnMapTexObj(const std::string& imageName, const int id,
                   cudaTextureObject_t& texture_obj_) {
    cudaSafeCall(cudaDestroyTextureObject(texture_obj_));
    cudaSafeCall(cudaGraphicsUnmapResources(
                     1, &GameModelList[imageName].cudaResourcesBuf[id]),
                 0);
  }
};

/*!
 * \class SafeMapTexObj
 *
 * \brief This could help to us cudaTextureObject safely.
 *
 * \author sireer
 * \date Oct 2018
 */
class SafeMapTexObj {
 public:
  SafeMapTexObj(cudaGenModels* gmodels, const std::string& gameModelName,
                const int handlerId)
      : gmodels_(gmodels),
        gameModelName_(gameModelName),
        handlerId_(handlerId) {
    gmodels_->MapTexObj(gameModelName_, handlerId_, texture_obj_);
  }

  // SafeMapTexObj(const SafeMapTexObj& obj):gmodels_(obj.gmodels_),
  //   gameModelName_(obj.gameModelName_),
  //   handlerId_(obj.handlerId_)
  // {}

  SafeMapTexObj(SafeMapTexObj&& obj)
      : gameModelName_(obj.gameModelName_), handlerId_(obj.handlerId_) {
    gmodels_ = obj.gmodels_;
    texture_obj_ = obj.texture_obj_;
    obj.gmodels_ = nullptr;
  }

  ~SafeMapTexObj() {
    if (gmodels_ != nullptr) {
      gmodels_->UnMapTexObj(gameModelName_, handlerId_, texture_obj_);
    }
  }

 private:
  cudaGenModels* gmodels_;
  const std::string gameModelName_;
  const int handlerId_;

 public:
  cudaTextureObject_t texture_obj_;

  // cudaTextureObject_t texture_obj_;
};

typedef cudaGenModels GenModels;
#endif  // USE_CUDA
