#pragma once
#include <Eigen\Eigen>
#include <fstream>
#include <string>
#include <vector>

#include "Common.h"
#include "ObjMesh.h"
#include "Parameters.h"

#define AVATAR_H
#ifdef AVATAR_H
class Avatar {
 public:
  Avatar(const std::string file_name,
         const int bs_num = NUM_EXPRESSION_INTEL_3DMM) {
    LOG(INFO) << "Load AVATAR " << file_name;
    LOG(INFO) << "BS Dim Num: " << bs_num;
    mesh.request_color_ = true;
    mesh.request_normal_ = false;
    mesh.request_tex_coord_ = false;
    mesh.request_position_ = true;
    mesh.load_obj(file_name + "mean.obj");
    mesh.update_normal();
    mesh.cuda_position_.copyTo(mean);

    blendshape.resize(mesh.n_verts_ * 3, bs_num);
    std::ifstream is(file_name + "blendshapes.bin", std::ios::binary);
    is.read((char *)blendshape.data(), blendshape.size() * sizeof(float));
    blendshape.transposeInPlace();
    cuda_blendshape.upload(blendshape.data(), blendshape.size());
    position_basel.create(mesh.n_verts_);
    normal_basel.create(mesh.n_verts_);
    position_sRT.create(mesh.n_verts_);
    normal_R.create(mesh.n_verts_);

    // load rigid transformation
    std::ifstream rigid_parameter(file_name + "sRT.txt");
    std::vector<float> rotation(9, 0), translation(4, 0);
    if (rigid_parameter.good()) {
      for (int i = 0; i < 9; ++i) {
        rigid_parameter >> rotation[i];
      }
      for (int i = 0; i < 4; ++i) {
        rigid_parameter >> translation[i];
      }
    } else {
      std::ifstream scale_file(file_name + "./sRT_list.txt");
      assert(scale_file.good());
      float scale;
      for (auto i = 0; i < 14; ++i) {
        scale_file >> scale;
      }
      scale /= 0.1f;

      blendshape *= scale;
      cuda_blendshape.upload(blendshape.data(), blendshape.size());
      mesh.position_ *= scale;
      mesh.position_.transposeInPlace();
      mesh.cuda_position_.upload((float3 *)mesh.position_.data(),
                                 mesh.n_verts_);
      mesh.cuda_position_.copyTo(mean);
      mesh.position_.transposeInPlace();
      mesh.update_normal();


      rotation[0] = 1;
      rotation[4] = 1;
      rotation[8] = 1;
      translation[3] = 1;
    }
    cuda_rotation.upload(rotation);
    cuda_translation.upload(translation);
  };

  Avatar(cudaObjMesh &obj) : mesh(obj) {
    position_basel.create(obj.n_verts_);
    normal_basel.create(obj.n_verts_);
    position_sRT.create(obj.n_verts_);
    normal_R.create(obj.n_verts_);
    std::vector<float> rotation(9), translation(4);
    rotation[0] = 1.0f;
    rotation[4] = 1.0f;
    rotation[8] = 1.0f;
    translation[3] = 1.0f;
    cuda_rotation.upload(rotation);
    cuda_translation.upload(translation);
  }

  cudaObjMesh mesh;
  void rigid_transform(FrameParameters &parameter);
  void updateblendshapes(pcl::gpu::DeviceArray<float> exp_coefficient,
                         const bool is_smooth = false);

  pcl::gpu::DeviceArray<float> cuda_blendshape;
  pcl::gpu::DeviceArray<float3> mean;
  pcl::gpu::DeviceArray<float3> position_basel;
  pcl::gpu::DeviceArray<float3> normal_basel;
  pcl::gpu::DeviceArray<float3> position_sRT;
  pcl::gpu::DeviceArray<float3> normal_R;
  pcl::gpu::DeviceArray<float> cuda_rotation;
  pcl::gpu::DeviceArray<float> cuda_translation;
  Eigen::MatrixXf blendshape;
};
#endif