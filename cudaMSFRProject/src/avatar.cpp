#include "avatar.h"

#include "BaselModel.h"
#include "solver.hpp"

void Avatar::rigid_transform(FrameParameters &parameter) {
  DLOG(INFO) << "Rigid Transform";
  cudaUpdateSRTPositionNormal(position_basel, normal_basel, mesh.cuda_position_,
                              mesh.cuda_normal_, cuda_rotation,
                              cuda_translation);
  cudaUpdateSRTPositionNormal(position_sRT, normal_R, position_basel,
                              normal_basel, parameter.cuda_rotation_,
                              parameter.cuda_translation_);
}

void Avatar::updateblendshapes(pcl::gpu::DeviceArray<float> exp_coefficient,
                               const bool is_smooth) {
  cudaProductMatVectorII(mesh.cuda_position_, cuda_blendshape, exp_coefficient,
                         mean);
  if (is_smooth) {
    cudaSmoothMesh(mesh.cuda_position_, mesh.cuda_tri_list_,
                   mesh.cudafvLookUpTable, mesh.cudafBegin, mesh.cudafEnd,
                   mesh.cuda_is_boundary_);
  }
  mesh.update_normal();
}
