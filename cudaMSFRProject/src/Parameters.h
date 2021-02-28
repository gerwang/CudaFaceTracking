#pragma once

#include <glm/glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <iostream>
#include <vector>
#include "Common.h"
#include "dual_quaternion.hpp"
#include "vector_operations.hpp"

#ifdef USE_CUDA
#include <pcl\gpu\containers\device_array.h>
#include <HostUtil.hpp>
#include <pcl\gpu\utils\\safe_call.hpp>
#include "PclUtil.h"
#endif  // USE_CUDA

// template<typename T>
// inline void clearCudaMem(pcl::gpu::DeviceArray<T> mem) {
//  cudaSafeCall(cudaMemset(mem.ptr(), 0,
//    mem.size() * mem.elem_size));
//}

struct FrameBlock {
  FrameBlock()
      : identity_coef_(NUM_IDENTITY_3DMM),
        expr_coef_(NUM_EXPRESSION_INTEL_3DMM),
#ifdef USE_CUDA
        cuda_id_coefficients_(NUM_IDENTITY_3DMM),
        cuda_exp_coefficients_(NUM_EXPRESSION_INTEL_3DMM),
        cuda_exp_prev_1_(NUM_EXPRESSION_INTEL_3DMM),
        cuda_exp_prev_2_(NUM_EXPRESSION_INTEL_3DMM)
#endif  // USE_CUDA
  {
    clearCudaMem(cuda_id_coefficients_);
    clearCudaMem(cuda_exp_coefficients_);
    clearCudaMem(cuda_exp_prev_1_);
    clearCudaMem(cuda_exp_prev_2_);
  }
  double R_x, R_y, R_z, T_x, T_y, T_z;
  Eigen::VectorXf identity_coef_, color_coef_, expr_coef_;
  double shParams[27];
  double shParams_false[27];
  double shParams_whitebalance[9];
  double colorTransform[NUM_COLORTRANSFORM_3DMM];

  void setIdParams(Eigen::VectorXf params) { identity_coef_ = params; }
  void setAlbedoParams(Eigen::VectorXf params) { color_coef_ = params; }
  void setExprParams(Eigen::VectorXf params) { expr_coef_ = params; }
  void setSHParams(double* params) {
    for (int kk = 0; kk < 27; kk++) shParams[kk] = params[kk];
  }

  void saveRT(std::string file_name, float orthoScale);

#ifdef USE_CUDA
  pcl::gpu::DeviceArray<float> cuda_id_coefficients_, cuda_exp_coefficients_;
  pcl::gpu::DeviceArray<float> cuda_exp_prev_1_, cuda_exp_prev_2_;
  void cudaSetIDParams();
  void cudaSetEXPParams();
#endif  // USE_CUDA
};

struct CameraBlock {
 public:
  static constexpr float cameraSpeed = 0.00005f;
  static constexpr double MouseSensitivity = 0.1f;

  CameraBlock() { updateCameraVectors(); }
  void lock_camera_view() { is_lock_camera_view = true;
  }
  void unlock_camera_view() { is_lock_camera_view = false;
  }
  glm::mat4 GetViewMatrix() {
    return glm::lookAt(this->position, this->position + this->front, this->up);
  }
  void ON_PRESS_W(float delay_time) {
    if (!is_lock_camera_view) position -= delay_time * cameraSpeed * front;
  }
  void ON_PRESS_S(float delay_time) {
    if (!is_lock_camera_view) position += delay_time * cameraSpeed * front;
  };
  void ON_PRESS_A(float delay_time) {
    if (!is_lock_camera_view) position += right * cameraSpeed * delay_time;
  };
  void ON_PRESS_D(float delay_time) {
    if (!is_lock_camera_view) position -= right * cameraSpeed * delay_time;
  };

  void ON_PRESS_R() {
    position = glm::vec3(0.0f, 0.0f, 0.0f);
    front = glm::vec3(0.0f, 0.0f, -1.0f);
    up = glm::vec3(0.0f, 1.0f, 0.0f);
    Yaw = -90.0f;
    Pitch = 0.0f;
    updateCameraVectors();
  }

  void on_mouse_move(double xoffset, double yoffset) {
    if (!is_lock_camera_view) {
      xoffset *= MouseSensitivity;
      yoffset *= MouseSensitivity;

      Yaw += xoffset;
      Pitch += yoffset;

      // Make sure that when pitch is out of bounds, screen doesn't get flipped

      if (Pitch > 89.0f) Pitch = 89.0f;
      if (Pitch < -89.0f) Pitch = -89.0f;

      // Update Front, Right and Up Vectors using the updated Euler angles
      updateCameraVectors();
    }
  }

  void updateCameraVectors() {
    // Calculate the new Front vector
    glm::vec3 Font;
    Font.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Font.y = sin(glm::radians(Pitch));
    Font.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front = glm::normalize(Font);
    // Also re-calculate the Right and Up vector
    right = glm::normalize(glm::cross(
        front,
        glm::vec3(0.0f, 1.0f,
                  0.0f)));  // Normalize the vectors, because their length gets
                            // closer to 0 the more you look up or down which
                            // results in slower movement.
    up = glm::normalize(glm::cross(right, front));
  }
  msfr::intrinsics camera_intr;

  void set_position(glm::vec3 position_) { position = position_; }
  void set_front(glm::vec3 front_) { front = front_; }
  void set_up(glm::vec3 up_) { up = up_; }

  const glm::vec3 get_position() const { return position; }
  const glm::vec3 get_front() const { return front; }
  const glm::vec3 get_up() const { return up; }

  double aspect;
  double orthoScale;
  double zFar;
  float Yaw = -90.0f;
  float Pitch = 0.0f;
  void setScaleParams(double params) { orthoScale = params; }

 private:
  glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
  glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec3 right;
  bool is_lock_camera_view = false;
};

class FrameParameters {
 public:
  FrameParameters()
      : cuda_rotation_(9),
        cuda_translation_(4),
        prev_RT_1(mat34::identity()),
        prev_RT_2(mat34::identity()) {
    modelParams.color_coef_ = Eigen::VectorXf(199);
    ResetParams();

    resetCUDARotation();
    resetCUDATranslation();
    // clearCudaMem(cuda_rotation_);
    // clearCudaMem(cuda_translation_);
  }
  void resetCUDATranslation() {
    std::vector<float> translation(4, 0);
    translation[3] = 0.10f;
    cuda_translation_.upload(translation);
  }
  void resetCUDARotation() {
    std::vector<float> matrix_identity(9, 0); 
    matrix_identity[0] = 1.0f;
    matrix_identity[4] = 1.0f;
    matrix_identity[8] = 1.0f;
    cuda_rotation_.upload(matrix_identity);
  }
  void ResetParams();
  void InitSH();
  void SetPoseParams(std::vector<double>& poseParams);
  void GetPoseParams(std::vector<double>& poseParams);
  void SetIdentityParams(std::vector<double>& idParams);
  void GetIdentityParams(std::vector<double>& idParams);
  void SetExpressionParams(std::vector<double>& exprParams);
  void GetExpressionParams(std::vector<double>& exprParams);
  void SetAlbedoParams(std::vector<double>& albParams);
  void GetAlbedoParams(std::vector<double>& albParams);
  void SetSHparams(std::vector<double>& shParams);
  void GetSHparams(std::vector<double>& shParams);
  glm::mat4& UpdateModelMat();
  glm::mat4& UpdateProjectionMat();
  void setOrdinaryShParams(std::vector<double>& shParams);
  void push_present_RT();
  void smooth_RT();
  void get_present_RT();

 public:
  FrameBlock modelParams;
  CameraBlock cameraParams;
  glm::mat4 model_mat_, projection_;
  glm::mat4 translateMat, rotateMat, scaleMat;
  double SH[9];
  std::vector<double> SH_vec;
  pcl::gpu::DeviceArray<float> cuda_rotation_, cuda_translation_;
  std::vector<float> host_rotation, host_translation;
  DualQuaternion present_RT, prev_RT_1, prev_RT_2;
  bool is_consis_1, is_consis_2;
  mat34 present_RT_mat34;
  float lambda = 0.6, lambda_1 = 0.6, lambda_2 = 0.36, lambda_3 = 10.0f;
};

class JointVideoParams {
 public:
  JointVideoParams() {}
  ~JointVideoParams() {}

  void setFrameParams(std::vector<FrameBlock>& params,
                      std::vector<double> errColor,
                      std::vector<double> errShape);
  void setCamParams(std::vector<CameraBlock>& params,
                    std::vector<double> errShape);
  void update();
  bool checkIdConverged(double thres);
  bool checkAlbedoConverged(double thres);
  bool checkSHConverged(double thres);

  void backup();

 public:
  std::vector<CameraBlock> camParams;
  std::vector<FrameBlock> frameParams;
  std::vector<double> errPixel;
  std::vector<double> errLandmark;

  Eigen::VectorXf idParams, albedoParams;
  double shParams[27];

  Eigen::VectorXf idParamsLast, albedoParamsLast;
  double shParamsLast[27];
};

/// Parameter util functions
void loadParams(std::vector<FrameBlock>& fP, std::vector<CameraBlock>& cP,
                CameraBlock def_c, FrameBlock def_f, int n,
                std::string filename);
void saveParams(std::vector<FrameBlock>& fP, std::vector<CameraBlock>& cP,
                int n, std::string filename);
FrameBlock getSmoothFrameParams(std::vector<FrameBlock>& videoFrameParams,
                                int numFrame, int frameIdx);
