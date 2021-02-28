#include "Parameters.h"

#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\euler_angles.hpp>

#include <fstream>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif  // USE_CUDA

const double R_x_default = 0.0;
const double R_y_default = 0.0;
const double R_z_default = 0.0;
const double T_x_default = 0.0;
const double T_y_default = 0.0;
const double T_z_default = 0.0;
const double Focal_default = 3.5;
const double orthoScale_default = 1;
const double zFar_default = 1000;

void FrameParameters::ResetParams() {
  InitSH();

  // Pose
  modelParams.R_x = R_x_default;
  modelParams.R_y = R_y_default;
  modelParams.R_z = R_z_default;
  modelParams.T_x = T_x_default;
  modelParams.T_y = T_y_default;
  modelParams.T_z = T_z_default;
  // Color transfer
  double ct[] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (int i = 0; i < NUM_COLORTRANSFORM_3DMM; i++)
    modelParams.colorTransform[i] = ct[i];
  // 3DMM
  modelParams.identity_coef_.setZero();
  modelParams.expr_coef_.setZero();
  modelParams.color_coef_.setZero();
  // SH Lighting
  modelParams.shParams[0] = modelParams.shParams[1] = modelParams.shParams[2] =
      sqrt(4.0 * M_PI);
  modelParams.shParams_whitebalance[0] = sqrt(4.0 * M_PI);
  for (int kk = 3; kk < 27; kk++) modelParams.shParams[kk] = 0;
  for (int kk = 1; kk < 9; kk++) modelParams.shParams_whitebalance[kk] = 0;
  // Camera
  cameraParams.orthoScale = orthoScale_default;
  cameraParams.zFar = zFar_default;
  cameraParams.aspect = 424.0 / 512.0;
  UpdateModelMat();
  UpdateProjectionMat();
}

void FrameParameters::InitSH() {
  SH[0] = 1.0 / sqrt(4 * M_PI);
  SH[1] = (2.0 * M_PI / 3.0) * sqrt(3.0 / (4 * M_PI));
  SH[2] = (2.0 * M_PI / 3.0) * sqrt(3.0 / (4 * M_PI));
  SH[3] = (2.0 * M_PI / 3.0) * sqrt(3.0 / (4 * M_PI));
  ;
  SH[4] = M_PI / 8.0 * sqrt(5.0 / (4 * M_PI));
  SH[5] = (3.0 * M_PI / 4.0) * sqrt(5 / (12 * M_PI));
  SH[6] = (3.0 * M_PI / 4.0) * sqrt(5 / (12 * M_PI));
  SH[7] = (3.0 * M_PI / 4.0) * sqrt(5 / (12 * M_PI));
  SH[8] = (3.0 * M_PI / 8.0) * sqrt(5 / (12 * M_PI));
  SH_vec.assign(SH, SH + 9);
}

void FrameParameters::SetPoseParams(std::vector<double>& poseParams) {
  poseParams[0] = modelParams.R_x;
  poseParams[1] = modelParams.R_y;
  poseParams[2] = modelParams.R_z;
  poseParams[3] = modelParams.T_x;
  poseParams[4] = modelParams.T_y;
  poseParams[5] = modelParams.T_z;
  poseParams[6] = cameraParams.orthoScale;
}

void FrameParameters::GetPoseParams(std::vector<double>& poseParams) {
  modelParams.R_x = poseParams[0];
  modelParams.R_y = poseParams[1];
  modelParams.R_z = poseParams[2];
  modelParams.T_x = poseParams[3];
  modelParams.T_y = poseParams[4];
  modelParams.T_z = poseParams[5];
  cameraParams.orthoScale = abs(poseParams[6]);
}

void FrameParameters::SetIdentityParams(std::vector<double>& idParams) {
  int nId = NUM_IDENTITY_3DMM;
  idParams.resize(nId);
  for (int kk = 0; kk < nId; kk++)
    idParams[kk] = (double)modelParams.identity_coef_(kk);
  return;
}

void FrameParameters::GetIdentityParams(std::vector<double>& idParams) {
  int nId = NUM_IDENTITY_3DMM;
  for (int kk = 0; kk < nId; kk++)
    modelParams.identity_coef_(kk) = (float)idParams[kk];
  return;
}

void FrameParameters::SetExpressionParams(std::vector<double>& exprParams) {
  int nExpr = NUM_EXPRESSION_3DMM;
  exprParams.resize(nExpr);
  for (int kk = 0; kk < nExpr; kk++)
    exprParams[kk] = (double)modelParams.expr_coef_(kk);
  return;
}

void FrameParameters::GetExpressionParams(std::vector<double>& exprParams) {
  int nExpr = NUM_EXPRESSION_3DMM;
  for (int kk = 0; kk < nExpr; kk++)
    modelParams.expr_coef_(kk) = (float)exprParams[kk];
  return;
}

void FrameParameters::SetAlbedoParams(std::vector<double>& albParams) {
  int nAlb = NUM_ALBEDO_3DMM;
  albParams.resize(nAlb);
  for (int kk = 0; kk < nAlb; kk++)
    albParams[kk] = (double)modelParams.color_coef_(kk);
  return;
}

void FrameParameters::SetSHparams(std::vector<double>& shParams) {
  shParams.resize(9);
  for (int i = 0; i < 9; i++)
    shParams[i] = (double)modelParams.shParams_whitebalance[i];
}

void FrameParameters::GetSHparams(std::vector<double>& shParams) {
  for (int i = 0; i < 9; i++)
    modelParams.shParams_whitebalance[i] = shParams[i];
}

void FrameParameters::GetAlbedoParams(std::vector<double>& albParams) {
  int nAlb = NUM_ALBEDO_3DMM;
  for (int kk = 0; kk < nAlb; kk++)
    modelParams.color_coef_(kk) = (float)albParams[kk];
  return;
}

glm::mat4& FrameParameters::UpdateModelMat() {
  double R_x = modelParams.R_x;
  double R_y = modelParams.R_y;
  double R_z = modelParams.R_z;
  double T_x = modelParams.T_x;
  double T_y = modelParams.T_y;
  double T_z = modelParams.T_z;
  double orthoScale = cameraParams.orthoScale;

  rotateMat =
      glm::eulerAngleZ(R_z) * glm::eulerAngleY(R_y) * glm::eulerAngleX(R_x);
  scaleMat =
      glm::scale(glm::mat4(1.0), glm::vec3(orthoScale, orthoScale, orthoScale));
  translateMat = glm::translate(glm::fmat4(1.0), glm::fvec3(T_x, T_y, T_z));
  model_mat_ = translateMat * scaleMat * rotateMat;
  return model_mat_;
}

glm::mat4& FrameParameters::UpdateProjectionMat() {
  float aspect_t = (float)cameraParams.aspect;
  float far_c_t = (float)cameraParams.zFar;
  projection_ = glm::perspective(45.0f, aspect_t, 0.1f, far_c_t);
  // projection_ = glm::perspective(, aspect_t, 0.f, far_c_t);
  // projection_ = glm::ortho(-aspect_t, aspect_t, -1.0f, 1.0f, -100.0f,
  // far_c_t);
  return projection_;
}

void FrameParameters::setOrdinaryShParams(std::vector<double>& shParams) {
  modelParams.setSHParams(shParams.data());
}



inline void set_RT(mat34& mat, std::vector<float>& rotation,
                   std::vector<float>& translation) {
  auto& rot = mat.rot;
  auto& trans = mat.trans;
  Eigen::Matrix3f rotation_;
  for (auto i = 0; i < 3; i++) {
    for (auto j = 0; j < 3; j++) {
      rotation_(i,j)= rotation[i * 3 + j];
    }
  }
  rot = mat33(rotation_);
  trans.x = translation[0];
  trans.y = translation[1];
  trans.z = translation[2];
}

void FrameParameters::push_present_RT() {
  prev_RT_2 = prev_RT_1;
  prev_RT_1 = present_RT;
}

void FrameParameters::smooth_RT() {

  get_present_RT();  
  mat34 prev_RT_1_mat34 = prev_RT_1, prev_RT_2_mat34 = prev_RT_2;
  // compute average rotation and translation
  Eigen::Matrix3f average_rotation = present_RT_mat34.rot;
  float3 average_translation = present_RT_mat34.trans;
  float cnt = 1;
  float delta;
  delta = lambda;
  auto delta_1 = prev_RT_1.conjugate() * present_RT;
  auto delta_2 = prev_RT_2.conjugate() * present_RT;
  if (abs(delta_1.q0.q0.w) >= cos(M_PI / 24) && delta_1.q1.norm() < 4) {
    average_rotation += delta * Eigen::Matrix3f(prev_RT_1_mat34.rot);
    average_translation += delta * prev_RT_1_mat34.trans;
    cnt += delta;
  }
  delta *= lambda;
  if (abs(delta_2.q0.q0.w) >= cos(M_PI / 24) && delta_2.q1.norm() < 4) {
    average_rotation += delta * Eigen::Matrix3f(prev_RT_2_mat34.rot);
    average_translation += delta * prev_RT_2_mat34.trans;
    cnt += delta;
  }
  Eigen::JacobiSVD<Eigen::Matrix3f> svd_solver(
      average_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
  average_rotation = svd_solver.matrixU() * svd_solver.matrixV().transpose();
  average_translation /= cnt;
  present_RT_mat34.rot = mat33(average_rotation);
  present_RT_mat34.trans = average_translation;
  present_RT = DualQuaternion(present_RT_mat34);

  for (auto i = 0; i < 3; i++) {
    for (auto j = 0; j < 3; j++) {
      host_rotation[i * 3 + j] = average_rotation(i, j);
    }
    host_translation[i] = (&average_translation.x)[i];
  }
  cuda_rotation_.upload(host_rotation);
  cuda_translation_.upload(host_translation);
}

void FrameParameters::get_present_RT() {
  cuda_rotation_.download(host_rotation);
  cuda_translation_.download(host_translation);
  set_RT(present_RT_mat34, host_rotation, host_translation);
  present_RT = DualQuaternion(present_RT_mat34);
  auto delta_1 = prev_RT_1.conjugate() * present_RT;
  auto delta_2 = prev_RT_2.conjugate() * present_RT;
  if (abs(delta_1.q0.q0.w) >= cos(M_PI / 24) && delta_1.q1.norm() < 4) {
    is_consis_1 = true;
  } else {
    is_consis_1 = false;
  }
  if (abs(delta_2.q0.q0.w) >= cos(M_PI / 24) && delta_2.q1.norm() < 4) {
    is_consis_2 = true;
  } else {
    is_consis_2 = false;
  }
}

void FrameBlock::saveRT(std::string file_name, float orthoScale) {
  std::clog << "Saving Rotation and Translation in " + file_name << std::endl;
  std::ofstream obj_file(file_name);
  obj_file << R_x << " " << R_y << " " << R_z << " " << T_x << " " << T_y << " "
           << T_z << " " << orthoScale << std::endl;
  obj_file.close();
}

void JointVideoParams::setFrameParams(std::vector<FrameBlock>& params,
                                      std::vector<double> errColor,
                                      std::vector<double> errShape) {
  frameParams = params;
  errPixel = errColor;
  errLandmark = errShape;
}

void JointVideoParams::setCamParams(std::vector<CameraBlock>& params,
                                    std::vector<double> errShape) {
  camParams = params;
  errLandmark = errShape;
}

void JointVideoParams::update() {
  // Backup
  backup();

  // Update
  double sumErrColor = 0;
  double sumErrLandmark = 0;

  for (int kk = 0; kk < errPixel.size(); kk++)
    errPixel[kk] = 1.0 / (errPixel[kk] + 20.0);
  for (int kk = 0; kk < errPixel.size(); kk++) sumErrColor += errPixel[kk];
  for (int kk = 0; kk < errPixel.size(); kk++) {
    errPixel[kk] /= sumErrColor;

    // errPixel[kk] = 1.0 / errPixel.size();
  }

  for (int kk = 0; kk < errLandmark.size(); kk++)
    errLandmark[kk] = 1.0 / (errLandmark[kk] + 10.0);
  for (int kk = 0; kk < errLandmark.size(); kk++)
    sumErrLandmark += errLandmark[kk];
  for (int kk = 0; kk < errLandmark.size(); kk++) {
    errLandmark[kk] /= sumErrLandmark;

    // errLandmark[kk] = 1.0 / errLandmark.size();
  }

  idParams = frameParams[0].identity_coef_;
  idParams.setZero();
  albedoParams = frameParams[0].color_coef_;
  albedoParams.setZero();
  // Identity & albedo
  for (int kk = 0; kk < frameParams.size(); kk++) {
    idParams.array() +=
        frameParams[kk].identity_coef_.array() * errLandmark[kk];
    albedoParams.array() += frameParams[kk].color_coef_.array() * errPixel[kk];
  }
  // SH lighting
  for (int kk = 0; kk < 27; kk++) shParams[kk] = 0.0;

  for (int kk = 0; kk < frameParams.size(); kk++) {
    for (int kk_i = 0; kk_i < 27; kk_i++) {
      shParams[kk_i] += frameParams[kk].shParams[kk_i] * errPixel[kk];
    }
  }
}

bool JointVideoParams::checkIdConverged(double thres) {
  double err = (idParams - idParamsLast).norm() / idParamsLast.norm();

  if (err < thres)
    return true;
  else
    return false;
}

bool JointVideoParams::checkAlbedoConverged(double thres) {
  double err =
      (albedoParams - albedoParamsLast).norm() / albedoParamsLast.norm();

  if (err < thres)
    return true;
  else
    return false;
}

bool JointVideoParams::checkSHConverged(double thres) {
  double err = 0.0;
  double normSH = 0.0;
  for (int kk = 0; kk < 27; kk++) {
    err +=
        (shParams[kk] - shParamsLast[kk]) * (shParams[kk] - shParamsLast[kk]);
    normSH += shParamsLast[kk] * shParamsLast[kk];
  }
  err = sqrt(err);
  normSH = sqrt(normSH);
  err = err / normSH;
  if (err < thres)
    return true;
  else
    return false;
}

void JointVideoParams::backup() {
  idParamsLast = idParams;
  albedoParamsLast = albedoParams;
  for (int kk = 0; kk < 27; kk++) shParamsLast[kk] = shParams[kk];
}

/// Parameter util functions
void loadParams(std::vector<FrameBlock>& fP, std::vector<CameraBlock>& cP,
                CameraBlock def_c, FrameBlock def_f, int n,
                std::string filename) {
  std::ifstream paramsIn;
  paramsIn.open(filename);
  CameraBlock camBlock;
  FrameBlock frameBlock;
  camBlock = def_c;
  frameBlock = def_f;
  cP.clear();
  fP.clear();
  if (paramsIn.is_open()) {
    for (int kk = 0; kk < n; kk++) {
      paramsIn >> camBlock.orthoScale;
      paramsIn >> frameBlock.R_x >> frameBlock.R_y >> frameBlock.R_z;
      paramsIn >> frameBlock.T_x >> frameBlock.T_y;
      for (int ii = 0; ii < 27; ii++) paramsIn >> frameBlock.shParams[ii];
      for (int ii = 0; ii < NUM_IDENTITY_3DMM; ii++)
        paramsIn >> frameBlock.identity_coef_(ii);
      for (int ii = 0; ii < NUM_ALBEDO_3DMM; ii++)
        paramsIn >> frameBlock.color_coef_(ii);
      for (int ii = 0; ii < NUM_EXPRESSION_3DMM; ii++)
        paramsIn >> frameBlock.expr_coef_(ii);
      cP.push_back(camBlock);
      fP.push_back(frameBlock);
    }
    paramsIn.close();
  }
}

void saveParams(std::vector<FrameBlock>& fP, std::vector<CameraBlock>& cP,
                int n, std::string filename) {
  std::ofstream paramsOut;
  paramsOut.open(filename);
  if (paramsOut.is_open()) {
    for (int kk = 0; kk < n; kk++) {
      paramsOut << cP[kk].orthoScale << " ";
      paramsOut << fP[kk].R_x << " " << fP[kk].R_y << " " << fP[kk].R_z << " "
                << fP[kk].T_x << " " << fP[kk].T_y << " ";

      for (int ii = 0; ii < 27; ii++) paramsOut << fP[kk].shParams[ii] << " ";
      for (int ii = 0; ii < NUM_IDENTITY_3DMM; ii++)
        paramsOut << fP[kk].identity_coef_(ii) << " ";
      for (int ii = 0; ii < NUM_ALBEDO_3DMM; ii++)
        paramsOut << fP[kk].color_coef_(ii) << " ";
      for (int ii = 0; ii < NUM_EXPRESSION_3DMM; ii++)
        paramsOut << fP[kk].expr_coef_(ii) << " ";

      paramsOut << std::endl;
    }
    paramsOut.close();
  }
}

FrameBlock getSmoothFrameParams(std::vector<FrameBlock>& videoFrameParams,
                                int numFrame, int frameIdx) {
  FrameBlock smoothParams;
  if (frameIdx == 0 || frameIdx == numFrame - 1)
    return videoFrameParams[frameIdx];
  else {
    // smooth pose and expression

    smoothParams = videoFrameParams[frameIdx];
    smoothParams.R_x = 0.25 * videoFrameParams[frameIdx - 1].R_x +
                       0.25 * videoFrameParams[frameIdx + 1].R_x +
                       0.5 * smoothParams.R_x;
    smoothParams.R_y = 0.25 * videoFrameParams[frameIdx - 1].R_y +
                       0.25 * videoFrameParams[frameIdx + 1].R_y +
                       0.5 * smoothParams.R_y;
    smoothParams.R_z = 0.25 * videoFrameParams[frameIdx - 1].R_z +
                       0.25 * videoFrameParams[frameIdx + 1].R_z +
                       0.5 * smoothParams.R_z;
    smoothParams.expr_coef_.array() =
        0.25 * videoFrameParams[frameIdx - 1].expr_coef_.array() +
        0.25 * videoFrameParams[frameIdx + 1].expr_coef_.array() +
        smoothParams.expr_coef_.array() * 0.5;

    // smoothParams = videoFrameParams[frameIdx];
    // smoothParams.R_x = (videoFrameParams[frameIdx - 1].R_x +
    // videoFrameParams[frameIdx + 1].R_x + smoothParams.R_x)/3.0;
    // smoothParams.R_y = (videoFrameParams[frameIdx - 1].R_y +
    // videoFrameParams[frameIdx + 1].R_y + smoothParams.R_y)/3.0;
    // smoothParams.R_z = (videoFrameParams[frameIdx - 1].R_z +
    // videoFrameParams[frameIdx + 1].R_z + smoothParams.R_z)/3.0;
    // smoothParams.expr_coef_.array() = (videoFrameParams[frameIdx -
    // 1].expr_coef_.array() +videoFrameParams[frameIdx + 1].expr_coef_.array() +
    //	smoothParams.expr_coef_.array()) / 3.0;

    return smoothParams;
    // return videoFrameParams[frameIdx];
  }
}

#ifdef USE_CUDA

void FrameBlock::cudaSetIDParams() {
  // std::cout << identity_coef_.size() << " " << cuda_id_coefficients_.size()
  // << std::endl; cuda_id_coefficients_.upload(identity_coef_.data(),
  // identity_coef_.size());
  cudaMemcpy(reinterpret_cast<void*>(cuda_id_coefficients_.ptr()),
             reinterpret_cast<const void*>(identity_coef_.data()),
             identity_coef_.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void FrameBlock::cudaSetEXPParams() {
  // std::cout << expr_coef_.size() << " " << cuda_exp_coefficients_.size() <<
  // std::endl;
  cudaMemcpy(reinterpret_cast<void*>(cuda_exp_coefficients_.ptr()),
             reinterpret_cast<const void*>(expr_coef_.data()),
             expr_coef_.size() * sizeof(float), cudaMemcpyHostToDevice);
}

#endif  // USE_CUDA
