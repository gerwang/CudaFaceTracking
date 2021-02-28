#include <iomanip>

#include "Common.h"
#ifdef USE_CUDA
#include "HostUtil.hpp"
#include "Timer.h"
#include "solution.h"

void cudaUpdateAbfromLandmarkTargetPoint2Line(
    pcl::gpu::DeviceArray<float> A, pcl::gpu::DeviceArray<float> b,
    pcl::gpu::DeviceArray<float3> unRT_target_position,
    const pcl::gpu::DeviceArray<float> base,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position_inv_sRT,
    const pcl::gpu::DeviceArray<float3> position,
    const pcl::gpu::DeviceArray<int2> landmarkIndex,
    const pcl::gpu::DeviceArray<float> coeff,
    const pcl::gpu::DeviceArray<float> reg,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> translation);

void cudaAddDelta(pcl::gpu::DeviceArray<float> v,
                  const pcl::gpu::DeviceArray<float> delta);

void cudaAddDeltaPartial(pcl::gpu::DeviceArray<float> v,
                         const pcl::gpu::DeviceArray<float> delta,
                         const pcl::gpu::DeviceArray<int> activate_basis_index);

void cudaUpdateScale(pcl::gpu::DeviceArray<float> T,
                     const pcl::gpu::DeviceArray<float> lambda,
                     const pcl::gpu::DeviceArray<float3> target_position,
                     const pcl::gpu::DeviceArray<float3> position);

void cudaUpdateMforRWithoutICP(
    pcl::gpu::DeviceArray<float> M, const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position,
    const pcl::gpu::DeviceArray<float3> position);

void cudaPushLandmarkTargetVertices(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position,
    const float lambda, const bool is_use_P2L = false);

void cudaPushNoseLandmarkTargetVertices(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position,
    const float lambda);

void cudaPushExpLandmarkTargetVertices(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position,
    const float lambda, const float eye_lambda, const float eyebrow_lambda);

void cudaPushExpLandmarkTargetVerticesFromUV(
    pcl::gpu::DeviceArray<float3> target_position,
    pcl::gpu::DeviceArray<float> lambda_position,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float2> input_target_uv,
    const msfr::intrinsics camera, const float lambda, const float eye_lambda,
    const float eyebrow_lambda);

void cudaPushSymVertices(pcl::gpu::DeviceArray<float3> target_position,
                         pcl::gpu::DeviceArray<float> lambda_position,
                         const pcl::gpu::DeviceArray<int> symlist,
                         const float lambda);

void cudaPushPresentVertices(pcl::gpu::DeviceArray<float3> target_position,
                             pcl::gpu::DeviceArray<float> lambda_position,
                             const pcl::gpu::DeviceArray<float3> position,
                             const float lambda);

void cudaUpdateTranslation(pcl::gpu::DeviceArray<float> translation,
                           const pcl::gpu::DeviceArray<float> rotation,
                           const pcl::gpu::DeviceArray<float> lambda,
                           const pcl::gpu::DeviceArray<float3> target_position,
                           const pcl::gpu::DeviceArray<float3> position);

void cudaUpdateErr(pcl::gpu::DeviceArray<float> err,
                   const pcl::gpu::DeviceArray<float> rotation,
                   const pcl::gpu::DeviceArray<float> translation,
                   const pcl::gpu::DeviceArray<float> lambda,
                   const pcl::gpu::DeviceArray<float3> target_position,
                   const pcl::gpu::DeviceArray<float3> position);

void cudaUpdateLandmarkErr(
    float& err, const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float3> position,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> input_target_position);

void cudaUpdateICPErr(float& err,
                      const pcl::gpu::DeviceArray<float3> target_position,
                      const pcl::gpu::DeviceArray<float3> position_sRT);

void cudaUpdateExpBase(
    pcl::gpu::DeviceArray<float> exp_base,
    pcl::gpu::DeviceArray<float> kf_A_values,
    pcl::gpu::DeviceArray<float> kf_b_values,
    const pcl::gpu::DeviceArray<float3> key_frames_position,
    const pcl::gpu::DeviceArray<float> key_frames_coefficients,
    const int key_frames_num, const int dim_exp);

void cudaTransposeA(pcl::gpu::DeviceArray<float> AT,
                    const pcl::gpu::DeviceArray<float> A, const int n,
                    const int ldn, const int m, const int row_offset);

void cudaUpdateAbfromTemporalSmoothness(
    pcl::gpu::DeviceArray<float> A, pcl::gpu::DeviceArray<float> b,
    const pcl::gpu::DeviceArray<float> present,
    const pcl::gpu::DeviceArray<float> prev_1,
    const pcl::gpu::DeviceArray<float> prev_2, const float smooth_lambda);

void cudaClamp(pcl::gpu::DeviceArray<float> x);

void cudaSampleExpBase(pcl::gpu::DeviceArray<float> sampled_base,
                       const pcl::gpu::DeviceArray<float> base,
                       const pcl::gpu::DeviceArray<int> sampled_key);

void cudaUpdateRTPoint2Line(pcl::gpu::DeviceArray<float> JTJ,
                            pcl::gpu::DeviceArray<float> Jr,
                            const pcl::gpu::DeviceArray<float> lambda,
                            const pcl::gpu::DeviceArray<float3> target_position,
                            const pcl::gpu::DeviceArray<float3> position_sRT,
                            const pcl::gpu::DeviceArray<float3> normal_R,
                            const pcl::gpu::PtrSz<int2> landmarkIndex);

void cudaUpdateRTPoint2Pixel(
    pcl::gpu::DeviceArray<float> JTJ, pcl::gpu::DeviceArray<float> Jr,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<int2> landmarkIndex,
    pcl::gpu::DeviceArray<float> translation);

void cudaUpdateSRTPoint2Pixel(
    pcl::gpu::DeviceArray<float> JTJ, pcl::gpu::DeviceArray<float> Jr,
    const pcl::gpu::DeviceArray<float> lambda,
    const pcl::gpu::DeviceArray<float3> target_position,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<float3> normal_R,
    const pcl::gpu::DeviceArray<int2> landmarkIndex,
    pcl::gpu::DeviceArray<float> translation);

void cudaUpdateContourLandmarkIndex(
    pcl::gpu::DeviceArray<int2> contour_landmark_index,
    const pcl::gpu::DeviceArray<float2> landmark_uv,
    const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<int> stripIndex,
    const pcl::gpu::DeviceArray<int2> stripBeginEnd,
    const msfr::intrinsics camera_intr);

void cudaSmoothExpBase(pcl::gpu::DeviceArray<float> smooth_base,
                       const pcl::gpu::DeviceArray<float> base,
                       const pcl::gpu::DeviceArray<float3> mean_shape,
                       const pcl::gpu::DeviceArray<int3> tri_list,
                       const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                       const pcl::gpu::DeviceArray<int1> fBegin,
                       const pcl::gpu::DeviceArray<int1> fEnd,
                       const pcl::gpu::DeviceArray<unsigned short> is_boundary,
                       const int dim_exp, const int n_verts,
                       const cudaStream_t stream);

void RtoEulerAngles(const Eigen::Matrix3f M, double& x, double& y, double& z) {
  float sy = sqrt(M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0));
  bool singular = sy < 1e-6;

  if (!singular) {
    x = atan2(M(2, 1), M(2, 2));
    y = atan2(-M(2, 0), sy);
    z = atan2(M(1, 0), M(0, 0));
  } else {
    x = atan2(-M(1, 2), M(1, 1));
    y = atan2(-M(2, 0), sy);
    z = 0;
  }
}

void Solution::updateSRTPositionNormal() {
  mesh_->updateSRTPositionNormal(parameters_->cuda_rotation_,
                                 parameters_->cuda_translation_);
}

void Solution::updateLandmarkErr(
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> landmark_position) {
  landmark_err_ = 0;
  cudaUpdateLandmarkErr(landmark_err_, parameters_->cuda_rotation_,
                        parameters_->cuda_translation_, mesh_->position(),
                        mesh_->cuda_is_front_now, landmark_id,
                        landmark_position);
}

void Solution::prepareTarget() {
  mesh_->updateInvSRTTargetPosition(parameters_->cuda_rotation_,
                                    parameters_->cuda_translation_);
}

void Solution::prepareTargetPosition() {
  mesh_->updateInvSRTTargetOnlyPosition(parameters_->cuda_rotation_,
                                        parameters_->cuda_translation_);
}

void Solution::prepareTargetLandmark() {
  mesh_->updateInvSRTTargetOnlyLandmark(parameters_->cuda_rotation_,
                                        parameters_->cuda_translation_);
}

void Solution::saveCameraView() {
  std::ofstream cameraview("./camera_view.txt");
  {
    auto position = parameters_->cameraParams.get_position();
    auto front = parameters_->cameraParams.get_front();
    auto up = parameters_->cameraParams.get_up();
    cameraview << position.x << std::endl;
    cameraview << position.y << std::endl;
    cameraview << position.z << std::endl;
    cameraview << front.x << std::endl;
    cameraview << front.y << std::endl;
    cameraview << front.z << std::endl;
    cameraview << up.x << std::endl;
    cameraview << up.y << std::endl;
    cameraview << up.z << std::endl;
  }
}

void Solution::loadCameraView() {
  std::ifstream cameraview("./camera_view.txt");
  {
    glm::vec3 position, front, up;
    cameraview >> position.x;
    cameraview >> position.y;
    cameraview >> position.z;
    cameraview >> front.x;
    cameraview >> front.y;
    cameraview >> front.z;
    cameraview >> up.x;
    cameraview >> up.y;
    cameraview >> up.z;
    parameters_->cameraParams.set_position(position);
    parameters_->cameraParams.set_front(front);
    parameters_->cameraParams.set_up(up);
  }
}

void Solution::save_exp() {
  cudaSafeCall(cudaStreamSynchronize(0));
  std::vector<float> host_exp;
  parameters_->modelParams.cuda_exp_coefficients_.download(host_exp);
  auto fp = fopen(
      ("./results/exp/" + std::to_string(image_idx_) + ".txt").c_str(), "w+");
  for (auto i : host_exp) {
    fprintf(fp, "%f\n", i);
  }
  fclose(fp);
}

void Solution::load_exp() {
  std::vector<float> host_exp(model_->dim_exp_);
  auto fp = fopen(
      ("./results/exp/" + std::to_string(image_idx_) + ".txt").c_str(), "r");
  for (auto& i : host_exp) {
    fscanf(fp, "%f", &i);
  }
  fclose(fp);
  parameters_->modelParams.cuda_exp_coefficients_.upload(host_exp);
}

void Solution::save_pose() {
  cudaSafeCall(cudaStreamSynchronize(0));
  std::vector<float> host_rotation, host_translation;
  parameters_->cuda_rotation_.download(host_rotation);
  parameters_->cuda_translation_.download(host_translation);
  std::ofstream os("./results/pose/" + std::to_string(image_idx_) + ".txt");
  for (auto i : host_rotation) {
    os << i << std::endl;
  }
  for (auto i : host_translation) {
    os << i << std::endl;
  }
}

void Solution::load_pose() {
  load_pose_from_file(
      ("./results/pose/" + std::to_string(image_idx_) + ".txt").c_str());
}

void Solution::load_pose_from_file(const std::string filename_) {
  LOG(INFO) << "Loading pose";
  std::string filename = filename_;
  if (filename.length() == 0) {
    filename = "./results/pose/" + std::to_string(image_idx_) + ".txt";
  }
  std::vector<float> host_rotation(9), host_translation(4);
  std::ifstream is(filename);
  for (auto& i : host_rotation) {
    is >> i;
  }
  for (auto& i : host_translation) {
    is >> i;
  }
  parameters_->cuda_rotation_.upload(host_rotation);
  parameters_->cuda_translation_.upload(host_translation);
}

void Solution::show_color_map() {
  std::cout << "show color idx: " << image_idx_ << std::endl;
  updateSRTPositionNormal();
  gmodels_->updateMesh(model_name(), mesh_->cuda_position_sRT,
                       mesh_->cuda_color_, mesh_->cuda_normal_R,
                       mesh_->cuda_tri_list_);
  tex_->renderModel(*gmodels_, canvas_name(), model_name(), rinfo_, renderer);
  static IplImage* c_img = cvCreateImage(
      cvSize(reader_->get_color_width(), reader_->get_color_height()),
      IPL_DEPTH_32F, 4);
  // static IplImage *c_img_uchar =
  // cvCreateImage(cvSize(reader_->get_color_width(),
  // reader_->get_color_height()), IPL_DEPTH_8U, 3);

  const auto color_map = get_color_image();
  color_map.download(c_img->imageData, sizeof(float4) * c_img->width);
  // const auto clip = [](float x) {
  //	return x;
  //};
  // for (auto i = 0; i < reader_->get_color_width() *
  // reader_->get_color_height(); ++i)
  //{
  //	reinterpret_cast<unsigned char*>(c_img_uchar->imageData)[3 * i + 2] =
  // static_cast<unsigned
  // char>(clip(reinterpret_cast<float*>(c_img->imageData)[4
  //* i] * 255.0f)); 	reinterpret_cast<unsigned
  // char*>(c_img_uchar->imageData)[3
  //* i + 1] = static_cast<unsigned
  // char>(clip(reinterpret_cast<float*>(c_img->imageData)[4 * i + 1] *
  // 255.0f)); 	reinterpret_cast<unsigned char*>(c_img_uchar->imageData)[3 * i +
  // 0] = static_cast<unsigned
  // char>(clip(reinterpret_cast<float*>(c_img->imageData)[4
  //* i + 2] * 255.0f));
  //}
  cvShowImage("color", c_img);
  cvWaitKey();
}

void Solution::save_color_map(const std::string& filename,
                              const std::string& canvas_, const int color_mode,
                              const bool is_save_now) {
  // initialize information
  std::string file_name = filename;
  if (filename.length() == 0) {
    std::stringstream str;
    str << "./results/color/" << std::setfill('0') << std::setw(4)
        << std::to_string(image_idx_) << ".png";
    std::getline(str, file_name);
  }
  auto folderPath = file_name.substr(0, file_name.rfind('/'));
  CreateDirectory(folderPath.c_str(), nullptr);
  auto cvt_color_mode = CV_RGBA2BGRA;
  auto dst_type_channel = CV_8UC4;
  if (color_mode == 1) {
    cvt_color_mode = CV_RGBA2BGR;
    dst_type_channel = CV_8UC3;
  } else if (color_mode != 0) {
    LOG(ERROR) << "Wrong Color Mode For Saving A Image: " << file_name
               << " invalid color mode:" << color_mode;
  }
  // download image
  auto image = gmodels_->GameModelList[canvas_];
  auto imageHeight = image.height;
  auto imageWidth = image.width;
  auto color_map = get_color_image(canvas_);
  cv::Mat img(imageHeight, imageWidth, CV_32FC4);
  color_map.download(img.data, imageWidth * sizeof(float4));
  // convert image
  cv::Mat bgrImg;
  cv::Mat ucharImg;
  cv::cvtColor(img, bgrImg, cvt_color_mode);
  bgrImg *= 255;
  bgrImg.convertTo(ucharImg, dst_type_channel);
  if (is_save_now) {
    DLOG(INFO) << "Save Image: " << file_name;
    cv::imwrite(file_name, ucharImg);
  } else {
    save_image_list.push_back({file_name, ucharImg});
  }
  // TODO
}

void Solution::pushNoseLandmarkTargetVertices(const float lambda) {
  cudaPushNoseLandmarkTargetVertices(
      mesh_->target_landmark_, mesh_->lambda_landmark_,
      detector_->cuda_landmark_index_, detector_->cuda_landmark_position_,
      lambda);
  mesh_->is_lambda_clear_ = false;
}

void Solution::pushExpLandmarkTargetVertices(const float lambda,
                                             const float eye_lambda,
                                             const float eyebrow_lambda) {
  cudaPushExpLandmarkTargetVerticesFromUV(
      mesh_->target_landmark_, mesh_->lambda_landmark_, detector_->cuda_landmark_index_,
      detector_->cuda_pre_landmark_uv_, parameters_->cameraParams.camera_intr,
      lambda, eye_lambda, eyebrow_lambda);
  mesh_->is_lambda_clear_ = false;
}

void Solution::pushLandmarkTargetVertices(const float lambda,
                                          const bool is_use_P2L) {
  {
    cudaPushLandmarkTargetVertices(
        mesh_->target_landmark_, mesh_->lambda_landmark_,
        detector_->cuda_landmark_index_, detector_->cuda_landmark_position_,
        lambda, is_use_P2L);
  }
  mesh_->is_lambda_clear_ = false;
}

void Solution::pushLandmarkTargetVertices_fuse(const float lambda) {
  cudaPushLandmarkTargetVertices(mesh_->target_position_,
                                 mesh_->lambda_position_,
                                 detector_->cuda_landmark_index_,
                                 detector_->cuda_landmark_position_, lambda);
  mesh_->is_lambda_clear_ = false;
}

void Solution::updateContourLandmarkIndex() {
  cudaUpdateContourLandmarkIndex(
      detector_->cuda_contour_landmark_index_,
      detector_->cuda_contour_landmark_uv_, mesh_->cuda_position_sRT,
      model_->cuda_strip_index_, model_->cuda_strip_begin_end_,
      parameters_->cameraParams.camera_intr);
}

void Solution::pushContourLandmarkTargetVertices(const float lambda,
                                                 const bool is_use_P2L) {
  updateContourLandmarkIndex();
  cudaPushLandmarkTargetVertices(
      mesh_->target_landmark_, mesh_->lambda_landmark_,
      detector_->cuda_contour_landmark_index_,
      detector_->cuda_contour_landmark_position_, lambda, is_use_P2L);
}

void Solution::pushLandmarkTargetVertices(
    const pcl::gpu::DeviceArray<int2> landmark_id,
    const pcl::gpu::DeviceArray<float3> landmark_position, const float lambda) {
  cudaPushLandmarkTargetVertices(mesh_->target_position_,
                                 mesh_->lambda_position_, landmark_id,
                                 landmark_position, lambda);
  mesh_->is_lambda_clear_ = false;
}

#if DELAY_ONE_FRAME
extern pcl::gpu::DeviceArray<float3> prevPosition1, prevNormal1;
bool firstSaveFrame = true;
#endif

void Solution::saveCoefficients(std::string file_name) {
  std::vector<float> host_exp;
  cudaSafeCall(cudaStreamSynchronize(0));
  parameters_->modelParams.cuda_exp_coefficients_.download(host_exp);
  bool isLegal = false;
  for (auto& iter : host_exp) {
    // std::cout << iter << std::endl;
    if (iter > 0.3f) {
      isLegal = true;
      break;
    }
  }
  if (isLegal) {
    FILE* fp = fopen(file_name.c_str(), "w+");
    fprintf(fp, "%d\n", model_->dim_exp_);
    for (int i = 0; i < model_->dim_exp_; ++i) {
      fprintf(fp, "%f\n", host_exp[i]);
    }
    fclose(fp);
  }
}

void Solution::sampleExpBase() {
  cudaSampleExpBase(sampled_exp_base_, model_->personalized_exp_base_,
                    sampled_key_);
  model_->updateExpBaseCorrelationReg(sampled_exp_base_ATA_,
                                      reg_sampled_exp_base_, sampled_exp_base_);
}

void Solution::smoothMesh() {
  for (int i = 0; i < 3; ++i) {
    cudaSmoothMesh(mesh_->cuda_position_, mesh_->cuda_tri_list_,
                   mesh_->cudafvLookUpTable, mesh_->cudafBegin, mesh_->cudafEnd,
                   mesh_->cuda_is_boundary_);
  }
}

void Solution::updatePersonalizedExpBasefromDeformationTransfer() {
  model_->UpdatePersonalizedExpBase(*mesh_);
  sampleExpBase();
}

void Solution::solveIDCoefficient(const float lambda) {
  // pushWeightedPositionInvSRT();
  if (mesh_->is_lambda_clear_) {
    std::cerr << __FILE__ << ": " << __FUNCTION__ << ": " << __LINE__ << ": "
              << "No terms for solve identity coefficients!" << std::endl;
    std::exit(1);
  }

  pcl::gpu::DeviceArray<float> A(A_.ptr(), model_->dim_id_ * model_->dim_id_);
  pcl::gpu::DeviceArray<float> b(model_->dim_id_);
  pcl::gpu::DeviceArray<float> delta_id_coeff(model_->dim_id_);

  clearCudaMem(A);
  clearCudaMem(b);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->id_base_, mesh_->lambda_landmark_,
      mesh_->target_landmark_inv_sRT, mesh_->cuda_position_,
      detector_->cuda_landmark_index_,
      parameters_->modelParams.cuda_id_coefficients_, model_->reg_id_,
      parameters_->cuda_rotation_, parameters_->cuda_translation_);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->id_base_, mesh_->lambda_landmark_,
      mesh_->target_landmark_inv_sRT, mesh_->cuda_position_,
      detector_->cuda_contour_landmark_index_,
      parameters_->modelParams.cuda_id_coefficients_, model_->reg_id_,
      parameters_->cuda_rotation_, parameters_->cuda_translation_);
  clearCudaMem(delta_id_coeff);
  cudaCGSolver(delta_id_coeff, A, b);
  cudaAddDelta(parameters_->modelParams.cuda_id_coefficients_, delta_id_coeff);

  updateMesh();
}

void Solution::solveIDCoefficientPoint2Line(const float lambda) {
  // pushWeightedPositionInvSRT();
  if (mesh_->is_lambda_clear_) {
    std::cerr << __FILE__ << ": " << __FUNCTION__ << ": " << __LINE__ << ": "
              << "No terms for solve identity coefficients!" << std::endl;
    std::exit(1);
  }

  pcl::gpu::DeviceArray<float> A(A_.ptr(), model_->dim_id_ * model_->dim_id_);
  pcl::gpu::DeviceArray<float> b(model_->dim_id_);
  pcl::gpu::DeviceArray<float> delta_id_coeff(model_->dim_id_);
  clearCudaMem(A);
  clearCudaMem(b);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->personalized_exp_base_,
      mesh_->lambda_landmark_, mesh_->target_landmark_inv_sRT,
      mesh_->cuda_position_, detector_->cuda_landmark_index_,
      parameters_->modelParams.cuda_exp_coefficients_,
      model_->reg_personalized_exp_, parameters_->cuda_rotation_,
      parameters_->cuda_translation_);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->personalized_exp_base_,
      mesh_->lambda_landmark_, mesh_->target_landmark_inv_sRT,
      mesh_->cuda_position_, detector_->cuda_contour_landmark_index_,
      parameters_->modelParams.cuda_exp_coefficients_,
      model_->reg_personalized_exp_, parameters_->cuda_rotation_,
      parameters_->cuda_translation_);
  clearCudaMem(delta_id_coeff);
  cudaCGSolver(delta_id_coeff, A, b);
  cudaAddDelta(parameters_->modelParams.cuda_id_coefficients_, delta_id_coeff);

  updateMesh();
}

void Solution::solveSparseCoefficient(const float reg_lambda,
                                      const float temporal_smooth_lambda,
                                      const float L1_reg_lambda) {
  // pushWeightedPositionInvSRT();
  if (mesh_->is_lambda_clear_) {
    std::cerr << __FILE__ << ": " << __FUNCTION__ << ": " << __LINE__ << ": "
              << "No terms for solve identity coefficients!" << std::endl;
    std::exit(1);
  }

  pcl::gpu::DeviceArray<float> A(A_.ptr(), model_->dim_exp_ * model_->dim_exp_);
  pcl::gpu::DeviceArray<float> b(b_.ptr(), model_->dim_exp_);
  pcl::gpu::DeviceArray<float> delta_exp_coeff(delta_coefficients_.ptr(),
                                               model_->dim_exp_);
  clearCudaMem(A);
  clearCudaMem(b);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->personalized_exp_base_,
      mesh_->lambda_landmark_, mesh_->target_landmark_inv_sRT,
      mesh_->cuda_position_, detector_->cuda_landmark_index_,
      parameters_->modelParams.cuda_exp_coefficients_,
      model_->reg_personalized_exp_, parameters_->cuda_rotation_,
      parameters_->cuda_translation_);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->personalized_exp_base_,
      mesh_->lambda_landmark_, mesh_->target_landmark_inv_sRT,
      mesh_->cuda_position_, detector_->cuda_contour_landmark_index_,
      parameters_->modelParams.cuda_exp_coefficients_,
      model_->reg_personalized_exp_, parameters_->cuda_rotation_,
      parameters_->cuda_translation_);

  cudaUpdateAbfromTemporalSmoothness(
      A, b, parameters_->modelParams.cuda_exp_coefficients_,
      parameters_->modelParams.cuda_exp_prev_1_,
      parameters_->modelParams.cuda_exp_prev_2_, temporal_smooth_lambda);
  shootingSolver.solve(parameters_->modelParams.cuda_exp_coefficients_, b, A,
                       L1_reg_lambda);

  updateExpMesh();
}

void Solution::solveSampledSparseCoefficient(const float reg_lambda,
                                             const float temporal_smooth_lambda,
                                             const float l1_reg_lambda) {
  // pushWeightedPositionInvSRT();
  if (mesh_->is_lambda_clear_) {
    std::cerr << __FILE__ << ": " << __FUNCTION__ << ": " << __LINE__ << ": "
              << "No terms for solve identity coefficients!" << std::endl;
    std::exit(1);
  }

  pcl::gpu::DeviceArray<float> A(A_.ptr(), model_->dim_exp_ * model_->dim_exp_);
  pcl::gpu::DeviceArray<float> b(b_.ptr(), model_->dim_exp_);
  pcl::gpu::DeviceArray<float> delta_exp_coeff(delta_coefficients_.ptr(),
                                               model_->dim_exp_);
  pcl::gpu::DeviceArray<float3> sampled_unRTposition(unRTposition_.ptr(),
                                                     sampled_num_);
  clearCudaMem(A);
  clearCudaMem(b);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->personalized_exp_base_,
      mesh_->lambda_landmark_, mesh_->target_landmark_inv_sRT,
      mesh_->cuda_position_, detector_->cuda_landmark_index_,
      parameters_->modelParams.cuda_exp_coefficients_,
      model_->reg_personalized_exp_, parameters_->cuda_rotation_,
      parameters_->cuda_translation_);
  cudaUpdateAbfromLandmarkTargetPoint2Line(
      A, b, unRTposition_, model_->personalized_exp_base_,
      mesh_->lambda_landmark_, mesh_->target_landmark_inv_sRT,
      mesh_->cuda_position_, detector_->cuda_contour_landmark_index_,
      parameters_->modelParams.cuda_exp_coefficients_,
      model_->reg_personalized_exp_, parameters_->cuda_rotation_,
      parameters_->cuda_translation_);
  cudaUpdateAbfromTemporalSmoothness(
      A, b, parameters_->modelParams.cuda_exp_coefficients_,
      parameters_->modelParams.cuda_exp_prev_1_,
      parameters_->modelParams.cuda_exp_prev_2_, temporal_smooth_lambda);
  shootingSolver.solve(parameters_->modelParams.cuda_exp_coefficients_, b, A,
                       l1_reg_lambda);
  updateExpMesh();
}

void Solution::solveScale() {
  cudaUpdateScale(parameters_->cuda_translation_, mesh_->lambda_landmark_,
                  mesh_->target_landmark_, mesh_->position());
}

void get_translation_JTJ_JTr(Eigen::MatrixXf& JTJ, Eigen::VectorXf JTr,
                             float3 v_, float3 v, float lambda) {
  Eigen::MatrixXf J(3, 6);
  Eigen::VectorXf r(3);
  J.setZero();
  r.setZero();
  J(0, 1) = v.z;
  J(0, 2) = -v.y;
  J(0, 3) = 1.0f;
  J(1, 0) = -v.z;
  J(1, 2) = v.x;
  J(1, 4) = 1.0f;
  J(2, 0) = v.y;
  J(2, 1) = -v.x;
  J(2, 5) = 1.0f;
  r(0) = v_.x - v.x;
  r(1) = v_.y - v.y;
  r(2) = v_.z - v.z;
  JTJ += lambda * J.transpose() * J;
  JTr += lambda * J.transpose() * r;
}

void get_rotation_JTJ_JTr(Eigen::MatrixXf& JTJ, Eigen::VectorXf JTr,
                          mat33 rotation, mat33 present_rotation,
                          float lambda) {
  Eigen::MatrixXf J(3, 6);
  Eigen::VectorXf r(3);
  for (auto i = 0; i < 3; ++i) {
    float3 v = present_rotation.cols[i];
    float3 v_ = rotation.cols[i];
    J.setZero();
    r.setZero();
    J(0, 1) = v.z;
    J(0, 2) = -v.y;
    J(1, 0) = -v.z;
    J(1, 2) = v.x;
    J(2, 0) = v.y;
    J(2, 1) = -v.x;
    r(0) = v_.x - v.x;
    r(1) = v_.y - v.y;
    r(2) = v_.z - v.z;
    JTJ += lambda * J.transpose() * J;
    JTr += lambda * J.transpose() * r;
  }
}

void add_rotation(Eigen::MatrixXf& JTJ, Eigen::VectorXf Jr, mat34 present_RT,
                  mat34 prev_RT, float lambda) {}

void Solution::solveRTwithNormal(const bool P2L) {
  Eigen::MatrixXf JTJ(6, 6);
  Eigen::VectorXf Jr(6), delta_x(6);

  clearCudaMem(RT_solver_JTJ);
  clearCudaMem(RT_solver_Jr);

  cudaUpdateRTPoint2Line(RT_solver_JTJ, RT_solver_Jr, mesh_->lambda_landmark_,
                         mesh_->target_landmark_, mesh_->cuda_position_sRT,
                         mesh_->cuda_normal_R, detector_->cuda_landmark_index_);
  cudaUpdateRTPoint2Line(RT_solver_JTJ, RT_solver_Jr, mesh_->lambda_landmark_,
                         mesh_->target_landmark_, mesh_->cuda_position_sRT,
                         mesh_->cuda_normal_R,
                         detector_->cuda_contour_landmark_index_);

  cudaSafeCall(cudaStreamSynchronize(0));
  RT_solver_JTJ.download(JTJ.data());
  RT_solver_Jr.download(Jr.data());

  if (parameters_->is_consis_1) {
    // std::cout << "Smooth 1" << std::endl;
    mat34 prev_1_mat34 = parameters_->prev_RT_1;
    get_rotation_JTJ_JTr(JTJ, Jr, prev_1_mat34.rot,
                         parameters_->present_RT_mat34.rot,
                         parameters_->lambda_1);
    get_translation_JTJ_JTr(JTJ, Jr, prev_1_mat34.trans,
                            parameters_->present_RT_mat34.trans,
                            parameters_->lambda_2);
  }
  if (parameters_->is_consis_2) {
    // std::cout << "Smooth 2" << std::endl;
    mat34 prev_2_mat34 = parameters_->prev_RT_2;
    get_rotation_JTJ_JTr(JTJ, Jr, prev_2_mat34.rot,
                         parameters_->present_RT_mat34.rot,
                         parameters_->lambda_2);
    get_translation_JTJ_JTr(JTJ, Jr, prev_2_mat34.trans,
                            parameters_->present_RT_mat34.trans,
                            parameters_->lambda_2);
  }
  if (parameters_->is_consis_1 && parameters_->is_consis_2) {
    // std::cout << "Smooth 3" << std::endl;
    mat34 prev_1_mat34 = parameters_->prev_RT_1;
    mat34 prev_2_mat34 = parameters_->prev_RT_2;
    mat33 delta_rot = prev_1_mat34.rot;
    for (auto i = 0; i < 3; i++) {
      delta_rot.cols[i] *= 2;
    }
    delta_rot = delta_rot - prev_2_mat34.rot;
    get_rotation_JTJ_JTr(JTJ, Jr, delta_rot, parameters_->present_RT_mat34.rot,
                         parameters_->lambda_3);
    get_translation_JTJ_JTr(
        JTJ, Jr, 2 * prev_1_mat34.trans - prev_2_mat34.trans,
        parameters_->present_RT_mat34.trans, parameters_->lambda_3);
  }
  delta_x = JTJ.ldlt().solve(Jr);
  mat34 SE3;
  {
    float3 twist_rot;
    twist_rot.x = delta_x(0);
    twist_rot.y = delta_x(1);
    twist_rot.z = delta_x(2);

    float3 twist_trans;
    twist_trans.x = delta_x(3);
    twist_trans.y = delta_x(4);
    twist_trans.z = delta_x(5);

    if (fabs_sum(twist_rot) < 1e-20f) {
      SE3.rot = mat33::identity();
      SE3.trans = twist_trans;
    } else {  /// needs to be check
      float angle = norm(twist_rot);
      float3 axis = 1.0f / angle * twist_rot;

      float c = cosf(angle);
      float s = sinf(angle);
      float t = 1.0f - c;

      SE3.rot.m00() = t * axis.x * axis.x + c;
      SE3.rot.m01() = t * axis.x * axis.y - axis.z * s;
      SE3.rot.m02() = t * axis.x * axis.z + axis.y * s;

      SE3.rot.m10() = t * axis.x * axis.y + axis.z * s;
      SE3.rot.m11() = t * axis.y * axis.y + c;
      SE3.rot.m12() = t * axis.y * axis.z - axis.x * s;

      SE3.rot.m20() = t * axis.x * axis.z - axis.y * s;
      SE3.rot.m21() = t * axis.y * axis.z + axis.x * s;
      SE3.rot.m22() = t * axis.z * axis.z + c;

      SE3.trans = s / angle * twist_trans +
                  (1 - s / angle) * (outer_prod(axis, axis) * twist_trans) +
                  (1 - c) / angle * cross(axis, twist_trans);
    }
  }
  mat34 SE3_prev;
  std::vector<float> translation(4);
  parameters_->cuda_rotation_.download(&SE3_prev.rot.cols[0].x);
  parameters_->cuda_translation_.download(translation);
  SE3_prev.rot = SE3_prev.rot.transpose();
  SE3_prev.trans.x = translation[0];
  SE3_prev.trans.y = translation[1];
  SE3_prev.trans.z = translation[2];
  SE3_prev = SE3 * SE3_prev;
  Eigen::Matrix4f host_RT = SE3_prev;
  Eigen::Matrix3f rotation;
  rotation = host_RT.block(0, 0, 3, 3).transpose();
  parameters_->cuda_rotation_.upload(rotation.data(), 9);
  translation[0] = host_RT(0, 3);
  translation[1] = host_RT(1, 3);
  translation[2] = host_RT(2, 3);
  parameters_->cuda_translation_.upload(translation);
  parameters_->get_present_RT();
  mat34 present_RT_mat34 = parameters_->present_RT;
  rotation = present_RT_mat34.rot;
  RtoEulerAngles(rotation, parameters_->modelParams.R_x,
                 parameters_->modelParams.R_y, parameters_->modelParams.R_z);

  parameters_->modelParams.T_x = present_RT_mat34.trans.x;
  parameters_->modelParams.T_y = present_RT_mat34.trans.y;
  parameters_->modelParams.T_z = present_RT_mat34.trans.z;
  parameters_->cameraParams.orthoScale = translation[3];
  updateSRTPositionNormal();
}

void Solution::solveSRTwithNormal() {
  Eigen::MatrixXf JTJ(7, 7);
  Eigen::VectorXf Jr(7), delta_x(7);

  clearCudaMem(SRT_solver_JTJ);
  clearCudaMem(SRT_solver_Jr);

  cudaUpdateSRTPoint2Pixel(
      SRT_solver_JTJ, SRT_solver_Jr, mesh_->lambda_landmark_,
      mesh_->target_landmark_, mesh_->cuda_position_sRT, mesh_->cuda_normal_R,
      detector_->cuda_landmark_index_, parameters_->cuda_translation_);
  cudaUpdateSRTPoint2Pixel(
      SRT_solver_JTJ, SRT_solver_Jr, mesh_->lambda_landmark_,
      mesh_->target_landmark_, mesh_->cuda_position_sRT, mesh_->cuda_normal_R,
      detector_->cuda_contour_landmark_index_, parameters_->cuda_translation_);
  cudaSafeCall(cudaStreamSynchronize(0));
  SRT_solver_JTJ.download(JTJ.data());
  SRT_solver_Jr.download(Jr.data());
  delta_x = JTJ.ldlt().solve(Jr);
  mat34 SE3;
  {
    float3 twist_rot;
    twist_rot.x = delta_x(0);
    twist_rot.y = delta_x(1);
    twist_rot.z = delta_x(2);

    float3 twist_trans;
    twist_trans.x = delta_x(3);
    twist_trans.y = delta_x(4);
    twist_trans.z = delta_x(5);

    if (fabs_sum(twist_rot) < 1e-20f) {
      SE3.rot = mat33::identity();
      SE3.trans = twist_trans;
    } else {  /// needs to be check
      float angle = norm(twist_rot);
      float3 axis = 1.0f / angle * twist_rot;

      float c = cosf(angle);
      float s = sinf(angle);
      float t = 1.0f - c;

      SE3.rot.m00() = t * axis.x * axis.x + c;
      SE3.rot.m01() = t * axis.x * axis.y - axis.z * s;
      SE3.rot.m02() = t * axis.x * axis.z + axis.y * s;

      SE3.rot.m10() = t * axis.x * axis.y + axis.z * s;
      SE3.rot.m11() = t * axis.y * axis.y + c;
      SE3.rot.m12() = t * axis.y * axis.z - axis.x * s;

      SE3.rot.m20() = t * axis.x * axis.z - axis.y * s;
      SE3.rot.m21() = t * axis.y * axis.z + axis.x * s;
      SE3.rot.m22() = t * axis.z * axis.z + c;

      SE3.trans = s / angle * twist_trans +
                  (1 - s / angle) * (outer_prod(axis, axis) * twist_trans) +
                  (1 - c) / angle * cross(axis, twist_trans);
    }
  }
  mat34 SE3_prev;
  std::vector<float> translation(4);
  parameters_->cuda_rotation_.download(&SE3_prev.rot.cols[0].x);
  parameters_->cuda_translation_.download(translation);
  SE3_prev.rot = SE3_prev.rot.transpose();
  SE3_prev.trans.x = translation[0];
  SE3_prev.trans.y = translation[1];
  SE3_prev.trans.z = translation[2];
  SE3_prev = SE3 * SE3_prev;
  Eigen::Matrix4f host_RT = SE3_prev;
  Eigen::Matrix3f rotation;
  rotation = host_RT.block(0, 0, 3, 3).transpose();
  parameters_->cuda_rotation_.upload(rotation.data(), 9);
  translation[0] = host_RT(0, 3);
  translation[1] = host_RT(1, 3);
  translation[2] = host_RT(2, 3);
  // std::cout << "delta: " << delta_x(6) << std::endl;
  translation[3] += delta_x(6);
  // std::cout << "scale " << translation[3] << std::endl;

  parameters_->cuda_translation_.upload(translation);

  rotation.transposeInPlace();
  RtoEulerAngles(rotation, parameters_->modelParams.R_x,
                 parameters_->modelParams.R_y, parameters_->modelParams.R_z);

  parameters_->modelParams.T_x = translation[0];
  parameters_->modelParams.T_y = translation[1];
  parameters_->modelParams.T_z = translation[2];
  parameters_->cameraParams.orthoScale = translation[3];
  updateSRTPositionNormal();
}

// using SVD method to solve R and T
void Solution::solveScaleRT() {
  if (mesh_->is_lambda_clear_) {
    std::cerr << __FILE__ << ": " << __FUNCTION__ << ": " << __LINE__ << ": "
              << "No terms for solve ROTATION and TRANSMITION!" << std::endl;
    std::exit(1);
  }
  static Eigen::Matrix3f host_R;
  static std::vector<float> host_T(4);
  {
    cudaUpdateMforRWithoutICP(parameters_->cuda_rotation_,
                              mesh_->lambda_landmark_, mesh_->target_landmark_,
                              mesh_->position());
    cudaSafeCall(cudaStreamSynchronize(0));
    parameters_->cuda_rotation_.download(host_R.data());
    host_R.transposeInPlace();
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(
        host_R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    host_R = svd.matrixU() *
             svd.matrixV().transpose();  // R = V^TU here computing R^T

    parameters_->cuda_rotation_.upload(host_R.data(), 9);
    cudaUpdateTranslation(parameters_->cuda_translation_,
                          parameters_->cuda_rotation_, mesh_->lambda_landmark_,
                          mesh_->target_landmark_, mesh_->position());
  }
  parameters_->get_present_RT();
  mat34 present_RT_mat34 = parameters_->present_RT;
  host_R = present_RT_mat34.rot;
  RtoEulerAngles(host_R, parameters_->modelParams.R_x,
                 parameters_->modelParams.R_y, parameters_->modelParams.R_z);

  parameters_->cuda_translation_.download(host_T);
  parameters_->modelParams.T_x = present_RT_mat34.trans.x;
  parameters_->modelParams.T_y = present_RT_mat34.trans.y;
  parameters_->modelParams.T_z = present_RT_mat34.trans.z;
  parameters_->cameraParams.orthoScale = host_T[3];
  updateSRTPositionNormal();
  cudaStreamSynchronize(0);
  DLOG(INFO) << "Finish Update Scale RT" << std::endl;
}

void Solution::saveLandmarkErrors(std::string file_name) {
  detector_->SaveLandmarkError(file_name, parameters_->cameraParams.camera_intr,
                               mesh_->cuda_position_sRT);
}

void Solution::saveCorrespond(std::string file_name) {
  mesh_->updateInvSRTProjectionPosition(parameters_->cuda_rotation_,
                                        parameters_->cuda_translation_);
  cudaStreamSynchronize(0);
  mesh_->position_.transposeInPlace();
  mesh_->projection_position_.download((float3*)mesh_->position_.data());
  mesh_->position_.transposeInPlace();
  mesh_->write_obj(file_name + "_corr.obj");
}

void Solution::saveModelFace(std::string file_name) {
  cudaStreamSynchronize(0);
  mesh_->position_.transposeInPlace();
  mesh_->cuda_position_.download((float3*)mesh_->position_.data());
  mesh_->position_.transposeInPlace();
  mesh_->write_obj(file_name);
}

void Solution::saveSRTModelFace(std::string file_name) {
  cudaSafeCall(cudaStreamSynchronize(0));
  std::vector<float3> host_position;
  std::vector<float> host_weight;
  mesh_->cuda_position_sRT.download(host_position);

  for (int i = 0; i < host_position.size(); ++i) {
    mesh_->position_(i, 0) = host_position[i].x;
    mesh_->position_(i, 1) = host_position[i].y;
    mesh_->position_(i, 2) = host_position[i].z;
  }
  mesh_->write_obj(file_name + "_sRT.obj");
}

void Solution::showLandmarks() {
  auto host_color_map = reader_->get_color_frame(image_idx_);
  detector_->showLandmarkCorrespondence(
      host_color_map, mesh_->cuda_position_sRT, mesh_->cuda_is_front_now,
      parameters_->cameraParams.camera_intr);
}

void Solution::saveLandmarks() {
  auto host_color_map = reader_->get_color_frame(image_idx_);
  cv::imshow("color", host_color_map);
  cv::waitKey();
  detector_->saveLandmarkCorrespondence(
      host_color_map, mesh_->cuda_position_sRT, mesh_->cuda_is_front_now,
      parameters_->cameraParams.camera_intr,
      "./results/landmarks_without_contour/" + std::to_string(image_idx_) +
          ".png");
}

#endif  // USE_CUDA
