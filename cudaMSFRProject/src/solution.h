#pragma once

#include <atomic>

#include "Common.h"
#ifdef USE_CUDA
#include <Windows.h>
#include <cuda_runtime.h>
#include <time.h>

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "BaselModel.h"
#include "GenModels_CUDA.h"
#include "HostUtil.hpp"
#include "LandmarkDetector.h"
#include "Parameters.h"
#include "oni_reader.h"
#include "solver.hpp"
#include "window.hpp"

enum BlendshapeLevel {
  DEFORMATION_TRANSFER = 0,
};

struct Status {
  bool isThreadWorking = false;
  bool isLoadingFusion = false;
  bool isStartTracking = false;
  bool isRetracking = false;
  bool isStarted = false;
  bool isStop = false;
  bool isSaveBlendshape = false;
  bool isSaveAvatar = false;
  bool isDebugSolveAlbedo = false;
  int poseOptNum = 3;
  int iterCnt = 0;
  double fpsOPT = 0;
  BlendshapeLevel blendshapeLevel;
  std::string sequence_folder;
  bool refineExp = true;
  bool saveRetrackingResult = false;

  std::string getBlendShapeLevelLiteral() {
    switch (blendshapeLevel) {
      case DEFORMATION_TRANSFER:
        return "deformation_transfer";
    }
    return {};
  }

  std::string getICPErrorPath() {
    return sequence_folder + "/" + getBlendShapeLevelLiteral() + "/error_" +
           std::to_string(int(refineExp)) + ".txt";
  }

  std::string getIdxString(int idx) {
    char buffer[10];
    sprintf(buffer, "%04d", idx);
    return std::string(buffer);
  }

  std::string getRetrackErrorPath(int idx) {
    const auto res = sequence_folder + "/" + getBlendShapeLevelLiteral() +
                     "/retrack_error" + std::to_string(int(refineExp));
    return res + "/" + getIdxString(idx) + ".png";
  }

  std::string getRetrackColorPath(int idx) {
    const auto res = sequence_folder + "/" + getBlendShapeLevelLiteral() +
                     "/retrack_color" + std::to_string(int(refineExp));
    return res + "/" + getIdxString(idx) + ".png";
  }
};

class Solution {
 public:
  static constexpr int MAX_KEY_FRAMES_NUM = 300;

 public:
  Solution()
      : model_(nullptr),
        parameters_(nullptr),
        mesh_(nullptr),
        gmodels_(nullptr),
        reader_(nullptr),
        detector_(nullptr),
        tex_(nullptr) {}

  Solution(cudaBaselModel& model, FrameParameters& parameters,
           meshToBeSolved& mesh, GenModels& gmodels, OniReader& reader,
           LandmarkDetector& detector, texture& tex)
      : model_(&model),
        parameters_(&parameters),
        mesh_(&mesh),
        gmodels_(&gmodels),
        reader_(&reader),
        detector_(&detector),
        tex_(&tex) {
    RT_solver_JTJ.create(36);
    RT_solver_Jr.create(6);
    SRT_solver_JTJ.create(49);
    SRT_solver_Jr.create(7);
    A_.create(200 * 200);
    b_.create(200);
    activate_basis_index_.create(200);
    cov_basis_.create(200);
    delta_coefficients_.create(200);
    unRTposition_.create(34508);
  }

  ~Solution() {
    is_should_exit = true;
    should_detect_cond.notify_one();
    is_update_blendshape_cond.notify_one();
    if (load_image_thread_ptr != nullptr && load_image_thread_ptr->joinable()) {
      load_image_thread_ptr->join();
    }
    if (update_blendshape_thread != nullptr &&
        update_blendshape_thread->joinable()) {
      update_blendshape_thread->join();
    }
  }

  void init() {
    std::vector<bool> is_sampled(mesh_->n_verts_, false);
    FILE* fp = fopen("./optimized_index.txt", "r");
    int optimized_num;
    fscanf(fp, "%d", &optimized_num);
    for (int i = 0; i < optimized_num; ++i) {
      int index;
      fscanf(fp, "%d", &index);
      is_sampled[index] = true;
    }
    fclose(fp);
    for (auto& iter : detector_->new_landmarkIndex) {
      is_sampled[iter.y] = true;
    }
    std::vector<int> host_optimized_index, host_optimized_key;
    for (int i = 0; i < mesh_->n_verts_; ++i) {
      if (is_sampled[i]) {
        host_optimized_index.push_back(host_optimized_key.size());
        host_optimized_key.push_back(i);
      } else {
        host_optimized_index.push_back(-1);
      }
    }
    sampled_index_.upload(host_optimized_index);
    sampled_key_.upload(host_optimized_key);
    sampled_num_ = sampled_key_.size();
    std::vector<int2> host_sampled_landmark_index(
        detector_->new_landmarkIndex.size());
    for (int i = 0; i < host_sampled_landmark_index.size(); ++i) {
      host_sampled_landmark_index[i] = detector_->new_landmarkIndex[i];
      host_sampled_landmark_index[i].y =
          host_optimized_index[host_sampled_landmark_index[i].y];
    }
    sampled_landmark_index_.upload(host_sampled_landmark_index);
    sampled_exp_base_.create(3 * sampled_num_ * model_->dim_exp_);
    sampled_exp_base_ATA_.create(model_->dim_exp_ * model_->dim_exp_);
    reg_sampled_exp_base_.create(model_->dim_exp_);
    smoothed_personalized_exp_base_.create(model_->dim_exp_ * mesh_->n_verts_ *
                                           3);
#ifdef USE_FACESHIFT_EXP
    parameters_->modelParams.cuda_id_coefficients_.create(model_->dim_id_);
    parameters_->modelParams.cuda_exp_coefficients_.create(model_->dim_exp_);
    parameters_->modelParams.cuda_exp_prev_1_.create(model_->dim_exp_);
    parameters_->modelParams.cuda_exp_prev_2_.create(model_->dim_exp_);
#endif
  }

  void setParameter(FrameParameters& parameters) { parameters_ = &parameters; }

  void setMesh(meshToBeSolved& mesh) { mesh_ = &mesh; }

  void setModel(cudaBaselModel& model) { model_ = &model; }

  void updateModel() {
    gmodels_->updateMesh2(model_name_, mesh_->cuda_position_sRT,
                          mesh_->cuda_color_, mesh_->cuda_normal_R,
                          mesh_->cudafvLookUpTable, mesh_->cudafBegin,
                          mesh_->cudafEnd, mesh_->cuda_fv_idx);
  }

  float solving_time = 0;

  void updateExpMesh() {
    model_->cudaUpdatePersonalizedModel(
        parameters_->modelParams.cuda_exp_coefficients_);
    model_->cuda_cur_shape_.copyTo(mesh_->cuda_position_);
    mesh_->update_normal();
    updateSRTPositionNormal();
  }

  void updateExpMeshStatic() {
    model_->cudaUpdateStaticPersonalizedModel(
        parameters_->modelParams.cuda_exp_coefficients_);
    model_->cuda_cur_shape_.copyTo(mesh_->cuda_position_);
    mesh_->update_normal();
    updateSRTPositionNormal();
  }

  void updateMesh() {
    model_->cudaUpdateModelID(parameters_->modelParams.cuda_id_coefficients_);
    model_->cuda_cur_shape_.copyTo(mesh_->cuda_position_);

    mesh_->update_normal();
    updateSRTPositionNormal();
  }

  void updateSRTPositionNormal();

  void updateLandmarkErr(const pcl::gpu::DeviceArray<int2> landmark_id,
                         const pcl::gpu::DeviceArray<float3> landmark_position);

  void prepareTarget();

  void prepareTargetPosition();

  void prepareTargetLandmark();

  void restart() {
    LOG(INFO) << "Restart!";
    std::vector<float> matrix_identity(9, 0), translation(4, 0);
    matrix_identity[0] = 1.0;
    matrix_identity[4] = 1.0;
    matrix_identity[8] = 1.0;
    translation[3] = 0.10f;
    parameters_->cuda_rotation_.upload(matrix_identity);
    parameters_->cuda_translation_.upload(translation);
    clearCudaMem(parameters_->modelParams.cuda_id_coefficients_);
    clearCudaMem(parameters_->modelParams.cuda_exp_coefficients_);
    clearCudaMem(mesh_->cuda_position_weight_);
    clearCudaMem(mesh_->cuda_optimal_weight_);
    updateMesh();
    model_->is_personalized_model_generated = false;
    image_idx_ = -1;
  }

  std::string save_folder_suffix = "";
  void saveCameraView();
  void loadCameraView();

  void save_exp();
  void load_exp();
  void save_pose();
  void load_pose();
  void load_pose_from_file(const std::string filename_ = std::string());
  void show_color_map();

  /**
   * @brief Save Rendered Color Image
   * @param filename The path and the name of the image
   * @param canvas The canvas that used to render this image
   * @param color_mode: RGBA 0 RGB 1
   */
  void save_color_map(const std::string& filename = std::string(),
                      const std::string& canvas = std::string(canvas_name_),
                      const int color_mode = 0, const bool is_save_now = false);

  int image_idx_ = -1;
  bool next_frame_has_landmark = false;
  bool cur_frame_has_landmark;

  void loadNextImage() {
    while (!is_should_exit) {
      {
        std::unique_lock<std::mutex> lk(shold_detect_mutex);
        if (!should_detect) {
          should_detect_cond.wait(
              lk, [&]() { return should_detect || is_should_exit; });
        }
        should_detect = false;
      }
      while (save_image_list.size() > 0) {
        LOG(INFO) << "DEBUG save image " << save_image_list[0].first;
        cv::imwrite(save_image_list[0].first, save_image_list[0].second);
        save_image_list.pop_front();
      }
      if (is_should_exit) {
        break;
      }
      if (image_idx_ + 1 < reader_->get_color_frame_num()) {
        auto host_color_map = reader_->get_color_frame(image_idx_ + 1);
        {
          detector_->readLandmarksFromFile(
              reader_->get_landmark_filename(image_idx_ + 1));
        }
#if ONLINE_SAVE_SEQ
        if (reader_->read_from_device) {
          reader_->saveFrame("results", image_idx_ + 1);
        }
#endif
        {
          std::unique_lock<std::mutex> lk(has_lm_mutex);
          has_lm = true;
          has_lm_cond.notify_one();
        }
      }
    }
  }

  std::thread* load_image_thread_ptr = nullptr;
  bool should_detect = false, has_lm = false;
  bool is_should_exit = false;
  std::mutex has_lm_mutex, shold_detect_mutex;
  std::condition_variable should_detect_cond, has_lm_cond;
  bool read_lm_from_file = false;
  bool lock_image_index = false;

  void setImageIndex(int idx, bool is_show_landmark = false) {
    if (lock_image_index) {
      return;
    }
    if (idx < reader_->get_color_frame_num()) {
      if (idx != 1) {
        if (idx == 0 || image_idx_ != idx - 1) {
          auto host_color_map = reader_->get_color_frame(idx);
         {
            detector_->readLandmarksFromFile(
                reader_->get_landmark_filename(idx));
          }

#if ONLINE_SAVE_SEQ
          if (reader_->read_from_device) {
            reader_->saveFrame("results", 0);
          }
#endif
          has_lm = true;
        }

        {
          std::unique_lock<std::mutex> lk(has_lm_mutex);
          if (!has_lm) {
            has_lm_cond.wait(lk, [&] { return has_lm; });
          }
          has_lm = false;
        }

        auto color_map = reader_->get_color_frame_device(idx);

        cur_frame_has_landmark = next_frame_has_landmark;

        swapInputColorMap();
        gmodels_->setImage(canvas_name(), input_color_map_handle_id(),
                           color_map);
        detector_->getLandmarkPos(parameters_->cameraParams.camera_intr);

        {
          std::unique_lock<std::mutex> lk(shold_detect_mutex);
          should_detect = true;
          should_detect_cond.notify_one();
        }
      } else {
        reader_->mock_cur_frame(idx);
      }
      image_idx_ = idx;

      if (load_image_thread_ptr == nullptr) {
        load_image_thread_ptr = new std::thread(&Solution::loadNextImage, this);
      }
    }
  }

  const std::string canvas_name() const { return std::string(canvas_name_); }
  const std::string large_canvas_name() const {
    return std::string(large_canvas_name_);
  }
  const std::string model_name() const { return std::string(model_name_); }
  const int color_handle_id() const { return 0; }

  int cur = 2;
  bool zeroFrame = true;
  bool firstFrame = true;
  const int prev_color_map_handle_id() const {
    if (firstFrame) {
      return cur;
    }
    return 2 + 9 - cur;
  }
  const int input_color_map_handle_id() const { return cur; }
  void swapInputColorMap() {
    if (zeroFrame) {
      zeroFrame = false;
      return;
    } else if (firstFrame) {
      firstFrame = false;
    }
    cur = 2 + 9 - cur;
  }


  pcl::gpu::DeviceArray2D<float4> get_color_image(
      const std::string canvas = std::string(canvas_name_)) {
    auto image = gmodels_->GameModelList[canvas];
    pcl::gpu::DeviceArray2D<float4> color_image(image.height, image.width);
    gmodels_->getImage(canvas, color_handle_id(), color_image);
    return color_image;
  }


  pcl::gpu::DeviceArray2D<float4> get_input_color_image() {
    pcl::gpu::DeviceArray2D<float4> color_image(reader_->get_color_height(),
                                                reader_->get_color_width());
    gmodels_->getImage(canvas_name(), input_color_map_handle_id(), color_image);
    return color_image;
  }

  pcl::gpu::DeviceArray2D<float4> get_prev_input_color_image() {
    pcl::gpu::DeviceArray2D<float4> color_image(reader_->get_color_height(),
                                                reader_->get_color_width());
    gmodels_->getImage(canvas_name(), prev_color_map_handle_id(), color_image);
    return color_image;
  }

  void updateContourLandmarkIndex();
  void pushContourLandmarkTargetVertices(const float lambda,
                                         const bool is_use_P2L = true);

  void pushLandmarkTargetVertices(
      const pcl::gpu::DeviceArray<int2> indices,
      const pcl::gpu::DeviceArray<float3> target_position, const float lambda);
  void pushLandmarkTargetVertices(const float lambda,
                                  const bool is_use_P2L = true);
  void pushLandmarkTargetVertices_fuse(const float lambda);
  void pushNoseLandmarkTargetVertices(const float lambda);
  void pushExpLandmarkTargetVertices(const float lambda,
                                     const float eyes_lambda,
                                     const float eyebrow_lambda);

  /// \todo: These two function can be combined into one function using one
  /// CUDA kernel

  /// \todo: These functions should use CUDA texture object to accelerate


  cudaBaselModel* model_;
  FrameParameters* parameters_;
  meshToBeSolved* mesh_;
  GenModels* gmodels_;
  OniReader* reader_;
  LandmarkDetector* detector_;
  texture* tex_;
  renderer::renderInfo rinfo_;
  GLuint renderer;

  float landmark_err_;

  float blendshape_err_;
  std::vector<std::pair<int, float>> blendshape_err_list_;


  static constexpr char* canvas_name_ = "canvas";
  static constexpr char* model_name_ = "dynamicFace_MRT";
  static constexpr char* large_canvas_name_ = "canvas2";

  /// key frames
  std::atomic_int key_frames_num_;

  pcl::gpu::DeviceArray<float3> key_frames_position_;
  pcl::gpu::DeviceArray<float> key_frames_coefficients_;
  pcl::gpu::DeviceArray<float> CTC_;
  pcl::gpu::DeviceArray<float> CTY_;
  pcl::gpu::DeviceArray<float> D_;
  pcl::gpu::DeviceArray<float> sor_personalized_blendshape_;

  pcl::gpu::DeviceArray<int> activated_index_, corr_activated_index_;
  /// solution

  pcl::gpu::DeviceArray<float> J_weights, r_weights;

  int sampled_num_;
  pcl::gpu::DeviceArray<int> sampled_index_, sampled_key_;
  pcl::gpu::DeviceArray<int2> sampled_landmark_index_;
  pcl::gpu::DeviceArray<float> sampled_exp_base_, sampled_exp_base_ATA_,
      sampled_lambda_, reg_sampled_exp_base_;
  pcl::gpu::DeviceArray<float3> sampled_target_position_, sampled_position_,
      sampled_normal_, sampled_target_position_inv_sRT_;
  void sampleExpBase();

  void updatePersonalizedExpBasefromDeformationTransfer();
  pcl::gpu::DeviceArray<float> smoothed_personalized_exp_base_;
  std::thread* update_blendshape_thread = nullptr;
  std::mutex blendshape_mutex;
  std::condition_variable is_update_blendshape_cond;
  bool is_update_blendshape = false;
  int pre_keyframes_num = 0;

  void smoothMesh();

  cudaShootingSolver shootingSolver;
  pcl::gpu::DeviceArray<float> A_, b_, delta_coefficients_, cov_basis_;
  pcl::gpu::DeviceArray<int> activate_basis_index_;
  int activate_base_num_;
  pcl::gpu::DeviceArray<float3> unRTposition_;
  void solveIDCoefficient(const float lambda);
  void solveIDCoefficientPoint2Line(const float lambda);
  void solveSparseCoefficient(const float reg_lambda,
                              const float temporal_smooth_lambda,
                              const float L1_reg_lambda = 50.0f);
  void solveSampledSparseCoefficient(const float reg_lambda,
                                     const float temporal_smooth_lambda,
                                     const float L1_reg_lambda = 50.0f);


  void solveScale();

  pcl::gpu::DeviceArray<float> RT_solver_JTJ, SRT_solver_JTJ;
  pcl::gpu::DeviceArray<float> RT_solver_Jr, SRT_solver_Jr;
  void solveScaleRT();
  void solveRTwithNormal(const bool P2L = true);
  void solveSRTwithNormal();

  void showLandmarks();
  void saveLandmarks();
  void saveLandmarkErrors(std::string file_name);
  void saveCorrespond(std::string file_name);
  void saveModelFace(std::string file_name);
  void saveSRTModelFace(std::string file_name);
  void saveCoefficients(std::string file_name);
  std::deque<std::pair<std::string, cv::Mat>> save_image_list;
  std::vector<std::pair<int, std::vector<float>>> exp_coefficients_list,
      sRT_list;
};

struct TimingInfo {
  cudaEvent_t startEvent;
  cudaEvent_t endEvent;
  float duration;
  std::string eventName;
};

struct CUDATimer {
  std::vector<TimingInfo> timingEvents;
  int currentIteration;
  cudaStream_t stream;

  CUDATimer(cudaStream_t stream = nullptr)
      : currentIteration(0), stream(stream) {
#if USE_CUDA_TIMER == 1
    TimingInfo timingInfo;
    cudaEventCreate(&timingInfo.startEvent);
    cudaEventCreate(&timingInfo.endEvent);
    cudaEventRecord(timingInfo.startEvent);
    timingInfo.eventName = "overall";
    timingEvents.push_back(timingInfo);
#endif
  }

  ~CUDATimer() {
#if USE_CUDA_TIMER == 1
    for (auto& info : timingEvents) {
      cudaEventDestroy(info.startEvent);
      cudaEventDestroy(info.endEvent);
    }
#endif
  }

  void nextIteration() {
#if USE_CUDA_TIMER == 1
    ++currentIteration;
#endif
  }

  void reset() {
#if USE_CUDA_TIMER == 1
    currentIteration = 0;
    timingEvents.clear();
#endif
  }

  void startEvent(const std::string& name) {
#if USE_CUDA_TIMER == 1
    TimingInfo timingInfo;
    cudaEventCreate(&timingInfo.startEvent);
    cudaEventCreate(&timingInfo.endEvent);
    cudaEventRecord(timingInfo.startEvent, stream);
    timingInfo.eventName = name;
    timingEvents.push_back(timingInfo);
#endif
  }

  void endEvent() {
#if USE_CUDA_TIMER == 1
    TimingInfo& timingInfo = timingEvents[timingEvents.size() - 1];
    cudaEventRecord(timingInfo.endEvent, stream);
#endif
  }

  void evaluate() {
#if USE_CUDA_TIMER == 1
    cudaEventRecord(timingEvents[0].endEvent);
    std::vector<std::string> aggregateTimingNames;
    std::vector<float> aggregateTimes;
    std::vector<int> aggregateCounts;
    for (int i = 0; i < timingEvents.size(); ++i) {
      TimingInfo& eventInfo = timingEvents[i];
      cudaEventSynchronize(eventInfo.endEvent);
      cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent,
                           eventInfo.endEvent);
      int index = findFirstIndex(aggregateTimingNames, eventInfo.eventName);
      if (index < 0) {
        aggregateTimingNames.push_back(eventInfo.eventName);
        aggregateTimes.push_back(eventInfo.duration);
        aggregateCounts.push_back(1);
      } else {
        aggregateTimes[index] = aggregateTimes[index] + eventInfo.duration;
        aggregateCounts[index] = aggregateCounts[index] + 1;
      }
    }
    printf("------------------------------------------------------------\n");
    printf("          Kernel          |   Count  |   Total   | Average \n");
    printf("--------------------------+----------+-----------+----------\n");
    for (int i = 0; i < aggregateTimingNames.size(); ++i) {
      printf("--------------------------+----------+-----------+----------\n");
      printf(" %-24s |   %4d   | %8.3fms| %7.4fms\n",
             aggregateTimingNames[i].c_str(), aggregateCounts[i],
             aggregateTimes[i], aggregateTimes[i] / aggregateCounts[i]);
    }
    printf("------------------------------------------------------------\n");
#endif
  }
};

#endif  // USE_CUDA
