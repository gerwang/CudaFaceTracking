// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <Windows.h>
#include <time.h>

#include <cstdlib>

#include "HostUtil.hpp"
#include "window.hpp"  // Include short list of convenience functions for rendering

/// include faceFitting
#include <BaselModel.h>
#include <LandmarkDetector.h>
#include <Parameters.h>
#ifdef USE_CUDA
#include <GenModels_CUDA.h>
#include <oni_reader.h>
#include <solution.h>
#else
#include <GenModels.h>
#endif  // USE_CUDA

#include <LandmarkUtil.h>
#include <io.h>

#include <fstream>
#include <string>
#include <thread>

#include "HostUtil.hpp"
#include "LambdaSetting.h"
#include "ShaderLoader.h"
#include "avatar.h"
#include "glog/logging.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"

void GenInit();

void int2str(const int &, std::string &);

void optimizingThread();
void nonRigidThread();

texture model_image;
cudaBaselModel BModel;
FrameParameters Parameters, Parameters_fusion;
GenModels *gModels = new GenModels;
meshToBeSolved CurrentFace;
OniReader Reader(OniReader::OFFLINE_MODE);
LandmarkDetector landmarkDetector;
Solution mainSolution(BModel, Parameters, CurrentFace, *gModels, Reader,
                      landmarkDetector, model_image);
renderer::renderInfo rInfo;
static int updateBlendshape_cnt = 0;

GLuint program;
GLuint programBall;

GLuint programIsomap;
GLuint programIsomapfromcolor;
GLuint programTriangle;

struct {
  int isShowVertex = 0;    // 0 normal	1 show ICP points	2 show Err
  bool isOptimize = true;  // true optimize false test
  int fusionRenderColorMode = 5;
  int modelRenderColorMode = 5;
  GLenum modelRenderShape = GL_TRIANGLES;
} functionOn;

struct {
  float lambda = 1.5f, LandmarkLambda = 1.6f, EyeLandmarkLambda = 500.0f,
        ICPLambda = 1.4e-2f, ContourLandmarkLambda = 0.000,
        temporal_smooth_exp = 200.0f;
  float SmoothLambda = 5.f;
  bool use_ICP = false;
  bool solve_Identity = false;
  bool use_exp = false;
} lambdaSetting;

Status status;

// GUI

struct {
  bool isOpen = false;
} imGuiStatus;

cv::Mat color_map;

int landmarkTriIndex[FACIAL_POINTS_NUM_FLT];
float landmarkDectected[FACIAL_POINTS_NUM_FLT][3];

window *refApp;

int save_pose_idx;

std::shared_ptr<Avatar> avatar;

#if DELAY_ONE_FRAME
pcl::gpu::DeviceArray<float3> prevPosition0, prevNormal0;
pcl::gpu::DeviceArray<float3> prevPosition1, prevNormal1;
bool firstFrame = true;
#endif

#define PROFILE_ONLINE 0

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main(int argc, char *argv[]) {
  LambdaSetting &lambdas = LambdaSetting::get_instance();

  // load parameters
  std::ifstream t("./configuration.json");
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  rapidjson::Document document;
  document.Parse(str.c_str());
  const auto color_folder = document["color_folder"].GetString();
  const auto avatar_folder = document["avatar_folder"].GetString();
  avatar = std::make_shared<Avatar>(avatar_folder, 51);
  {
    auto tmpFolder = std::string(color_folder);
    status.sequence_folder = tmpFolder.substr(0, tmpFolder.rfind('/'));
  }
  status.refineExp = document["refine_exp"].GetBool();

  status.saveRetrackingResult = document["save_retracking_result"].GetBool();
  const auto image_width = document["image_width"].GetInt();
  const auto image_height = document["image_height"].GetInt();
  mainSolution.parameters_->lambda = document["pose_smooth_coeff"].GetFloat();
  mainSolution.parameters_->lambda_1 =
      document["pose_smooth_coeff_1"].GetFloat();
  mainSolution.parameters_->lambda_2 =
      document["pose_smooth_coeff_2"].GetFloat();
  mainSolution.parameters_->lambda_3 =
      document["pose_smooth_coeff_3"].GetFloat();
  lambdaSetting.temporal_smooth_exp =
      document["temporal_smooth_exp"].GetFloat();

  lambdaSetting.LandmarkLambda = document["LandmarkLambda"].GetFloat();
  mainSolution.read_lm_from_file = document["read_lm_from_file"].GetBool();
  status.blendshapeLevel =
      static_cast<BlendshapeLevel>(document["blendshape_level"].GetInt());
  status.isSaveBlendshape = document["is_save_blendshape"].GetBool();
  status.isSaveAvatar = document["is_save_avatar"].GetBool();
  mainSolution.save_folder_suffix = document["save_folder_suffix"].GetString();
  save_pose_idx = document["pose_idx"].GetInt();
  Reader.open(color_folder, image_width, image_height);

  FLAGS_log_dir = "./Logs";
  google::InitGoogleLogging(__FILE__);
  google::SetStderrLogging(google::GLOG_INFO);

  // Create a simple OpenGL window for rendering:

  window app(1280, 720, "MSFR");
  refApp = &app;
  bool spacePressed = false, enterPressed = false;
  app.ON_PRESS_W =
      std::bind(&CameraBlock::ON_PRESS_W,
                &mainSolution.parameters_->cameraParams, std::placeholders::_1);
  app.ON_PRESS_A =
      std::bind(&CameraBlock::ON_PRESS_A,
                &mainSolution.parameters_->cameraParams, std::placeholders::_1);
  app.ON_PRESS_S =
      std::bind(&CameraBlock::ON_PRESS_S,
                &mainSolution.parameters_->cameraParams, std::placeholders::_1);
  app.ON_PRESS_D =
      std::bind(&CameraBlock::ON_PRESS_D,
                &mainSolution.parameters_->cameraParams, std::placeholders::_1);
  app.ON_PRESS_R = std::bind(&CameraBlock::ON_PRESS_R,
                             &mainSolution.parameters_->cameraParams);
  app.ON_PRESS_SPACE = [&]() { spacePressed = true; };
  app.ON_PRESS_ENTER = [&]() { enterPressed = true; };
  app.ON_PRESS_ESCAPE = [&]() { status.isStop = true; };
  app.on_mouse_move = std::bind(&CameraBlock::on_mouse_move,
                                &mainSolution.parameters_->cameraParams,
                                std::placeholders::_1, std::placeholders::_2);

  // 3DMM Init
  GenInit();
  mainSolution.init();
  mainSolution.model_->updateLaplacianSmoothMask(*mainSolution.mesh_);

  rs2_intrinsics *colorIntrinsic = new rs2_intrinsics;

  *colorIntrinsic = mainSolution.reader_->get_color_intrinsics();
  LOG(INFO) << "Color Camera Intrinsics: fx: " << colorIntrinsic->fx
            << " fy: " << colorIntrinsic->fy << " ppx: " << colorIntrinsic->ppx
            << " ppy: " << colorIntrinsic->ppy
            << " height: " << colorIntrinsic->height
            << " width: " << colorIntrinsic->width;

  {
    auto &camera_intr = Parameters.cameraParams.camera_intr;
    camera_intr.fx = colorIntrinsic->fx;
    camera_intr.fy = colorIntrinsic->fy;
    camera_intr.cx = colorIntrinsic->ppx;
    camera_intr.cy = colorIntrinsic->ppy;
  }
  rInfo.width_ = 640;
  rInfo.height_ = 480;

  mainSolution.updateMesh();

  {
    std::vector<float> translation(4, 0);
    translation[3] = 0.1f;
    Parameters.cuda_translation_.upload(translation);
  }
  mainSolution.restart();
  rInfo.parameters_ = &Parameters;
  rInfo.camera_ = *colorIntrinsic;
  rInfo.emode_ = functionOn.modelRenderShape;
  mainSolution.rinfo_ = rInfo;
  mainSolution.renderer = program;

  if (!Reader.read_from_device) {
    status.isStarted = true;
  }

  mainSolution.load_pose_from_file("data/model/init_pose.txt");
  while (app)  // Application still alive?
  {
    // Start the Dear ImGui frame
#if PROFILE_ONLINE
    SimpleTimer timer;
    timer.init();
#endif

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    {
      static float f = 0.0f;
      static int counter = 0;
      ImGui::Begin("Information Box");  // Create a window called "Hello,
                                        // world!" and append into it.
      ImGui::Text("Image Index: %d Key Frames: %d Solving Time: %f",
                  mainSolution.image_idx_, mainSolution.key_frames_num_,
                  mainSolution.solving_time);
      static bool lock_camera_view = true;
      ImGui::Checkbox("Lock Camera View", &lock_camera_view);
      ImGui::Checkbox("Lock Image Index", &mainSolution.lock_image_index);
      if (lock_camera_view) {
        mainSolution.parameters_->cameraParams.lock_camera_view();
      } else {
        mainSolution.parameters_->cameraParams.unlock_camera_view();
      }

      static std::vector<float> exp_coeffi;
      mainSolution.parameters_->modelParams.cuda_exp_coefficients_.download(
          exp_coeffi);
      ImGui::PlotHistogram("Histogram", exp_coeffi.data(), exp_coeffi.size(), 0,
                           NULL, 0.0f, 1.0f, ImVec2(0, 80));
      if (ImGui::Button("Open")) {
        imGuiStatus.isOpen = true;
      }

      if (ImGui::Button("Restart")) {
        mainSolution.restart();

        status.iterCnt = 0;
        lambdaSetting.solve_Identity = false;
        lambdaSetting.use_ICP = false;
        status.isStartTracking = false;
      }
      ImGui::SameLine();
      {
        bool is_optimize = functionOn.isOptimize;
        ImGui::Checkbox("Optimize", &functionOn.isOptimize);
        // if (is_optimize && !functionOn.isOptimize)
        //{
        //  status.iterCnt = 0;
        //}
      }
      static int start_cnt = 0;
      if (!status.isStarted && spacePressed) {
        status.isStarted = true;
      }
      if (status.isStarted) start_cnt++;
      if ((ImGui::Button("Start Tracking") || enterPressed || start_cnt == 5) &&
          !status.isStartTracking &&
          mainSolution.reader_->get_read_mode() ==
              OniReader::ONLINE_MODE)  // enter
      {
        mainSolution.mesh_->position().copyTo(
            mainSolution.model_->cuda_personalized_mean_shape_);
        // mainSolution.updatePersonalizedExpBasefromDeformationTransfer();
        std::thread t(
            &Solution::updatePersonalizedExpBasefromDeformationTransfer,
            &mainSolution);
        t.detach();
        status.isStartTracking = true;
      }
      ImGui::SameLine();
      if (mainSolution.model_->is_personalized_model_generated &&
          status.isStartTracking) {
        ImGui::Text("Tracking...");
      } else {
        ImGui::Text("Neutral Face Reconstruction...");
      }
      static bool is_show_landmark = false;
      ImGui::Checkbox("Show Landmark", &is_show_landmark);
      if (is_show_landmark) {
        mainSolution.showLandmarks();
      }
      static int blendshape_name = 999;
      ImGui::InputInt("BlendshapeName", &blendshape_name);
      if (ImGui::Button("Save Blendshape")) {
        mainSolution.model_->saveExpBlendshapeModel(
            CurrentFace, "./results/blendshapes/", blendshape_name);
      }
      ImGui::SameLine();
      if (ImGui::Button("Deformation Transfer")) {
        mainSolution.model_->UpdatePersonalizedExpBase(CurrentFace);
      }
      if (ImGui::TreeNode("Camera Parameters")) {
        CameraBlock &camera = rInfo.parameters_->cameraParams;
        static float position[3], front[3], up[3];
        static bool flag = true;
        if (flag) {
          position[0] = camera.get_position().x;
          position[1] = camera.get_position().y;
          position[2] = camera.get_position().z;

          front[0] = camera.get_front().x;
          front[1] = camera.get_front().y;
          front[2] = camera.get_front().z;

          up[0] = camera.get_up().x;
          up[1] = camera.get_up().y;
          up[2] = camera.get_up().z;
          flag = false;
        }
        ImGui::DragFloat3("Camera position", position, 0.01);
        ImGui::DragFloat3("Camera front", front, 0.01);
        ImGui::DragFloat3("Camera up", up, 0.01);
        camera.set_position(*reinterpret_cast<glm::vec3 *>(position));
        camera.set_front(*reinterpret_cast<glm::vec3 *>(front));
        camera.set_up(*reinterpret_cast<glm::vec3 *>(up));
        ImGui::TreePop();
      }
      {
        bool previous_is_use_exp = lambdaSetting.use_exp;
        ImGui::Checkbox("Use Exp", &lambdaSetting.use_exp);
        if (lambdaSetting.use_exp != previous_is_use_exp &&
            !lambdaSetting.use_exp) {
          clearCudaMem(
              mainSolution.parameters_->modelParams.cuda_exp_coefficients_);
          Parameters.modelParams.expr_coef_.setZero();
        }
      }
      lambdas.show();
      if (lambdaSetting.use_exp) {
        ImGui::Checkbox("Use Personalized Exp",
                        &mainSolution.model_->is_use_personalized_model);
        ImGui::SameLine();
        if (ImGui::TreeNode("Expression Coefficients")) {
          bool is_exp_coefficients_changed = false;
          for (int i = 0; i < BModel.dim_exp_; i++) {
            float previous_expc = Parameters.modelParams.expr_coef_[i];
            ImGui::SliderFloat((std::string("#") + std::to_string(i)).c_str(),
                               &Parameters.modelParams.expr_coef_[i], 0.0f,
                               1.0f, "%.4f");
            if (previous_expc != Parameters.modelParams.expr_coef_[i]) {
              is_exp_coefficients_changed = true;
            }
          }
          ImGui::TreePop();
          if (is_exp_coefficients_changed) {
            mainSolution.parameters_->modelParams.cuda_exp_coefficients_.upload(
                Parameters.modelParams.expr_coef_.data(),
                Parameters.modelParams.expr_coef_.size());
            mainSolution.updateMesh();
          }
        }
      }

      ImGui::Text("Scale: %.4f", Parameters.cameraParams.orthoScale);
      ImGui::Text("Rotation: R_x:%.4f R_y:%.4f R_z:%.4f",
                  Parameters.modelParams.R_x, Parameters.modelParams.R_y,
                  Parameters.modelParams.R_z);
      ImGui::Text("Translation: T_x:%.4f T_y:%.4f T_z:%.4f",
                  Parameters.modelParams.T_x, Parameters.modelParams.T_y,
                  Parameters.modelParams.T_z);
      {
        static float filer_c = 0.6f;
        static float landmark_err = 0.0f;
        static float icp_err = 0.0f;

        landmark_err =
            landmark_err * filer_c + mainSolution.landmark_err_ * (1 - filer_c);

        ImGui::Text("Landmark Error: %.6f", landmark_err);
        ImGui::SameLine();
        ImGui::Text("ICP Error: %.3e", icp_err);
        ImGui::SameLine();
        ImGui::SliderFloat("Error Filter", &filer_c, 0.0f, 1.0f, "%.4f");
      }
      ImGui::SliderFloat("Landmark", &lambdaSetting.LandmarkLambda, 0.0f, 10.0f,
                         "%.4f");
      ImGui::SliderFloat("ICP lambda", &lambdaSetting.ICPLambda, 0.0f, 2e-1f,
                         "%.6f");
      ImGui::SameLine();
      ImGui::Checkbox("Use ICP", &lambdaSetting.use_ICP);
      ImGui::Checkbox("Solve Identity", &lambdaSetting.solve_Identity);
      ImGui::SliderFloat("Smooth lambda", &lambdaSetting.SmoothLambda, 0.0f,
                         10.0f, "%.6f");
      ImGui::SliderFloat("Regulation", &lambdaSetting.lambda, 0.0f, 2.0f,
                         "%.4f");
      ImGui::Text("Show Elements");

      static int model_render_shape =
          static_cast<GLenum>(functionOn.modelRenderShape);
      ImGui::RadioButton("Faces", &model_render_shape, GL_TRIANGLES);
      ImGui::SameLine();
      ImGui::RadioButton("Points", &model_render_shape, GL_POINTS);
      ImGui::RadioButton("White", (int *)&rInfo.rmode_, renderer::WHITE);
      ImGui::SameLine();
      ImGui::RadioButton("ICP", (int *)&rInfo.rmode_, renderer::ICP);
      ImGui::SameLine();
      ImGui::RadioButton("ERROR", (int *)&rInfo.rmode_, renderer::ERR);
      ImGui::SameLine();
      ImGui::RadioButton("DEPTH", (int *)&rInfo.rmode_, renderer::DEPTH);
      ImGui::SameLine();
      ImGui::RadioButton("ALBEDO", (int *)&rInfo.rmode_, renderer::ALBEDO);
      ImGui::SameLine();
      ImGui::RadioButton("RENDER", (int *)&rInfo.rmode_, renderer::RENDER);
      ImGui::SameLine();
      ImGui::RadioButton("WEIGHT (SH)", (int *)&rInfo.rmode_, renderer::WEIGHT);
      ImGui::SameLine();
      ImGui::RadioButton("SH FALSE", (int *)&rInfo.rmode_, renderer::SH_FALSE);

      functionOn.modelRenderShape = static_cast<int>(model_render_shape);

      ImGui::Text("Color Mode");
      ImGui::RadioButton("Common", &functionOn.modelRenderColorMode, 5);
      ImGui::SameLine();

      for (int i = 0; i < 9; i++) {
        float tmp = rInfo.parameters_->modelParams.shParams[i * 3];
        ImGui::DragFloat(("SH" + std::to_string(i)).c_str(), &tmp, 0.01);
        for (int j = 0; j < 3; j++) {
          rInfo.parameters_->modelParams.shParams[i * 3 + j] = tmp;
        }
      }

      ImGui::Text("iterCnt = %d", status.iterCnt);
      ImGui::Text("update blendshape cnt: %d", updateBlendshape_cnt);
      ImGui::Text(
          "Application average %.3f ms/frame (%.1f FPS), Optimization "
          "%.3f ms/frame (%.1f FPS)",
          1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate,
          1000 / status.fpsOPT, status.fpsOPT);
      ImGui::End();
    }

#if PROFILE_ONLINE
    timer.report("ImGui");
#endif

    rInfo.camera_ = *colorIntrinsic;
    rInfo.emode_ = functionOn.modelRenderShape;

    if (!status.isThreadWorking) {
      if (mainSolution.reader_->read_from_device) {
        if (status.isStop || !status.isStarted ||
            !mainSolution.cur_frame_has_landmark) {
          mainSolution.setImageIndex(mainSolution.image_idx_ + 1);
        } else if (!(mainSolution.model_->is_personalized_model_generated &&
                     status.isStartTracking)) {
          optimizingThread();
        } else {
          nonRigidThread();
        }
      } else {
        if (functionOn.isOptimize) {
          static bool is_retracking = false;
          if (mainSolution.image_idx_ < 5) {
            optimizingThread();
            if (mainSolution.image_idx_ == 5 - 1) {
              mainSolution.mesh_->position().copyTo(
                  mainSolution.model_->cuda_personalized_mean_shape_);
              mainSolution.updatePersonalizedExpBasefromDeformationTransfer();

              std::vector<float> host_rotation, host_translation;
              mainSolution.parameters_->cuda_rotation_.download(host_rotation);
              mainSolution.parameters_->cuda_translation_.download(
                  host_translation);
              for (auto iter : host_translation) {
                host_rotation.push_back(iter);
              }
              mainSolution.sRT_list.push_back(
                  {mainSolution.image_idx_, host_rotation});
              status.isStartTracking = true;
              if (status.blendshapeLevel == DEFORMATION_TRANSFER) {
                status.isRetracking = true;
              }
            }
          } else if (mainSolution.image_idx_ <
                     mainSolution.reader_->get_color_frame_num()) {
            if (!status.isRetracking) {
              nonRigidThread();
            } else {
              if (!status.saveRetrackingResult && status.isSaveBlendshape) {
                mainSolution.image_idx_ =
                    mainSolution.reader_->get_color_frame_num();
              }
              exit(0);  // fixme
            }
          }
        }
      }
      mainSolution.updateLandmarkErr(landmarkDetector.cuda_landmark_index_,
                                     landmarkDetector.cuda_landmark_position_);
    }
#if PROFILE_ONLINE
    timer.report("optimize");
#endif

    mainSolution.updateSRTPositionNormal();
#if DELAY_ONE_FRAME
    if (firstFrame) {
      mainSolution.updateModel();
    } else {
      gModels->updateMesh2(
          "dynamicFace_MRT", prevPosition0, CurrentFace.cuda_color_,
          prevNormal0, CurrentFace.cudafvLookUpTable, CurrentFace.cudafBegin,
          CurrentFace.cudafEnd, CurrentFace.cuda_fv_idx);
    }
    CurrentFace.cuda_position_sRT.copyTo(prevPosition0);
    CurrentFace.cuda_normal_R.copyTo(prevNormal0);
#else
    mainSolution.updateModel();
#endif

    {
      rect r;
      if (!status.isStartTracking) {
        auto imgWidth =
            app.height() / Reader.get_color_height() * Reader.get_color_width();
        r = {app.width() / 2 - imgWidth / 2, 0, imgWidth, app.height()};
      } else {
        auto imgHeight = app.width() / 2 / Reader.get_color_width() *
                         Reader.get_color_height();
        r = {0, app.height() / 2 - imgHeight / 2, app.width() / 2, imgHeight};
      }
      gModels->updateMesh("dynamicFace_MRT", CurrentFace.cuda_position_sRT,
                          CurrentFace.cuda_color_, CurrentFace.cuda_normal_R,
                          CurrentFace.cuda_tri_list_);
      model_image.render(r, gModels->GameModelList["canvas"].width,
                         gModels->GameModelList["canvas"].height,
#if DELAY_ONE_FRAME
                         gModels->GameModelList["canvas"]
                             .vbos[mainSolution.prev_color_map_handle_id()],
#else
                         gModels->GameModelList["canvas"]
                             .vbos[mainSolution.input_color_map_handle_id()],
#endif
                         app.width(), app.height(), "Color");
      model_image.renderModel(*gModels, "canvas", "dynamicFace_MRT", rInfo,
                              program, true);
      model_image.render(r, gModels->GameModelList["canvas"].width,
                         gModels->GameModelList["canvas"].height,
                         gModels->GameModelList["canvas"].vbos[0], app.width(),
                         app.height(), "3DMM Face");
      if (status.isRetracking) {
#if DELAY_ONE_FRAME
        int idx = mainSolution.image_idx_ - 1;
#else
        int idx = mainSolution.image_idx_;
#endif
      }
    }

    mainSolution.updateSRTPositionNormal();
#if DELAY_ONE_FRAME
    if (firstFrame) {
      mainSolution.updateModel();
      firstFrame = false;
    } else {
      gModels->updateMesh2(
          "dynamicFace_MRT", prevPosition1, CurrentFace.cuda_color_,
          prevNormal1, CurrentFace.cudafvLookUpTable, CurrentFace.cudafBegin,
          CurrentFace.cudafEnd, CurrentFace.cuda_fv_idx);
    }
    CurrentFace.cuda_position_sRT.copyTo(prevPosition1);
    CurrentFace.cuda_normal_R.copyTo(prevNormal1);
#else
    mainSolution.updateModel();
#endif
    if (status.isStartTracking) {
      auto imgHeight = app.width() / 2 / Reader.get_color_width() *
                       Reader.get_color_height();
      rect r = {app.width() / 2, app.height() / 2 - imgHeight / 2,
                app.width() / 2, imgHeight};

      model_image.renderModel(*gModels, "canvas", "dynamicFace_MRT", rInfo,
                              program, true);
      model_image.renderModel(*gModels, mainSolution.large_canvas_name(),
                              "dynamicFace_MRT", rInfo, program, true);
      if (status.saveRetrackingResult && status.isRetracking) {
        mainSolution.save_color_map(
            status.sequence_folder + "/retracking_warped_" +
                status.getBlendShapeLevelLiteral() +
                mainSolution.save_folder_suffix + "/" +
                std::to_string(mainSolution.image_idx_) + ".png",
            mainSolution.large_canvas_name());
      }

      gModels->updateMesh("avatar", avatar->position_sRT,
                          avatar->mesh.cuda_color_, avatar->normal_R,
                          avatar->mesh.cuda_tri_list_);
      model_image.renderModel(*gModels, "canvas", "avatar", rInfo, program);
      model_image.render(r, gModels->GameModelList["canvas"].width,
                         gModels->GameModelList["canvas"].height,
                         gModels->GameModelList["canvas"].vbos[0], app.width(),
                         app.height(), "Warped Face");
      auto smallHeight = app.height() / 4,
           smallWidth = smallHeight / Reader.get_color_height() *
                        Reader.get_color_width();
      rect middleR = {app.width() / 2 - smallWidth / 2, 0, smallWidth,
                      smallHeight};
      model_image.render(middleR, gModels->GameModelList["canvas"].width,
                         gModels->GameModelList["canvas"].height,
#if DELAY_ONE_FRAME
                         gModels->GameModelList["canvas"]
                             .vbos[mainSolution.prev_color_map_handle_id()],
#else
                         gModels->GameModelList["canvas"]
                             .vbos[mainSolution.input_color_map_handle_id()],
#endif
                         app.width(), app.height(), "Color");
#if SAVE_IMAGE
      if (!status.isRetracking) {
        saveRenderImage("results/nonRigidRender/" +
                        std::to_string(mainSolution.image_idx_) + ".png");
      }
#endif
#if OUTPUT_EXPERIMENT
      if (!status.isRetracking) {
#if DELAY_ONE_FRAME
        int idx = mainSolution.image_idx_ - 1;
#else
        int idx = mainSolution.image_idx;
#endif
      }
#endif
    }
#if PROFILE_ONLINE
    timer.report("render");
#endif

    ImGui::Render();
    int display_w, display_h;
    glfwMakeContextCurrent(app);
    glfwGetFramebufferSize(app, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

#if PROFILE_ONLINE
    timer.report("GL");
#endif
  }
  google::ShutdownGoogleLogging();
  delete colorIntrinsic;
  return EXIT_SUCCESS;
}

void GenInit() {
  int imgWidth = Reader.get_color_width();
  int imgHeight = Reader.get_color_height();

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glClearDepth(1);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  CurrentFace.request_tex_coord_ = true;
  CurrentFace.load_obj(MORPH_DEFAULT_INTEL_TEMPLATE_MODEL);

  mainSolution.updateMesh();
  gModels->createMesh("dynamicFace_MRT", CurrentFace);
  gModels->createImage("canvas", imgWidth, imgHeight);
  gModels->createImage("canvas2", imgWidth * 2, imgHeight * 2);
  gModels->createMesh("avatar", avatar->mesh);

  ShaderLoader shaderLoader_face;
  program = shaderLoader_face.CreateProgram(
      "./shaders/face_vertex_shader.glsl",
      "./shaders/face_fragment_shader_sh_viewer.glsl");
  ShaderLoader shaderLoader_isomap;
  programIsomap = shaderLoader_isomap.CreateProgram(
      "./shaders/isomap_vertex_shader.glsl",
      "./shaders/isomap_fragment_shader.glsl");
  ShaderLoader shaderLoader_isomapfromcolor;
  programIsomapfromcolor = shaderLoader_isomapfromcolor.CreateProgram(
      "./shaders/isomapfromcolor_vertex_shader.glsl",
      "./shaders/isomapfromcolor_fragment_shader.glsl");
  ShaderLoader shaderLoader_testTriangle;
  programTriangle = shaderLoader_testTriangle.CreateProgram(
      "./shaders/TestTriangle_vertex.glsl",
      "./shaders/TestTriangle_fragment.glsl");
}

/// util
void int2str(const int &int_temp, std::string &string_temp) {
  std::stringstream stream;
  stream << int_temp;
  string_temp = stream.str();
}

void optimizingThread() {
  static LambdaSetting &lambdas = LambdaSetting::get_instance();
  const float icp_lambda = lambdas("neutral", "lambda_icp");
  const float lm_lambda = lambdas("neutral", "lambda_lm");
  const float l2_lambda = lambdas("neutral", "lambda_l2");
  const float pose_lm_lambda = lambdas("neutral", "lambda_pose_lm");

  int iter_num_1_image = 4;
  {
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

#if PROFILE_ONLINE
    SimpleTimer timer;
    timer.init();
#endif
    mainSolution.setImageIndex(mainSolution.image_idx_ + 1);
#if PROFILE_ONLINE
    timer.report("set image index");
#endif

    float threshold = 2e-2f;
    for (int iter = 0; iter < iter_num_1_image; ++iter) {
      if (threshold > 5e-3f) {
        threshold = threshold - (threshold - 0.8f) * 0.8f;
      }
      if (mainSolution.image_idx_ == 3) {
        lambdaSetting.use_ICP = true;
      }
      int res = (status.iterCnt - 28) % iter_num_1_image;
      status.isThreadWorking = true;
      {
        {
          mainSolution.mesh_->clear_lambda();
          if (lambdaSetting.use_ICP) {
            mainSolution.pushLandmarkTargetVertices(pose_lm_lambda, true);
            mainSolution.pushContourLandmarkTargetVertices(pose_lm_lambda,
                                                           true);
          } else {
            mainSolution.pushLandmarkTargetVertices(pose_lm_lambda);
          }
          mainSolution.solveRTwithNormal();
        }
        if (mainSolution.image_idx_ >= 2 && iter >= 2 &&
            !status.isStartTracking) {
          mainSolution.mesh_->clear_lambda();
          mainSolution.pushLandmarkTargetVertices(
              2.0f * lambdaSetting.LandmarkLambda, true);
          mainSolution.pushContourLandmarkTargetVertices(
              0.5f * lambdaSetting.LandmarkLambda);
          mainSolution.prepareTarget();
          mainSolution.solveIDCoefficient(l2_lambda);
        }
      }
      status.isThreadWorking = false;
      status.iterCnt++;
      mainSolution.updateContourLandmarkIndex();
    }
    if (!mainSolution.lock_image_index) {
      mainSolution.parameters_->push_present_RT();
    }
    cudaStreamSynchronize(0);
    QueryPerformanceCounter(&end);
    status.fpsOPT =
        status.fpsOPT * 0.6 +
        0.4 * (double)freq.QuadPart / (double)(end.QuadPart - start.QuadPart);
  }
}

void nonRigidThread() {
  static LambdaSetting &lambdas = LambdaSetting::get_instance();
  const int model_iters = lambdas("nonrigid", "model_iters") + 0.1;
  const float icp_lambda = lambdas("nonrigid", "lambda_icp");
  const float pose_icp_lambda = lambdas("nonrigid", "lambda_pose_icp");

  const float model_lm_lambda = lambdas("nonrigid", "lambda_lm");
  const float contour_lm_lambda = lambdas("nonrigid", "labmda_contour_lm");
  const float exp_smooth = lambdas("nonrigid", "exp_smooth");
  const float l1_lambda = lambdas("nonrigid", "lambda_l1");
  const float eye_lambda = lambdas("nonrigid", "eye_lambda");
  const float eyebrow_lambda = lambdas("nonrigid", "eyebrow_lambda");
  mainSolution.parameters_->lambda_1 = lambdas("nonrigid", "pose_smooth_1");
  mainSolution.parameters_->lambda_2 = lambdas("nonrigid", "pose_smooth_2");
  mainSolution.parameters_->lambda_3 = lambdas("nonrigid", "pose_smooth_3");

  CUDATimer timer;
  {
    timer.startEvent("set image index");
    mainSolution.setImageIndex(mainSolution.image_idx_ + 1);
    timer.endEvent();
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    float threshold = 2e-2f;
    timer.startEvent("iteration");
    mainSolution.updateExpMesh();
    for (int iter = 0; iter < model_iters; ++iter) {
      threshold =
          threshold - 0.5f * (threshold - 3e-3f);  /// reduce the threhold
      mainSolution.updateSRTPositionNormal();
      mainSolution.mesh_->clear_lambda();
      mainSolution.pushNoseLandmarkTargetVertices(model_lm_lambda);
      if (iter >= 2) {
        mainSolution.pushContourLandmarkTargetVertices(contour_lm_lambda);
      }
      mainSolution.solveRTwithNormal();

      mainSolution.mesh_->clear_lambda();
      mainSolution.pushExpLandmarkTargetVertices(model_lm_lambda, eye_lambda,
                                                 eyebrow_lambda);
      mainSolution.pushContourLandmarkTargetVertices(contour_lm_lambda);

      mainSolution.prepareTarget();
      mainSolution.solveSampledSparseCoefficient(
          0.0 * lambdaSetting.lambda, exp_smooth, l1_lambda);
      mainSolution.updateContourLandmarkIndex();
    }
    timer.endEvent();

    mainSolution.updateExpMesh();
    if (!mainSolution.lock_image_index) {
      mainSolution.parameters_->push_present_RT();
      mainSolution.parameters_->modelParams.cuda_exp_prev_1_.swap(
          mainSolution.parameters_->modelParams.cuda_exp_prev_2_);
      mainSolution.parameters_->modelParams.cuda_exp_coefficients_.copyTo(
          mainSolution.parameters_->modelParams.cuda_exp_prev_1_);

      std::vector<float> host_rotation, host_translation;
      mainSolution.parameters_->cuda_rotation_.download(host_rotation);
      mainSolution.parameters_->cuda_translation_.download(host_translation);
      for (auto iter : host_translation) {
        host_rotation.push_back(iter);
      }
      mainSolution.sRT_list.push_back({mainSolution.image_idx_, host_rotation});
    }

    QueryPerformanceCounter(&end);
    {
      mainSolution.updateSRTPositionNormal();
      gModels->updateMesh("dynamicFace_MRT", CurrentFace.cuda_position_sRT,
                          CurrentFace.cuda_color_, CurrentFace.cuda_normal_R,
                          CurrentFace.cuda_tri_list_);
      model_image.renderModel(*gModels, "canvas", "dynamicFace_MRT", rInfo,
                              program);
    }

    mainSolution.updateSRTPositionNormal();
    mainSolution.updateModel();
    model_image.renderModel(*gModels, mainSolution.large_canvas_name(),
                            "dynamicFace_MRT", rInfo, program);
    mainSolution.save_color_map(
        status.sequence_folder + "/retracking_blendshapes_" +
            status.getBlendShapeLevelLiteral() +
            mainSolution.save_folder_suffix + "/" +
            std::to_string(mainSolution.image_idx_) + ".png",
        mainSolution.large_canvas_name());

    avatar->updateblendshapes(  // fixme
        mainSolution.parameters_->modelParams.cuda_exp_coefficients_);
    avatar->rigid_transform(*mainSolution.parameters_);

    gModels->updateMesh("avatar", avatar->position_sRT,
                        avatar->mesh.cuda_color_, avatar->normal_R,
                        avatar->mesh.cuda_tri_list_);
    model_image.renderModel(*gModels, mainSolution.large_canvas_name(),
                            "avatar", rInfo, program);
  }
}

