#pragma once
#include <LandmarkUtil.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <vector>
#include "pcl\gpu\utils\safe_call.hpp"
#include "SmoothFilter.h"

constexpr int FACIAL_POINTS_NUM_FLT = 68;


class LandmarkDetector : public LandmarkUtil {
 public:
  LandmarkDetector();
  ~LandmarkDetector();

  void startDetector();
  void closeDetector();
  void detectLandmark(char *imgName);
  void readLandmarksFromFile(std::string &landmark_file);
  void showLandmark();
  void showLandmarkCorrespondence(
      cv::Mat img, const pcl::gpu::DeviceArray<float3> position_sRT,
      const pcl::gpu::DeviceArray<unsigned short> is_front,
      const msfr::intrinsics camera);
  void saveLandmarkCorrespondence(
      cv::Mat img, const pcl::gpu::DeviceArray<float3> position_sRT,
      const pcl::gpu::DeviceArray<unsigned short> is_front,
      const msfr::intrinsics camera, const std::string file_name);
  void getLandmarkPos(const msfr::intrinsics &camera);
  void SaveLandmarkError(std::string file_name, const msfr::intrinsics camera,
                         const pcl::gpu::DeviceArray<float3> position_sRT);

  float getLandMarkPrevDifference();
  cv::Mat& get_cropped_image() { return cur_cropped_img; }

 private:
  void setLandmarkDeviceUV();

 public:
  std::vector<float2> landmark_;
  std::vector<float3> landmark_position_, contour_landmark_position_;
  pcl::gpu::DeviceArray<int2> cuda_landmark_index_;
  pcl::gpu::DeviceArray<float2> cuda_landmark_uv_;
  pcl::gpu::DeviceArray<float2> cuda_pre_landmark_uv_;
  pcl::gpu::DeviceArray<float2> cuda_pre_contour_landmark_uv_;
  pcl::gpu::DeviceArray<float3> cuda_landmark_position_;
  pcl::gpu::DeviceArray<int2> cuda_contour_landmark_index_;
  pcl::gpu::DeviceArray<float2> cuda_contour_landmark_uv_;
  pcl::gpu::DeviceArray<float3> cuda_contour_landmark_position_;
  std::vector<float2> landmark_uv_, pre_landmark_uv_, contour_landmark_uv_,
      pre_contour_landmark_uv_;
  std::vector<int2> new_landmarkIndex;

  int start_x, start_y, end_x, end_y;
  float smooth_lambda = 0.01;
  SmoothFilter contour_lm_filter, lm_filter;

 private:
  int max_face_num;
  IplImage *src_img;
  cv::Mat cur_cropped_img = cv::Mat(480, 640, CV_8UC3);
  int prevLandMarkIdx = -1;
};
