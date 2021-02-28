#include "LandmarkDetector.h"

#include <fstream>
#include <string>

#include "MSFRUtil.cu"

constexpr float MAX_LANDMARK_DISTANCE = 50000.0f;
constexpr int FRAME_INTERVAL = 10;

LandmarkDetector::LandmarkDetector()
    : LandmarkUtil(),
      cuda_landmark_index_(landmarkNum),
      cuda_landmark_uv_(landmarkNum),
      cuda_landmark_position_(landmarkNum),
      cuda_contour_landmark_index_(contourLandmarkNum),
      cuda_contour_landmark_uv_(contourLandmarkNum),
      cuda_contour_landmark_position_(contourLandmarkNum),
      landmark_(FACIAL_POINTS_NUM_FLT),
      landmark_uv_(landmarkNum),
      landmark_position_(landmarkNum),
      contour_landmark_uv_(contourLandmarkNum),
      contour_landmark_position_(contourLandmarkNum),
      pre_landmark_uv_(FACIAL_POINTS_NUM_FLT),
      lm_filter(landmarkNum),
      contour_lm_filter(contourLandmarkNum) {
  new_landmarkIndex.resize(landmarkNum);
  for (int i = 0; i < landmarkNum; ++i) {
    new_landmarkIndex[i] = landmarkIndex[i];
  }
  max_face_num = 1;
  cuda_landmark_index_.upload(new_landmarkIndex);

  /// init contour landmark index on device
  cuda_contour_landmark_index_.upload(contourLandmarkIndex);

  startDetector();
}

LandmarkDetector::~LandmarkDetector() { closeDetector(); }

void LandmarkDetector::startDetector() {}

void LandmarkDetector::closeDetector() {}

void LandmarkDetector::detectLandmark(char *imgName) {
  src_img = cvLoadImage(imgName, 1);
  setLandmarkDeviceUV();
}

void LandmarkDetector::readLandmarksFromFile(std::string &landmark_file) {
  for (int i = 0; i < FACIAL_POINTS_NUM_FLT; ++i) {
    pre_landmark_uv_[i].x = landmark_[i].x;
    pre_landmark_uv_[i].y = landmark_[i].y;
  }
  std::ifstream file;
  file.open(landmark_file);
  for (int i = 0; i < FACIAL_POINTS_NUM_FLT; ++i) {
    file >> landmark_[i].x >> landmark_[i].y;
  }
  setLandmarkDeviceUV();
}

void LandmarkDetector::showLandmark() {
  for (int j = 0; j < FACIAL_POINTS_NUM_FLT; j++) {
    cvCircle(src_img, cvPoint(landmark_[j].x, landmark_[j].y), 2,
             CV_RGB(0, 255, 0), -1);
  }
  cvShowImage("Landmarks", src_img);
  cvWaitKey(0);
}

void LandmarkDetector::getLandmarkPos(const msfr::intrinsics &camera) {
  for (int k = 0; k < landmark_uv_.size(); k++) {
    float3 index = {landmark_uv_[k].x, landmark_uv_[k].y, 1.0f};
    auto pos3 = unProjectedFromIndex(camera, index);
    landmark_position_[k] = {-pos3.x, -pos3.y, -pos3.z};
  }
  for (int k = 0; k < contour_landmark_uv_.size(); k++) {
    float3 index = {contour_landmark_uv_[k].x, contour_landmark_uv_[k].y, 1.0f};
    auto pos3 = unProjectedFromIndex(camera, index);
    contour_landmark_position_[k] = {-pos3.x, -pos3.y, -pos3.z};
  }
  cuda_landmark_uv_.copyTo(cuda_pre_landmark_uv_);
  cuda_contour_landmark_uv_.copyTo(cuda_pre_contour_landmark_uv_);
  cuda_landmark_position_.upload(landmark_position_);
  cuda_contour_landmark_position_.upload(contour_landmark_position_);
}

void LandmarkDetector::SaveLandmarkError(
    std::string file_name, const msfr::intrinsics camera,
    pcl::gpu::DeviceArray<float3> position_sRT) {
  std::vector<float3> host_position_sRT;
  position_sRT.download(host_position_sRT);
  std::ofstream file;
  file.open(file_name);
  file << landmarkIndex.size() << std::endl;
  for (int j = 0; j < landmarkIndex.size(); ++j) {
    auto uv = getProjectPos(camera, host_position_sRT[landmarkIndex[j].y]);
    uv.x -= landmark_uv_[j].x;
    uv.y -= landmark_uv_[j].y;
    file << sqrtf(uv.x * uv.x + uv.y * uv.y) << std::endl;
  }
  file.close();
}

float LandmarkDetector::getLandMarkPrevDifference() {
  auto res = 0.0f;
  const auto sqr = [](float x) { return x * x; };
  const auto condDist = [&](float a, float b) {
    if (a < 0.0f || b < 0.0f) {
      return 0.0f;
    }
    return sqr(a - b);
  };
  const auto clamp = [](float x, float l, float r) {
    if (l <= x && x < r) {
      return x;
    }
    return -1.0f;
  };
  for (auto i = 0; i < landmarkIndex.size(); i++) {
    res += condDist(clamp(pre_landmark_uv_[landmarkIndex[i].x].x, 0, 640),
                    landmark_uv_[i].x) +
           condDist(clamp(pre_landmark_uv_[landmarkIndex[i].x].y, 0, 480),
                    landmark_uv_[i].y);
  }
  return res;
}

void LandmarkDetector::showLandmarkCorrespondence(
    cv::Mat img, const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const msfr::intrinsics camera) {
  cudaSafeCall(cudaStreamSynchronize(0));
  std::vector<float3> host_position_sRT;
  position_sRT.download(host_position_sRT);
  src_img = (IplImage *)&IplImage(img);
  CvFont font;
  double hScale = 0.3;
  double vScale = 0.3;
  int lineWidth = 0.05;
  cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0,
             lineWidth);
  for (int j = 0; j < FACIAL_POINTS_NUM_FLT; j++) {
    cvCircle(src_img, cvPoint(pre_landmark_uv_[j].x, pre_landmark_uv_[j].y), 2,
             CV_RGB(0, 255, 0), -1);
    cvPutText(src_img, std::to_string(j).c_str(),
              cvPoint(pre_landmark_uv_[j].x, pre_landmark_uv_[j].y), &font,
              CV_RGB(0, 255, 0));
  }
  std::vector<float3> host_landmark_position;
  cuda_landmark_position_.download(host_landmark_position);
  {
    float mean_depth = 0.0f;
    float max_depth = 0.0f;
    float min_depth = 1.0f;
    int cnt = 0;
    for (auto &iter : host_landmark_position) {
      if (iter.z > 0.0f) {
        cnt++;
        mean_depth += iter.z;
        max_depth = max_depth < iter.z ? iter.z : max_depth;
        min_depth = min_depth < iter.z ? min_depth : iter.z;
      }
    }
    mean_depth /= cnt;
    float stv = 0.0f;
    for (auto &iter : host_landmark_position) {
      if (iter.z > 0.0f) {
        stv += (iter.z - mean_depth) * (iter.z - mean_depth);
      }
    }
    stv /= cnt;
    for (auto &iter : host_landmark_position) {
      if (iter.z > 0.0f) {
        float weight =
            (iter.z - mean_depth) * (iter.z - mean_depth) / cnt / stv;
        if (weight > 0.1f) {
          std::cout << "Outlier" << weight << std::endl;
        }
      }
    }
    std::cout << "landmarks' mean depth: " << mean_depth << " variance: " << stv
              << " max depth: " << max_depth << " min depth: " << min_depth
              << std::endl;
  }

  std::vector<unsigned short> host_is_front;
  std::vector<int2> host_contour_index;
  is_front.download(host_is_front);
  cuda_contour_landmark_index_.download(host_contour_index);
  for (int j = 0; j < new_landmarkIndex.size(); ++j) {
    auto uv = getProjectPos(camera, host_position_sRT[new_landmarkIndex[j].y]);

    if (uv.x > 0.0f && uv.x < src_img->width - 1 && uv.y > 0.0f &&
        uv.y < src_img->height - 1) {
      int landmarkId = new_landmarkIndex[j].x;

      if (landmarkId == 93 || landmarkId == 99 || landmarkId == 105 ||
          landmarkId == 109) {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(255, 0, 0), -1);
        cvPutText(src_img, std::to_string(landmarkId).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(255, 0, 0));
      } else if (host_landmark_position[j].z > 0.0f &&
                 host_is_front[new_landmarkIndex[j].y] == 1) {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(0, 0, 255), -1);
        cvPutText(src_img, std::to_string(landmarkId).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(0, 0, 255));
      } else {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(255, 0, 255), -1);
        cvPutText(src_img, std::to_string(landmarkId).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(255, 0, 255));
      }
    }
  }
  for (int j = 0; j < host_contour_index.size(); ++j) {
    auto uv = getProjectPos(camera, host_position_sRT[host_contour_index[j].y]);

    if (uv.x > 0.0f && uv.x < src_img->width - 1 && uv.y > 0.0f &&
        uv.y < src_img->height - 1) {
      {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(0, 0, 255), -1);
        cvPutText(src_img, std::to_string(host_contour_index[j].x).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(0, 0, 255));
      }
    }
  }
  cvShowImage("Landmarks", src_img);
  cvWaitKey(50);
}

void LandmarkDetector::saveLandmarkCorrespondence(
    cv::Mat img, const pcl::gpu::DeviceArray<float3> position_sRT,
    const pcl::gpu::DeviceArray<unsigned short> is_front,
    const msfr::intrinsics camera, const std::string file_name) {
  std::vector<float3> host_position_sRT;
  position_sRT.download(host_position_sRT);
  src_img = (IplImage *)&IplImage(img);
  const auto font = cvFont(30, 5);
  for (int j = 0; j < FACIAL_POINTS_NUM_FLT; j++) {
    cvCircle(src_img, cvPoint(pre_landmark_uv_[j].x, pre_landmark_uv_[j].y), 2,
             CV_RGB(0, 255, 0), -1);
    cvPutText(src_img, std::to_string(j).c_str(),
              cvPoint(pre_landmark_uv_[j].x, pre_landmark_uv_[j].y), &font,
              CV_RGB(0, 255, 0));
  }
  std::vector<float3> host_landmark_position;
  cuda_landmark_position_.download(host_landmark_position);
  std::vector<unsigned short> host_is_front;
  std::vector<int2> host_contour_index;
  is_front.download(host_is_front);
  cuda_contour_landmark_index_.download(host_contour_index);

  for (int j = 0; j < new_landmarkIndex.size(); ++j) {
    auto uv = getProjectPos(camera, host_position_sRT[new_landmarkIndex[j].y]);

    if (uv.x > 0.0f && uv.x < src_img->width - 1 && uv.y > 0.0f &&
        uv.y < src_img->height - 1) {
      int landmarkId = new_landmarkIndex[j].x;

      if (landmarkId == 93 || landmarkId == 99 || landmarkId == 105 ||
          landmarkId == 109) {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(255, 0, 0), -1);
        cvPutText(src_img, std::to_string(landmarkId).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(255, 0, 0));
      } else if (host_landmark_position[j].z > 0.0f &&
                 host_is_front[new_landmarkIndex[j].y] == 1) {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(0, 0, 255), -1);
        cvPutText(src_img, std::to_string(landmarkId).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(0, 0, 255));
      } else {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(255, 0, 255), -1);
        cvPutText(src_img, std::to_string(landmarkId).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(255, 0, 255));
      }
    }
  }

  for (int j = 0; j < host_contour_index.size(); ++j) {
    auto uv = getProjectPos(camera, host_position_sRT[host_contour_index[j].y]);

    if (uv.x > 0.0f && uv.x < src_img->width - 1 && uv.y > 0.0f &&
        uv.y < src_img->height - 1) {
      {
        cvCircle(src_img, cvPoint(uv.x, uv.y), 2, CV_RGB(0, 0, 255), -1);
        cvPutText(src_img, std::to_string(host_contour_index[j].x).c_str(),
                  cvPoint(uv.x, uv.y), &font, CV_RGB(0, 0, 255));
      }
    }
  }
  cvShowImage("Landmarks", src_img);
  cvSaveImage(file_name.c_str(), src_img);
  cvWaitKey(50);
}

void LandmarkDetector::setLandmarkDeviceUV() {
  for (int i = 0; i < landmarkNum; i++) {
    auto x = landmark_[landmarkIndex[i].x].x;
    auto y = landmark_[landmarkIndex[i].x].y;
    if (x >= 0 && x < 640 && y >= 0 && y < 480)  // todo
    {
      landmark_uv_[i].x = x;
      landmark_uv_[i].y = y;
    } else {
      landmark_uv_[i].x = -1.0f;
      landmark_uv_[i].y = -1.0f;
    }
  }
  lm_filter.smooth_data(landmark_uv_, smooth_lambda);
  cuda_landmark_uv_.upload(landmark_uv_);

  for (int i = 0; i < contourLandmarkNum; ++i) {
    auto x = landmark_[contourLandmarkIndex[i].x].x;
    auto y = landmark_[contourLandmarkIndex[i].x].y;
    if (x >= 0 && x < 640 && y >= 0 && y < 480) {
      contour_landmark_uv_[i].x = x;
      contour_landmark_uv_[i].y = y;
    } else {
      contour_landmark_uv_[i].x = -1.0f;
      contour_landmark_uv_[i].y = -1.0f;
    }
  }
  contour_lm_filter.smooth_data(contour_landmark_uv_, smooth_lambda);
  cuda_contour_landmark_uv_.upload(contour_landmark_uv_);
}
