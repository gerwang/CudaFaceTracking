#include "oni_reader.h"

#include <Windows.h>
#include <time.h>

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>

#include "pcl/gpu/utils/safe_call.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

void convertUchar2Float(pcl::gpu::DeviceArray2D<float4> output,
                        const pcl::gpu::DeviceArray2D<uchar3> input);

void convertUint2Uchar(pcl::gpu::DeviceArray2D<unsigned short> output,
                       const pcl::gpu::DeviceArray2D<unsigned int> input);

OniReader::~OniReader() {
  try {
    std::cout << "\nLuckily, openNi device has been successfully shut down!\n";

    // free page locked buffers
    cudaSafeCall(cudaFreeHost(m_color_page_lock));

  } catch (...) {
    std::cout << "\nException founded in OniReader's destructor\n";
  }
}

void OniReader::init() {
  /*if (openni::OpenNI::initialize() != openni::STATUS_OK)
  throw std::runtime_error("fatal error: openni cannot be initialized !");*/
  if (read_mode_ != OFFLINE_MODE) {
    config.enable_stream(RS2_STREAM_COLOR, 0, 640, 480, RS2_FORMAT_BGR8, 30);
    profile = pipe.start(config);

    cur_frame_idx = -1;

    auto spv = profile.get_streams();
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR);
    auto cvsp = color_stream.as<rs2::video_stream_profile>();
    color_intrinsics_ = cvsp.get_intrinsics();
    m_c_frame_num = 65536;
    m_d_frame_num = 65536;
    m_c_width = 640;
    m_c_height = 480;
    m_c_fps = 30;
    m_d_width = 640;
    m_d_height = 480;
    m_c_fps = 30;
    read_from_device = true;
  }
}

void OniReader::open(const std::string& _color_dir, int image_width,
                     int image_height) {
  if (read_mode_ == OFFLINE_MODE) {
    m_c_width = image_width;
    m_c_height = image_height;
    m_c_fps = 30;
    m_d_width = image_width;
    m_d_height = image_height;
    m_c_fps = 30;

    m_color_source = _color_dir;
    m_source = _color_dir.substr(0, _color_dir.rfind("/"));
    m_landmark_source =
        m_color_source.substr(0, m_color_source.find("color")) + "landmark";
    {
      std::ifstream t(_color_dir.substr(0, _color_dir.rfind('/')) +
                      "/camera.json");
      std::string str((std::istreambuf_iterator<char>(t)),
                      std::istreambuf_iterator<char>());
      rapidjson::Document document;
      document.Parse(str.c_str());
      {
        rapidjson::Value& color = document["color"];
        color_intrinsics_.fx = color["fx"].GetFloat();
        color_intrinsics_.fy = color["fy"].GetFloat();
        color_intrinsics_.ppx = color["cx"].GetFloat();
        color_intrinsics_.ppy = color["cy"].GetFloat();
        color_intrinsics_.width = color["width"].GetFloat();
        color_intrinsics_.height = color["height"].GetFloat();
      }
      {
        rapidjson::Value& frame_num = document["frame_num"];
        m_c_frame_num = m_d_frame_num = frame_num["total_frame_num"].GetInt();
      }
    }

    read_from_device = false;
  }
}

std::string OniReader::s_recordLogFileName;

// get one color frame in cv::Mat (BGR order)
cv::Mat OniReader::get_color_frame(int _frame_idx) {
  get_new_frame(_frame_idx);
  return cur_color_frame;
}

void OniReader::mock_cur_frame(int _frame_idx) { cur_frame_idx = _frame_idx; }

// get color frame in DeviceArray2D
pcl::gpu::DeviceArray2D<float4> OniReader::get_color_frame_device(
    int _frame_idx) {
  get_new_frame(_frame_idx);
  return color_frame_device_float;
}

std::string OniReader::get_landmark_filename(int _frame_idx) {
  char landmark_file[512] = {0};
  sprintf_s(landmark_file, "/%04d.txt", _frame_idx);
  return m_landmark_source + landmark_file;
}

void OniReader::get_new_frame(const int frame_idx) {
  if (frame_idx != cur_frame_idx) {
    if (read_mode_ == ONLINE_MODE) {
      rs2::frameset frameset = pipe.wait_for_frames();

      rs2::video_frame color_frame = frameset.get_color_frame();

      int img_height = color_frame.get_height();
      int img_width = color_frame.get_width();

      cur_color_frame = cv::Mat(img_height, img_width, CV_8UC3,
                                (uchar*)color_frame.get_data());

      if (color_frame_device.cols() != m_c_width &&
          color_frame_device.rows() != m_c_height) {
        color_frame_device.release();
        color_frame_device.create(m_c_height, m_c_width);
      }
      if (color_frame_device_float.cols() != m_c_width &&
          color_frame_device_float.rows() != m_c_height) {
        color_frame_device_float.release();
        color_frame_device_float.create(m_c_height, m_c_width);
      }
      color_frame_device.upload(cur_color_frame.data, 3 * cur_color_frame.cols,
                                cur_color_frame.rows, cur_color_frame.cols);
      convertUchar2Float(color_frame_device_float, color_frame_device);

      cur_frame_idx = frame_idx;
    } else if (read_mode_ == OFFLINE_MODE) {
      char color_file[512] = {0};
      sprintf_s(color_file, "/%04d.png", frame_idx);
      cur_color_frame =
          cv::imread(m_color_source + color_file, cv::IMREAD_UNCHANGED);  //

      if (color_frame_device.cols() != m_c_width &&
          color_frame_device.rows() != m_c_height) {
        color_frame_device.release();
        color_frame_device.create(m_c_height, m_c_width);
      }
      if (color_frame_device_float.cols() != m_c_width &&
          color_frame_device_float.rows() != m_c_height) {
        color_frame_device_float.release();
        color_frame_device_float.create(m_c_height, m_c_width);
      }
      color_frame_device.upload(cur_color_frame.data, 3 * cur_color_frame.cols,
                                cur_color_frame.rows, cur_color_frame.cols);
      convertUchar2Float(color_frame_device_float, color_frame_device);
    }
  }
}

void OniReader::save_this_frame(const std::string file_name) {}

void OniReader::saveFrame(const std::string& prefix, int imageIdx) const {
  char buffer[10];
  sprintf(buffer, "%04d", imageIdx);
  auto number = std::string(buffer);
  auto color_path = prefix + "/color/" + number + ".png";
  std::vector<unsigned> cpuBuffer(m_d_width * m_d_height);

  cv::imwrite(color_path, cur_color_frame);
}
