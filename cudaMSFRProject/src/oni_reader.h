#ifndef _ONI_READER_H_
#define _ONI_READER_H_

#include <librealsense2/rsutil.h>
#include <pcl/gpu/containers/device_array.h> /*used with device type*/
#include <vector_types.h>

#include <hpp\rs_frame.hpp>
#include <librealsense2/rs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>

class OniReader {
 public:
  enum Read_mode { ONLINE_MODE, OFFLINE_MODE };

  OniReader(Read_mode read_mode = OniReader::OFFLINE_MODE)
      : /* m_pb_ctrl(nullptr),*/ m_c_width(0),
        m_c_height(0),
        m_d_width(0),
        m_d_height(0),
        m_c_fps(0),
        m_d_fps(0),
        m_c_frame_num(0),
        m_d_frame_num(0),
        read_from_device(0),
        read_from_oni(0),
        read_mode_(read_mode) {
    init();
  }
  ~OniReader();

  // noncopyable
  OniReader(const OniReader &) = delete;
  OniReader &operator=(const OniReader &) = delete;

  void init();
  void open(const std::string &_color_dir, int image_width, int image_height);

  cv::Mat get_color_frame(int _frame_idx); /*BGR order*/

  // get one frame in DeviceArray2D
  pcl::gpu::DeviceArray2D<float4> get_color_frame_device(int _frame_idx);
  void mock_cur_frame(int _frame_idx);

  int get_color_frame_num() { return m_c_frame_num; }
  std::pair<int, int> get_color_res() { return {m_c_width, m_c_height}; }
  int get_color_fps() { return m_c_fps; }
  int get_color_width() { return m_c_width; }
  int get_color_height() { return m_c_height; }
  std::string OniReader::get_landmark_filename(int _frame_idx);
  bool read_from_oni;
  bool read_from_device;

  rs2_intrinsics get_color_intrinsics() const { return color_intrinsics_; }
  void save_this_frame(const std::string file_name);

  void saveFrame(const std::string &prefix, int imageIdx) const;
  const std::string source_dir() const { return m_source; }
  const Read_mode get_read_mode() const { return read_mode_; }

 private:
  typedef std::tuple<int, uint64_t, int, uint64_t> FrameInfo;
  enum FrameInfoIndex { ColorIndex, ColorTime };
  Read_mode read_mode_;
  std::vector<FrameInfo> m_frameInfoList;
  static std::string s_recordLogFileName;

  int m_c_width;
  int m_c_height;
  int m_d_width;
  int m_d_height;
  int m_c_fps;
  int m_d_fps;
  int m_c_frame_num;
  int m_d_frame_num;

  std::string m_source;
  std::string m_color_source;
  std::string m_landmark_source;

  // staging buffer for repacking rgba data on device
  pcl::gpu::DeviceArray<uchar3> m_rgb_buffer;

  // page locked memory for color processing
  void *m_color_page_lock;

  int cur_frame_idx = -1;
  cv::Mat cur_color_frame;
  rs2::pipeline pipe;
  rs2::config config;
  rs2::pipeline_profile profile;
  rs2_intrinsics color_intrinsics_;

  pcl::gpu::DeviceArray2D<uchar3> color_frame_device;
  pcl::gpu::DeviceArray2D<float4> color_frame_device_float;

  void get_new_frame(const int frame_idx);
};

#endif
