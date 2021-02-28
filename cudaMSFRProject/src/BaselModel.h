#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen\Eigen>
#include <memory>
#include <string>
#include <vector>

#include "Common.h"
#include "GraphUtil.h"
#include "ObjMesh.h"

void cudaProductMatVector(pcl::gpu::DeviceArray<float3> x,
                          const pcl::gpu::DeviceArray<float> A,
                          const pcl::gpu::DeviceArray<float> b,
                          const pcl::gpu::DeviceArray<float3> c);
void cudaProductMatVectorII(pcl::gpu::DeviceArray<float3> x,
                            const pcl::gpu::DeviceArray<float> A,
                            const pcl::gpu::DeviceArray<float> b,
                            const pcl::gpu::DeviceArray<float3> c);
void cudaProductMatVector_Corr(pcl::gpu::DeviceArray<float3> x,
                               const pcl::gpu::DeviceArray<float> A,
                               const pcl::gpu::DeviceArray<float> b,
                               const pcl::gpu::DeviceArray<float3> c);

/// Basel model class
class BaselModel {
 public:
  BaselModel() { LoadModel(); }
  bool LoadModel(std::string filename = MORPH_DEFAULT_INTEL_MODEL_PATH);
  bool UpdateModel(Eigen::VectorXf identity_coef, Eigen::VectorXf color_coef,
                   Eigen::VectorXf expr_coef, Mesh &mesh_model);
  bool RandomFace(Mesh &mesh_model);

  /// Data access functions
  float mShape(int idx) { return params_[mu_shape()](idx, 0); }
  float mShapePCA(int idx) { return mu_shapePCA_(idx, 0); }
  float mShapeValue(int idx) {
    return isUseExprPCA ? mu_shapePCA_(idx, 0) : params_[mu_shape()](idx, 0);
  }
  float mColor(int idx) { return params_[mu_color()](idx, 0); }

  float pcaShape(int idx, int idx_pca) {
    return params_[pca_shape()](idx, idx_pca);
  }
  float pcaShapeUn(int idx, int idx_pca) {
    return params_[pca_shape_unnorm()](idx, idx_pca);
  }
  float pcaColor(int idx, int idx_pca) {
    return params_[pca_color()](idx, idx_pca);
  }
  float pcaExpr(int idx, int idx_pca) {
    return params_[pca_exp()](idx, idx_pca);
  }
  float pcaExprUn(int idx, int idx_pca) {
    return params_[pca_exp_unnorm()](idx, idx_pca);
  }
  float intelExpr(int idx, int idx_expr) {
    return params_[expr_disp_intel()](idx, idx_expr);
  }
  float valueExpr(int idx, int idx_expr) {
    return isUseExprPCA ? pcaExpr(idx, idx_expr) : intelExpr(idx, idx_expr);
  }

  float sigmaExpr(int idx_expr) { return params_[sigma_exp()](idx_expr); }
  float sigmaShape(int idx_shape) { return params_[sigma_shape()](idx_shape); }
  float sigmaColor(int idx_color) { return params_[sigma_color()](idx_color); }

  float curShape(int idx) { return cur_shape_(idx, 0); }
  float curShapeId(int idx) { return cur_shapeId_(idx, 0); }
  float curShapeExpr(int idx) { return cur_shapeExpr_(idx, 0); }
  float curColor(int idx) { return cur_color_(idx, 0); }

  int numIdParams() { return dim_id_; }
  int numPcaExprParams() { return dim_exp_; }
  int numIntelExprParams() { return dim_intel_; }
  int numAlbParams() { return dim_color_; }

  Eigen::MatrixXf &GetFaceMask() { return params_[mask_big()]; }

  int numStrip() { return params_[num_strip()].rows(); }
  int numStrip0() { return params_[num_strip0()].rows(); }

  int numStrip0ByIndex(int idx) { return params_[num_strip0()](idx, 0); }
  int indexStrip0ByIndex(int idx) { return params_[index_strip0()](idx, 0); }

  int numStripByIndex(int idx) { return params_[num_strip()](idx, 0); }
  int indexStripByIndex(int idx) { return params_[index_strip()](idx, 0); }

  float segByIndex(int idx) { return params_[mask_seg4()](idx); }

  Eigen::MatrixXf getSymList() { return params_[symlist()]; }

  std::vector<Eigen::MatrixXf> params_;
  Eigen::MatrixXf cur_shape_;
  Eigen::MatrixXf cur_color_;
  Eigen::MatrixXf cur_colorOld_;
  Eigen::MatrixXf cur_shapeId_;
  Eigen::MatrixXf cur_shapeExpr_;
  Eigen::MatrixXf mu_shapePCA_;

  Eigen::MatrixXf strip_start;
  Eigen::MatrixXf strip_end;

  int dim_intel_;
  int dim_id_;
  int dim_color_;
  int dim_exp_;
  int idx_exp_;
  int vertexNum;

  bool isUseExprPCA;

 public:
  int mu_shape() { return 0; }
  int pca_shape() { return 1; }
  int sigma_shape() { return 2; }
  int mu_exp() { return 3; }
  int pca_exp() { return 4; }
  int sigma_exp() { return 5; }
  int mu_color() { return 6; }
  int pca_color() { return 7; }
  int sigma_color() { return 8; }
  int expr_disp_intel() { return 9; }
  int tri() { return 10; }
  int pca_shape_unnorm() { return 11; }
  int pca_color_unnorm() { return 12; }
  int pca_exp_unnorm() { return 13; }
  int face_contour_front() { return 14; }
  int face_contour_front_line() { return 15; }
  int index_strip() { return 16; }
  int index_strip0() { return 17; }
  int index_strip_contour() { return 18; }
  int num_strip() { return 19; }
  int num_strip0() { return 20; }
  int num_strip_contour() { return 21; }
  int mask_big() { return 22; }
  int mask_left() { return 23; }
  int mask_seg10() { return 24; }
  int mask_seg4() { return 25; }
  int mask_seg8() { return 26; }
  int mask_small() { return 27; }
  int mask_zhu() { return 28; }
  int symlist() { return 29; }
};

#ifdef USE_CUDA

class cudaBaselModel : public BaselModel {
 public:
  cudaBaselModel()
      : id_base_(params_[pca_shape()].size()),
        exp_base_(params_[expr_disp_intel()].size()),
        color_base_(params_[pca_color()].size()),
        cuda_cur_shape_(vertexNum),
        cuda_mean_shape_(vertexNum),
        cuda_cur_color_(vertexNum),
        cuda_mean_color_(vertexNum) {
    /// Read Rigid Index
    {
      FILE *fp = fopen(MORPH_DEFAULT_RIGID_INDEX, "r");
      int rigid_cnt;
      fscanf(fp, "%d", &rigid_cnt);
      rigid_index_.resize(rigid_cnt, -1);
      for (int i = 0; i < rigid_cnt; ++i) {
        fscanf(fp, "%uh", &rigid_index_[i]);
      }
      fclose(fp);
      cuda_rigid_index_.upload(rigid_index_);
    }
    /// Estimate Rigid Transformation
    Eigen::Vector3f rigid_transformation, exp_mean_position, neu_mean_position;
    rigid_transformation.setZero();
    Eigen::Matrix3f R;
    {
      neu_mean_position.setZero();
      for (int i = 0; i < rigid_index_.size(); ++i) {
        neu_mean_position +=
            params_[mu_shape()].block(rigid_index_[i] * 3, 0, 3, 1);
      }
      neu_mean_position /= rigid_index_.size();
      for (int k = 0; k < dim_exp_; ++k) {
        exp_mean_position.setZero();
        for (int i = 0; i < rigid_index_.size(); ++i) {
          exp_mean_position +=
              params_[expr_disp_intel()].block(rigid_index_[i] * 3, k, 3, 1);
        }
        exp_mean_position /= rigid_index_.size();
        exp_mean_position += neu_mean_position;
        R.setZero();
        for (int i = 0; i < rigid_index_.size(); ++i) {
          const int rigid_index_i = rigid_index_[i];
          R += (params_[expr_disp_intel()].block(rigid_index_[i] * 3, k, 3, 1) +
                params_[mu_shape()].block(rigid_index_[i] * 3, 0, 3, 1) -
                exp_mean_position) *
               (params_[mu_shape()].block(rigid_index_[i] * 3, 0, 3, 1) -
                neu_mean_position)
                   .transpose();
        }
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(
            R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixV() * svd.matrixU().transpose();
        rigid_transformation.setZero();
        for (auto &iter : rigid_index_) {
          rigid_transformation +=
              params_[mu_shape()].block(iter * 3, 0, 3, 1) -
              R * (params_[expr_disp_intel()].block(iter * 3, k, 3, 1) +
                   params_[mu_shape()].block(iter * 3, 0, 3, 1));
        }
        rigid_transformation /= rigid_index_.size();
        //#pragma omp parallel for
        for (int i = vertexNum - 1; i >= 0; --i) {
          params_[expr_disp_intel()].block(i * 3, k, 3, 1) =
              R * (params_[expr_disp_intel()].block(i * 3, k, 3, 1) +
                   params_[mu_shape()].block(i * 3, 0, 3, 1)) +
              rigid_transformation - params_[mu_shape()].block(i * 3, 0, 3, 1);
        }
      }
    }
    params_[pca_shape()].transposeInPlace();
    params_[expr_disp_intel()].transposeInPlace();
    params_[pca_color()].transposeInPlace();
    params_[sigma_color()].transposeInPlace();
    id_base_.upload(params_[pca_shape()].data(), params_[pca_shape()].size());
    exp_base_.upload(params_[expr_disp_intel()].data(),
                     params_[expr_disp_intel()].size());
    exp_base_.copyTo(personalized_exp_base_);
    cuda_mean_shape_.upload(
        reinterpret_cast<const float3 *>(params_[mu_shape()].data()),
        vertexNum);
    cuda_mean_shape_.copyTo(cuda_personalized_mean_shape_);
    is_use_personalized_model = false;

    color_base_.upload(params_[pca_color()].data(),
                       params_[pca_color()].size());
    cuda_mean_color_.upload(
        reinterpret_cast<const float3 *>(params_[mu_color()].data()),
        vertexNum);
    cuda_mean_color_.copyTo(cuda_cur_color_);
    cuda_sigma_color.upload(params_[sigma_color()].data(),
                            params_[sigma_color()].size());

    params_[pca_shape()].transposeInPlace();
    params_[expr_disp_intel()].transposeInPlace();
    std::vector<int> host_symlist(vertexNum, -1);
    auto symlist = getSymList();
    for (int i = 0; i < symlist.rows(); ++i) {
      int x = floor(symlist(i, 0) + 0.1);
      int y = floor(symlist(i, 1) + 0.1);
      host_symlist[x] = y;
      host_symlist[y] = x;
    }
    cuda_symlist_.upload(host_symlist);  // prevbug: download or upload?
    params_[pca_color()].transposeInPlace();
    params_[sigma_color()].transposeInPlace();
    std::vector<float> host_reg_id(dim_id_), host_reg_exp(dim_exp_);
    for (int i = 0; i < dim_id_; ++i) {
      host_reg_id[i] = params_[pca_shape()].col(i).norm();
    }
    for (int i = 0; i < dim_exp_; ++i) {
      host_reg_exp[i] = params_[expr_disp_intel()].col(i).norm();
    }
    reg_id_.upload(host_reg_id);
    reg_exp_.upload(host_reg_exp);
    reg_personalized_exp_.create(dim_exp_);

    {
      std::vector<float> host_mouth(vertexNum, 0.0f);
      FILE *fp = fopen("./mouthIndex_1.txt", "r");
      int index_num = 0;
      fscanf(fp, "%d", &index_num);
      for (int i = 0; i < index_num; ++i) {
        int id;
        fscanf(fp, "%d", &id);
        host_mouth[id] = 1.0f;
      }
      fclose(fp);
      cuda_mouth_.upload(host_mouth);

      FILE *fp_jaw = fopen("./jawIndex.txt", "r");
      fscanf(fp_jaw, "%d", &index_num);
      for (int i = 0; i < index_num; ++i) {
        int id;
        fscanf(fp_jaw, "%d", &id);
        host_mouth[id] = 1.0f;
        params_[mask_seg4()](id) = 4.0f;
      }
      fclose(fp_jaw);

      // FILE *fp_forehead = fopen("./foreheadIndex.txt", "r");
      // fscanf(fp_forehead, "%d", &index_num);
      // for (int i = 0; i < index_num; ++i) {
      //  int id;
      //  fscanf(fp_forehead, "%d", &id);
      //  params_[mask_seg4()](id) = 5.0f;
      //}
      // fclose(fp_forehead);
      cuda_mouth_and_jaw_.upload(host_mouth);

      auto fp_innermouth = fopen("./data/model/innermouth.txt", "r");
      int innermouth_cnt = 0;
      fscanf(fp_innermouth, "%d", &innermouth_cnt);
      // std::cout << innermouth_cnt << std::endl;
      for (int i = 0; i < innermouth_cnt; ++i) {
        int idx;
        fscanf(fp_innermouth, "%d", &idx);
        params_[mask_seg4()](idx) = 6.0f;
      }
      fclose(fp_innermouth);
    }
    cuda_mask_seg_.upload(params_[mask_seg4()].data(), vertexNum);
    std::vector<float> host_blendshape_mask(exp_base_.size(), 0.0f);
    // for (int i = 0; i < dim_exp_; ++i) {
    //  FILE *fp = fopen(
    //      ("./data/model/mask/" + std::to_string(i) + ".txt").c_str(), "r");
    //  int mask_index_num;
    //  fscanf(fp, "%d", &mask_index_num);
    //  for (int j = 0; j < mask_index_num; ++j) {
    //    int index;
    //    fscanf(fp, "%d", &index);
    //    host_blendshape_mask[index * 3 * dim_exp_ + i] = 1.0f;
    //    host_blendshape_mask[(index * 3 + 1) * dim_exp_ + i] = 1.0f;
    //    host_blendshape_mask[(index * 3 + 2) * dim_exp_ + i] = 1.0f;
    //  }
    //  fclose(fp);
    //}
    cuda_blendshape_mask_.upload(host_blendshape_mask);
    exp_base_.copyTo(cuda_blendshape_mask_);
    /// load contour landmark Index
    std::vector<int2> host_strip_begin_end(17);
    std::vector<int> host_strip_index(params_[index_strip()].size());
    for (int i = 0; i < 17; ++i) {
      host_strip_begin_end[i].x = strip_start(i, 0);
      host_strip_begin_end[i].y = strip_end(i, 0);
    }
    for (int i = 0; i < host_strip_index.size(); ++i) {
      host_strip_index[i] = indexStripByIndex(i);
    }
    cuda_strip_begin_end_.upload(host_strip_begin_end);
    cuda_strip_index_.upload(host_strip_index);
#ifdef USE_FACESHIFT_EXP
    LoadFaceShift();
#endif
  }
  void cudaUpdateModelID(const pcl::gpu::DeviceArray<float> &id_coefficients);
  void cudaUpdateModelEXP(const pcl::gpu::DeviceArray<float> &exp_coefficients);
  void cudaUpdateModelColor(
      const pcl::gpu::DeviceArray<float> &color_coefficients) const;
  void cudaUpdateModel(const pcl::gpu::DeviceArray<float> &id_coefficients,
                       const pcl::gpu::DeviceArray<float> &exp_coefficients);
  void cudaUpdatePersonalizedModel(
      const pcl::gpu::DeviceArray<float> exp_coefficients);
  void cudaUpdateStaticPersonalizedModel(
      const pcl::gpu::DeviceArray<float> exp_coefficients);
  void saveExpBlendshapeModel(ObjMesh &obj, const std::string &filePath,
                              const int idx);
  void updatePersonalizedCorrelationReg();
  void updateExpBaseCorrelationReg(pcl::gpu::DeviceArray<float> ATA,
                                   pcl::gpu::DeviceArray<float> reg,
                                   const pcl::gpu::DeviceArray<float> base);

  void UpdatePersonalizedExpBase(const Mesh &personalized_mean_shape_);
  void updateLaplacianSmoothMask(ObjMesh &obj);
  void RefineBlendshape(pcl::gpu::DeviceArray<float3> mean_shape,
                        pcl::gpu::DeviceArray<float> exp_base,
                        cudaStream_t stream);

 public:
  pcl::gpu::DeviceArray<float> id_base_;
  pcl::gpu::DeviceArray<float> exp_base_;
  pcl::gpu::DeviceArray<float> color_base_;
  pcl::gpu::DeviceArray<float> reg_id_;
  pcl::gpu::DeviceArray<float> reg_exp_;
  pcl::gpu::DeviceArray<float> reg_personalized_exp_;
  pcl::gpu::DeviceArray<float> cuda_sigma_color;
  pcl::gpu::DeviceArray<float> personalized_exp_base_, personalized_ATA_,
      personalized_exp_base_static_, personalized_exp_base_corr_;
  pcl::gpu::DeviceArray<float> cuda_mask_seg_;
  pcl::gpu::DeviceArray<float> cuda_mouth_and_jaw_, cuda_mouth_;
  pcl::gpu::DeviceArray<float3> cuda_cur_shape_;
  pcl::gpu::DeviceArray<float3> cuda_mean_shape_;
  pcl::gpu::DeviceArray<float3> cuda_cur_color_;
  pcl::gpu::DeviceArray<float3> cuda_mean_color_;
  pcl::gpu::DeviceArray<float3> cuda_personalized_mean_shape_;
  pcl::gpu::DeviceArray<int2> cuda_strip_begin_end_;
  pcl::gpu::DeviceArray<int> cuda_strip_index_;
  Eigen::MatrixXf personalized_mean_shape_;
  pcl::gpu::DeviceArray<unsigned short> cuda_rigid_index_;
  pcl::gpu::DeviceArray<float> cuda_blendshape_mask_;
  pcl::gpu::DeviceArray<int> cuda_symlist_;
  std::vector<unsigned short> rigid_index_;
  bool is_use_personalized_model, is_personalized_model_generated = false;

#ifdef USE_FACESHIFT_EXP
  pcl::gpu::DeviceArray<float3> cuda_loaded_mean_shape_;
  pcl::gpu::DeviceArray<float> loaded_exp_base_;
  Eigen::MatrixXf host_faceshift_exp_base;
  void LoadFaceShift();
#endif

};


#endif  // USE_CUDA