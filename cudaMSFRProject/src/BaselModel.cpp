#include "BaselModel.h"

#include <Windows.h>
#include <matlab/mat.h>
#include <time.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#define N_Params 32
char* Params_Name[N_Params] = {"mu_shape",
                               "pca_shape",
                               "sigma_shape",
                               "mu_exp",
                               "pca_exp",
                               "sigma_exp",
                               "mu_color",
                               "pca_color",
                               "sigma_color",
                               "expr_disp_intel",
                               "tri",
                               "pca_shape_unnorm",
                               "pca_color_unnorm",
                               "pca_exp_unnorm",
                               "face_contour_front",
                               "face_contour_front_line",
                               "index_strip",
                               "index_strip0",
                               "index_strip_contour",
                               "num_strip",
                               "num_strip0",
                               "num_strip_contour",
                               "mask_big",
                               "mask_left",
                               "mask_seg10",
                               "mask_seg4",
                               "mask_seg8",
                               "mask_small",
                               "mask_zhu",
                               "symlist",
                               "pca_lap",
                               "sigma_lap"};

void cudaInitV_(pcl::gpu::DeviceArray<float> v_,
                const pcl::gpu::DeviceArray<float3> position,
                const pcl::gpu::DeviceArray<int3> tri_list);
void cudaComputeRotation(
    pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<unsigned short> rigid_index);
void cudaComputeBlendshape(pcl::gpu::DeviceArray<float> exp_base,
                           const pcl::gpu::DeviceArray<float3> mean_shape,
                           cudaStream_t stream = 0);
void cudaComputeRotationRefine(pcl::gpu::DeviceArray<float> rotation,
                               const pcl::gpu::DeviceArray<float> exp_base,
                               const pcl::gpu::DeviceArray<float3> mean_shape,
                               cudaStream_t stream = 0);
void cudaComputeRotationRefineMask(
    pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<float> exp_mask, cudaStream_t stream = 0);
void cudaComputeTranslationRefine(
    pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape, cudaStream_t stream = 0);
void cudaComputeTranslationRefineMask(
    pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<float> exp_mask, cudaStream_t stream = 0);
void cudaComputeTranslation(
    pcl::gpu::DeviceArray<float> translation,
    const pcl::gpu::DeviceArray<float> rotation,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<unsigned short> rigid_index);
void cudaComputeSRTBase(pcl::gpu::DeviceArray<float> base,
                        const pcl::gpu::DeviceArray<float> rotation,
                        const pcl::gpu::DeviceArray<float> translation,
                        const pcl::gpu::DeviceArray<float3> mean_shape,
                        cudaStream_t stream = 0);

void cudaUpdateDimIdAtb(
    pcl::gpu::DeviceArray<float> Atb, const pcl::gpu::DeviceArray<float> vs_,
    const pcl::gpu::DeviceArray<float> vt_,
    const pcl::gpu::DeviceArray<float> exp_base,
    const pcl::gpu::DeviceArray<float3> mean_shape,
    const pcl::gpu::DeviceArray<float3> personalized_mean_shape,
    const pcl::gpu::DeviceArray<int3> tri_list,
    const pcl::gpu::DeviceArray<int2> fvLookUpTable,
    const pcl::gpu::DeviceArray<int1> fbegin,
    const pcl::gpu::DeviceArray<int1> fend, const int dimId);

void cudaCompuateATA_diag(pcl::gpu::DeviceArray<float> ATA,
                          pcl::gpu::DeviceArray<float> diag,
                          const pcl::gpu::DeviceArray<float> A, const int n,
                          const int m);

Eigen::MatrixXf load_blendshapes_from_binary(const std::string& file_name,
                                             const int v_num) {
  std::ifstream is(file_name, std::ios::binary);
  if (!is) {
    std::cout << "Error loading blendshapes!" << std::endl;
  }

  // loading data
  std::vector<float> data;
  while (true) {
    float x;
    if (is.read(reinterpret_cast<char*>(&x), sizeof(float))) {
      data.push_back(x);
    } else {
      break;
    }
  }  // end of loading

  if (data.size() % (v_num * 3) != 0) {
    std::cout << "Blendshapes Format Error!" << std::endl;
  }

  // converting data into Eigen::Matrix
  const int rows = v_num * 3;
  const int cols = data.size() / rows;
  Eigen::MatrixXf m_data(rows, cols);
  int cnt = 0;
  for (auto i = 0; i < cols; i++) {
    for (auto j = 0; j < rows; j++) {
      m_data(j, i) = data[cnt];
      ++cnt;
    }
  }  // end of converting

  return m_data;
}

bool BaselModel::LoadModel(std::string filename) {
  mxArray* p_array = NULL;
  float* pt = NULL;
  int mDim, nDim;
  int pt_cnt = 0;
  MATFile* p_file = matOpen(filename.c_str(), "r");
  if (p_file == NULL) std::runtime_error("Cannot Open Model.mat!");
  params_.clear();
  params_.resize(N_Params);
  for (int kk = 0; kk < N_Params; kk++) {
    p_array = matGetVariable(p_file, Params_Name[kk]);
    mDim = mxGetM(p_array);
    nDim = mxGetN(p_array);
    pt = (float*)mxGetData(p_array);
    pt_cnt = 0;
    params_[kk] = Eigen::MatrixXf(mDim, nDim);
    // todo-more stable copy method
    for (int idx_n = 0; idx_n < nDim; idx_n++)
      for (int idx_m = 0; idx_m < mDim; idx_m++, pt_cnt++)
        params_[kk](idx_m, idx_n) = *(pt + pt_cnt);
    mxFree(pt);
  }
  // close file
  matClose(p_file);
  // normalize color
  params_[mu_color()].array() = params_[mu_color()].array() / 255.f;
  params_[pca_color()].array() = params_[pca_color()].array() / 255.f;
  params_[pca_color_unnorm()].array() =
      params_[pca_color_unnorm()].array() / 255.f;

  {
    std::vector<int> blendshapes_index;
    FILE* fp = fopen("./blendshapesIndex2.txt", "r");
    int index_id, indexNum;
    fscanf(fp, "%d", &indexNum);

    for (int i = 0; i < indexNum; ++i) {
      fscanf(fp, "%d", &index_id);
      blendshapes_index.push_back(index_id);
    }
    fclose(fp);

    Eigen::MatrixXf new_expr_disp_index(params_[expr_disp_intel()].rows(),
                                        blendshapes_index.size());
    for (int i = 0; i < blendshapes_index.size(); ++i) {
      if (blendshapes_index[i] != -1)
        new_expr_disp_index.col(i) =
            params_[expr_disp_intel()].col(blendshapes_index[i]);
      else
        new_expr_disp_index.col(i).setZero();
    }
    params_[expr_disp_intel()] = new_expr_disp_index;
  }

  dim_id_ = params_[pca_shape()].cols();
  dim_color_ = params_[pca_color()].cols();
  dim_exp_ = params_[pca_exp()].cols();
  dim_intel_ = params_[expr_disp_intel()].cols();
  dim_exp_ = dim_intel_;
  std::cout << dim_exp_ << std::endl;
  idx_exp_ = 0;
  mu_shapePCA_ = params_[mu_shape()] + params_[mu_exp()];
  vertexNum = params_[mu_shape()].size() / 3;
  // process strip
  int nStrip = numStrip();
  std::cout << "Pre-processing " << nStrip << " Strips" << std::endl;
  strip_start = Eigen::MatrixXf(nStrip, 1);
  strip_end = Eigen::MatrixXf(nStrip, 1);
  strip_start(0, 0) = 0;
  strip_end(0, 0) = numStripByIndex(0) - 1;
  for (int kk = 1; kk < nStrip; kk++) {
    strip_start(kk, 0) = strip_end(kk - 1, 0) + 1;
    strip_end(kk, 0) = strip_start(kk, 0) + numStripByIndex(kk) - 1;
  }

#ifdef USE_EXPR_PCA
  isUseExprPCA = true;
#else
  isUseExprPCA = false;
#endif  // USE_EXPR_PCA
  return true;
}

bool BaselModel::UpdateModel(Eigen::VectorXf identity_coef,
                             Eigen::VectorXf color_coef,
                             Eigen::VectorXf expr_coef, Mesh& mesh_model) {
  if (isUseExprPCA) {
    cur_shapeId_ = mu_shapePCA_ + params_[pca_shape()] * identity_coef;
    cur_shape_ = cur_shapeId_ + params_[pca_exp()] * expr_coef;
  } else {
    cur_shapeId_ = params_[mu_shape()] + params_[pca_shape()] * identity_coef;
    cur_shape_ = cur_shapeId_ + params_[expr_disp_intel()] * expr_coef;
  }
  int n_verts = mesh_model.position_.rows();
  int off3 = 0;
#pragma omp parallel for schedule( \
    dynamic)  // Using OpenMP to try to parallelise the loop
  for (int kk = 0; kk < n_verts; kk++) {
    off3 = 3 * kk;
    mesh_model.position_(kk, 0) = cur_shape_(off3, 0);
    mesh_model.position_(kk, 1) = cur_shape_(off3 + 1, 0);
    mesh_model.position_(kk, 2) = cur_shape_(off3 + 2, 0);
  }
  // update normal
  mesh_model.update_normal();
  return true;
}

// todo : make a better implementation
bool BaselModel::RandomFace(Mesh& mesh_model) {
  Eigen::VectorXf identity_coef = Eigen::VectorXf(dim_id_);
  identity_coef.setZero();
  Eigen::VectorXf color_coef = Eigen::VectorXf(dim_color_);
  color_coef.setZero();
  Eigen::VectorXf expr_coef = Eigen::VectorXf(dim_exp_);
  expr_coef.setZero();
  idx_exp_ = idx_exp_ % dim_exp_;
  expr_coef(idx_exp_) = 1.0f;

  cur_shape_ = params_[mu_shape()] + params_[pca_shape()] * identity_coef +
               params_[expr_disp_intel()] * expr_coef;
  cur_color_ = params_[mu_color()] + params_[pca_color()] * color_coef;
  // copy to model
  int n_verts = mesh_model.position_.rows();
  int off3 = 0;
  for (int kk = 0; kk < n_verts; kk++) {
    off3 = 3 * kk;
    mesh_model.position_(kk, 0) = cur_shape_(off3);
    mesh_model.position_(kk, 1) = cur_shape_(off3 + 1);
    mesh_model.position_(kk, 2) = cur_shape_(off3 + 2);
    mesh_model.color_(kk, 0) = cur_color_(off3);
    mesh_model.color_(kk, 1) = cur_color_(off3 + 1);
    mesh_model.color_(kk, 2) = cur_color_(off3 + 2);
  }
  mesh_model.update_normal();
  idx_exp_++;
  return true;
}

#ifdef USE_CUDA
void cudaBaselModel::cudaUpdateModelID(
    const pcl::gpu::DeviceArray<float>& id_coefficients) {
  if (id_coefficients.size() != dim_id_) {
    std::cerr << __FUNCTION__ << ": ID Coefficients Mismatch!" << std::endl;
    return;
  }
  cudaProductMatVectorII(cuda_cur_shape_, id_base_, id_coefficients,
                         cuda_mean_shape_);
}

void cudaBaselModel::cudaUpdateModelEXP(
    const pcl::gpu::DeviceArray<float>& exp_coefficients) {
#ifdef USE_FACESHIFT_EXP
  auto cuda_mean_shape_ = cuda_loaded_mean_shape_;
  auto exp_base_ = loaded_exp_base_;
#endif
  if (exp_coefficients.size() != dim_exp_) {
    std::cerr << __FUNCTION__ << ": ID Coefficients Mismatch!" << std::endl;
    return;
  }
  cudaProductMatVectorII(cuda_cur_shape_, exp_base_, exp_coefficients,
                         cuda_mean_shape_);
}

void cudaBaselModel::cudaUpdateModelColor(
    const pcl::gpu::DeviceArray<float>& color_coefficients) const {
  if (color_coefficients.size() != dim_color_) {
    std::cerr << __FUNCTION__ << ": Color Coefficients Mismatch!" << std::endl;
    return;
  }
  cudaProductMatVectorII(cuda_cur_color_, color_base_, color_coefficients,
                         cuda_mean_color_);
}

void cudaBaselModel::cudaUpdateModel(
    const pcl::gpu::DeviceArray<float>& id_coefficients,
    const pcl::gpu::DeviceArray<float>& exp_coefficients) {
  if (id_coefficients.size() != dim_id_) {
    std::cerr << __FUNCTION__ << ": ID Coefficients Mismatch! Input DIM: "
              << id_coefficients.size() << " expected " << dim_id_ << std::endl;
    return;
  }
  if (exp_coefficients.size() != dim_exp_) {
    std::cerr << __FUNCTION__
              << ": EXP Coefficients Mismatch! Input DIM: " << exp_coefficients
              << "expected " << dim_exp_ << std::endl;
    return;
  }
  if (!is_use_personalized_model) {
    cudaProductMatVectorII(cuda_cur_shape_, id_base_, id_coefficients,
                           cuda_mean_shape_);
    cudaProductMatVectorII(cuda_cur_shape_, exp_base_, exp_coefficients,
                           cuda_cur_shape_);
  } else {
    cudaProductMatVectorII(cuda_cur_shape_, personalized_exp_base_,
                           exp_coefficients, cuda_personalized_mean_shape_);
  }
}

inline void computeV(Eigen::Matrix3f& V, const Eigen::MatrixXf& position_,
                     int id0, int id1, int id2) {
  // Eigen::Matrix3f Vs;
  // Eigen::Matrix3f V;
  Eigen::Vector3f v0 = position_.row(id0).transpose();
  Eigen::Vector3f v1 = position_.row(id1).transpose();
  Eigen::Vector3f v2 = position_.row(id2).transpose();
  V.col(0) = v1 - v0;
  V.col(1) = v2 - v0;
  Eigen::Vector3f v3 = (v1 - v0).cross(v2 - v0);
  V.col(2) = v3 / v3.norm();
  // std::cout << V << std::endl;
  // return V;
}

void cudaBaselModel::cudaUpdatePersonalizedModel(
    const pcl::gpu::DeviceArray<float> exp_coefficients) {
  cudaProductMatVectorII(cuda_cur_shape_, personalized_exp_base_,
                         exp_coefficients, cuda_personalized_mean_shape_);
}

void cudaBaselModel::cudaUpdateStaticPersonalizedModel(
    const pcl::gpu::DeviceArray<float> exp_coefficients) {
  cudaProductMatVectorII(cuda_cur_shape_, personalized_exp_base_static_,
                         exp_coefficients, cuda_personalized_mean_shape_);
}

void cudaBaselModel::saveExpBlendshapeModel(ObjMesh& obj,
                                            const std::string& filePath,
                                            const int idx) {
  DLOG(INFO) << "Save Blendshapes in " << filePath;
  CreateDirectory(filePath.c_str(), nullptr);
  std::vector<float> host_exp(dim_exp_, 0.0f);
  pcl::gpu::DeviceArray<float> exp_coefficients;
  obj.position_.transposeInPlace();
  cuda_personalized_mean_shape_.download((float3*)obj.position_.data());
  obj.position_.transposeInPlace();
  obj.write_obj(filePath + "/" + std::to_string(idx) + "_mean" + ".obj");
  obj.write_obj(filePath + "/mean.obj");

  // save blendshapes in the exp file
  Eigen::MatrixXf personalized_base(3 * vertexNum, dim_exp_);
  personalized_base.transposeInPlace();
  personalized_exp_base_.download(personalized_base.data());
  personalized_base.transposeInPlace();
  std::ofstream os(filePath + "/blendshapes.bin", std::ios::binary);
  os.write(reinterpret_cast<const char*>(personalized_base.data()),
           sizeof(float) * personalized_base.size());
  DLOG(INFO) << "Save Blendshapes Ends";
}

void cudaBaselModel::updatePersonalizedCorrelationReg() {
  cudaCompuateATA_diag(personalized_ATA_, reg_personalized_exp_,
                       personalized_exp_base_,
                       personalized_exp_base_.size() / dim_exp_, dim_exp_);
}

void cudaBaselModel::updateExpBaseCorrelationReg(
    pcl::gpu::DeviceArray<float> ATA, pcl::gpu::DeviceArray<float> reg,
    const pcl::gpu::DeviceArray<float> exp_base) {
  cudaCompuateATA_diag(ATA, reg, exp_base, exp_base.size() / dim_exp_,
                       dim_exp_);
}

void cudaBaselModel::RefineBlendshape(pcl::gpu::DeviceArray<float3> mean_shape,
                                      pcl::gpu::DeviceArray<float> base,
                                      cudaStream_t stream) {
  pcl::gpu::DeviceArray<float> rotation(dim_exp_ * 9);
  pcl::gpu::DeviceArray<float> translation(dim_exp_ * 3);
  for (int k = 0; k < 3; ++k) {
    cudaComputeBlendshape(base, mean_shape, stream);
    cudaComputeRotationRefine(rotation, base, mean_shape, stream);
    Eigen::MatrixXf host_rotation(3, 3 * dim_exp_);
    rotation.download(host_rotation.data());
    for (int i = 0; i < dim_exp_; ++i) {
      Eigen::Matrix3f rotation_i = host_rotation.block(0, 3 * i, 3, 3);
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(
          rotation_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
      rotation_i = svd.matrixV() * svd.matrixU().transpose();
      host_rotation.block(0, 3 * i, 3, 3) = rotation_i;
    }
    rotation.upload(host_rotation.data(), host_rotation.size());
    cudaComputeTranslationRefine(translation, rotation, base, mean_shape,
                                 stream);
    cudaComputeSRTBase(base, rotation, translation, mean_shape, stream);
    cudaStreamSynchronize(0);
  }
}

#ifdef USE_FACESHIFT_EXP
void cudaBaselModel::LoadFaceShift() {
  auto bs =
      load_blendshapes_from_binary("./data/model/faceshift.bin", vertexNum);
  dim_exp_ = bs.cols() - 1;

  Eigen::VectorXf host_faceshift_mean_shape = bs.col(0);
  cuda_loaded_mean_shape_.upload(
      reinterpret_cast<float3*>(host_faceshift_mean_shape.data()),
      host_faceshift_mean_shape.size() / 3);
  cuda_loaded_mean_shape_.copyTo(cuda_mean_shape_);
  host_faceshift_exp_base = bs.block(0, 1, vertexNum * 3, dim_exp_);
  host_faceshift_exp_base.transposeInPlace();
  loaded_exp_base_.upload(host_faceshift_exp_base.data(),
                          host_faceshift_exp_base.size());
}

#endif

void cudaBaselModel::UpdatePersonalizedExpBase(
    const Mesh& personalized_mean_shape) {
  // personalized_mean_shape.position().copyTo(cuda_personalized_mean_shape_);
#ifdef USE_FACESHIFT_EXP
  auto cuda_mean_shape_ = cuda_loaded_mean_shape_;
  auto exp_base_ = loaded_exp_base_;
#endif

  pcl::gpu::DeviceArray<float> vs_(personalized_mean_shape.n_tri_ * 12),
      vt_(personalized_mean_shape.n_tri_ * 12),
      Atb(3 *
          (personalized_mean_shape.n_tri_ + personalized_mean_shape.n_verts_));
  cudaInitV_(vs_, cuda_mean_shape_, personalized_mean_shape.cuda_tri_list_);
  cudaInitV_(vt_, cuda_personalized_mean_shape_,
             personalized_mean_shape.cuda_tri_list_);
  std::cout << "Successfully Init V" << std::endl;

  cudaStreamSynchronize(0);
  Eigen::MatrixXf host_personalized_mean_shape(personalized_mean_shape.n_verts_,
                                               3);
  Eigen::MatrixXf host_mean_shape(personalized_mean_shape.n_verts_, 3);
  host_mean_shape.transposeInPlace();
  host_personalized_mean_shape.transposeInPlace();
  cuda_mean_shape_.download((float3*)host_mean_shape.data());
  cuda_personalized_mean_shape_.download(
      (float3*)host_personalized_mean_shape.data());
  host_mean_shape.transposeInPlace();
  host_personalized_mean_shape.transposeInPlace();

  // copy to model
  int n_verts = personalized_mean_shape.position_.rows();

  Eigen::SparseMatrix<float> A(
      9 * personalized_mean_shape.n_tri_,
      3 * (personalized_mean_shape.n_tri_ + personalized_mean_shape.n_verts_));
  typedef Eigen::Triplet<float> T;
  std::vector<T> triplet_list;
  Eigen::VectorXf b(9 * personalized_mean_shape.n_tri_);
  Eigen::VectorXf host_Atb(Atb.size());
  Eigen::MatrixXf cMatrix(4, 3);
  cMatrix << -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1;
  // compute A
  for (int i = 0; i < personalized_mean_shape.n_tri_; i++) {
    auto id0 = personalized_mean_shape.tri_list_(i, 0);
    auto id1 = personalized_mean_shape.tri_list_(i, 1);
    auto id2 = personalized_mean_shape.tri_list_(i, 2);

    Eigen::Matrix3f Vt;
    computeV(Vt, host_personalized_mean_shape, id0, id1, id2);

    Eigen::Matrix3f Vt_inv = Vt.inverse();

    // init A
    Eigen::MatrixXf A_ = cMatrix * Vt_inv;
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        triplet_list.push_back(T(i * 9 + j * 3 + k, id0 * 3 + k, A_(0, j)));
        triplet_list.push_back(T(i * 9 + j * 3 + k, id1 * 3 + k, A_(1, j)));
        triplet_list.push_back(T(i * 9 + j * 3 + k, id2 * 3 + k, A_(2, j)));
        triplet_list.push_back(
            T(i * 9 + j * 3 + k, n_verts * 3 + i * 3 + k, A_(3, j)));
      }
    }
  }
  A.setFromTriplets(triplet_list.begin(), triplet_list.end());
  Eigen::SparseMatrix<float> AtA = A.transpose() * A;
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> solver(AtA);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    std::cout << false;
  }
  Eigen::MatrixXf host_personalized_exp_base_(n_verts * 3, dim_exp_);
  Eigen::VectorXf host_personalized_mean_shape_V(
      personalized_mean_shape.n_verts_ * 3);
  cuda_personalized_mean_shape_.download(
      (float3*)host_personalized_mean_shape_V.data());

  for (int i = 0; i < dim_exp_; ++i) {
    cudaUpdateDimIdAtb(Atb, vs_, vt_, exp_base_, cuda_mean_shape_,
                       cuda_personalized_mean_shape_,
                       personalized_mean_shape.cuda_tri_list_,
                       personalized_mean_shape.cudafvLookUpTable,
                       personalized_mean_shape.cudafBegin,
                       personalized_mean_shape.cudafEnd, i);
    cudaStreamSynchronize(0);
    Atb.download(host_Atb.data());
    Eigen::VectorXf x = solver.solve(host_Atb);
    if (solver.info() != Eigen::Success) {
      std::cout << false;
    }
    host_personalized_exp_base_.col(i) =
        x.block(0, 0, vertexNum * 3, 1) - host_personalized_mean_shape_V;
    DLOG(INFO) << "Solve Successfully " << i;
  }
  personalized_ATA_.create(dim_exp_ * dim_exp_);
  reg_personalized_exp_.create(dim_exp_);

  host_personalized_exp_base_.transposeInPlace();
  personalized_exp_base_.upload(host_personalized_exp_base_.data(),
                                host_personalized_exp_base_.size());
  host_personalized_exp_base_.setZero();
  personalized_exp_base_corr_.upload(host_personalized_exp_base_.data(),
                                     host_personalized_exp_base_.size());
  RefineBlendshape(cuda_personalized_mean_shape_, personalized_exp_base_, 0);
  updatePersonalizedCorrelationReg();
  personalized_exp_base_static_.create(personalized_exp_base_.size());
  personalized_exp_base_.copyTo(personalized_exp_base_static_);
  is_personalized_model_generated = true;
  cudaStreamSynchronize(0);
}

void cudaBaselModel::updateLaplacianSmoothMask(ObjMesh& obj) {
#ifdef USE_FACESHIFT_EXP
  auto exp_base_ = loaded_exp_base_;
  auto cuda_mean_shape_ = cuda_loaded_mean_shape_;
#endif
  pcl::gpu::DeviceArray<float> rotation(dim_exp_ * 9);
  pcl::gpu::DeviceArray<float> translation(dim_exp_ * 3);
  std::cout << dim_exp_ << std::endl;
  for (int k = 0; k < 3; ++k) {
    cudaComputeBlendshape(exp_base_, cuda_mean_shape_, 0);
    cudaComputeRotationRefine(rotation, exp_base_, cuda_mean_shape_, 0);
    Eigen::MatrixXf host_rotation(3, 3 * dim_exp_);
    rotation.download(host_rotation.data());
    for (int i = 0; i < dim_exp_; ++i) {
      Eigen::Matrix3f rotation_i = host_rotation.block(0, 3 * i, 3, 3);
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(
          rotation_i, Eigen::ComputeFullU | Eigen::ComputeFullV);
      rotation_i = svd.matrixV() * svd.matrixU().transpose();
      host_rotation.block(0, 3 * i, 3, 3) = rotation_i;
    }
    rotation.upload(host_rotation.data(), host_rotation.size());
    cudaComputeTranslationRefine(translation, rotation, exp_base_,
                                 cuda_mean_shape_, 0);
    Eigen::MatrixXf host_translation(3, dim_exp_);
    translation.download(host_translation.data());

    cudaComputeSRTBase(exp_base_, rotation, translation, cuda_mean_shape_, 0);
    cudaStreamSynchronize(0);
  }
  // params_[expr_disp_intel()].transposeInPlace();
  // exp_base_.download(params_[expr_disp_intel()].data());
  // params_[expr_disp_intel()].transposeInPlace();

  Eigen::SparseMatrix<float> L(obj.n_verts_, obj.n_verts_);
  typedef Eigen::Triplet<float> T;
  std::vector<T> triplet_list;
  EdgeSet edge_set(obj);
  // FILE *fp = fopen("./laplacian_matrix.txt", "w+");
  for (int i = 0; i < obj.n_verts_; ++i) {
    int adj_num = edge_set.end(i) - edge_set.begin(i);
    float adj_num_inv = 1.0f / adj_num;
    triplet_list.push_back(T(i, i, 1.0f));
    // fprintf(fp, "%d %d %f\n", i + 1, i + 1, 1.0f);
    for (auto j = edge_set.begin(i); j != edge_set.end(i); ++j) {
      triplet_list.push_back(T(i, j->end_, -adj_num_inv));
      // fprintf(fp, "%d %d %f\n", i + 1, j->end_ + 1, -adj_num_inv);
    }
  }
  // fclose(fp);
  L.setFromTriplets(triplet_list.begin(), triplet_list.end());
  Eigen::SparseMatrix<float> I(obj.n_verts_, obj.n_verts_);
  I.setIdentity();
  Eigen::SparseMatrix<float> A = I + 1000000.0f * L.transpose() * L;

  Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> solver(A);
  // solver.compute(A);
  std::ifstream is("./mask_threshold.txt");
  std::vector<std::pair<float, float>> mask_threshold(dim_exp_);
  for (auto i = 0; i < dim_exp_; i++) {
    is >> mask_threshold[i].first >> mask_threshold[i].second;
  }
  std::vector<float> host_mask(vertexNum * 3 * dim_exp_);
  Eigen::VectorXf b(obj.n_verts_), x(obj.n_verts_);
  Eigen::MatrixXf host_exp_base(vertexNum * 3, dim_exp_);
  host_exp_base.transposeInPlace();
  exp_base_.download(host_exp_base.data());
  host_exp_base.transposeInPlace();

  // cuda_blendshape_mask_.download(host_mask);
  for (int i = 0; i < dim_exp_; ++i) {
    float max_i = 0.0f;
    for (int j = 0; j < vertexNum; ++j) {
      auto delta = host_exp_base.block(j * 3, i, 3, 1).norm();
      if (delta > mask_threshold[i].first) {
        b(j) = 1.0f;
      } else {
        b(j) = 0.0f;
      }
      // b(j) = host_mask[j * 3 * dim_exp_ + i];
    }
    x = solver.solve(b);
    auto process = [](float a) {
      if (a < 0) {
        return 0.0f;
      }
      // if (a > 0.4) {
      //  a = (a - 0.4) * 20 + 0.4;
      //}
      return a;
    };
    for (int j = 0; j < vertexNum; ++j) {
      host_mask[j * 3 * dim_exp_ + i] =
          process(x(j)) * mask_threshold[i].second;
      host_mask[(j * 3 + 1) * dim_exp_ + i] =
          process(x(j)) * mask_threshold[i].second;
      host_mask[(j * 3 + 2) * dim_exp_ + i] =
          process(x(j)) * mask_threshold[i].second;
    }
  }
  cuda_blendshape_mask_.upload(host_mask.data(), host_mask.size());
}

#endif  // USE_CUDA
