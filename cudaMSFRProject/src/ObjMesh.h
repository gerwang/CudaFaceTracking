#pragma once
#include <Eigen\Eigen>
#include <iostream>
#include <string>
#include <vector>

#include "Common.h"
#include "HostUtil.hpp"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <pcl\gpu\containers\device_array.h>
#endif  // USE_CUDA

struct finfo {
  int fid, vid, idx;
  bool finfo::operator<(const finfo& b) const { return vid < b.vid; }
};

struct einfo {
  int sid, eid, fid, indid = -1;
  bool einfo::operator<(const einfo& b) const {
    if (sid < b.sid) {
      return true;
    } else if (sid > b.sid) {
      return false;
    } else {
      return eid < b.eid;
    }
  }

  bool einfo::operator!=(const einfo& b) const {
    if (sid != b.sid) {
      return true;
    }
    if (eid > b.eid) {
      return true;
    }
    return false;
  }
};

void cudaSmoothMesh(pcl::gpu::DeviceArray<float3> shape,
                    const pcl::gpu::DeviceArray<int3> tri_list,
                    const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                    const pcl::gpu::DeviceArray<int1> fBegin,
                    const pcl::gpu::DeviceArray<int1> fEnd,
                    const pcl::gpu::DeviceArray<unsigned short> is_boundary);

struct ObjMesh {
  ObjMesh() {
    request_position_ = true;
    request_color_ = false;
    request_normal_ = false;
    request_tex_coord_ = false;
    request_face_normal_ = false;
    request_tri_list_ = true;
    save_color_ = true;
    save_normal_ = false;
    save_tex_coord_ = false;
  }

  ~ObjMesh() {}

  // io functions
  virtual void load_obj(std::string filename);
  void load_off(std::string filename);
  void write_obj(std::string filename);
  void write_textured_obj(std::string filename);

  void print_summary();
  virtual void update_normal();
  void normalize_model();
  void center_model();

  // deformation transfer: using the source deformation to deform this Objmeshs
  bool deform_transfer(const ObjMesh& source_undeformed,
                       const ObjMesh& source_deformed,
                       ObjMesh& target_deformed) const;

  ObjMesh(const ObjMesh& obj)
      : position_(obj.position_),
        color_(obj.color_),
        normal_(obj.normal_),
        tri_list_(obj.tri_list_),
        n_verts_(obj.n_verts_),
        n_tri_(obj.n_tri_) {}

  ObjMesh& operator=(const ObjMesh& obj) {
    fbegin = fbegin;
    fend = fend;
    position_ = obj.position_;
    color_ = obj.position_;
    normal_ = obj.normal_;
    tex_coord_ = obj.tex_coord_;
    face_normal_ = obj.face_normal_;
    tri_list_ = obj.tri_list_;
    tri_listTex_ = obj.tri_listTex_;
    tri_listNormal_ = obj.tri_listNormal_;
    positionIsomap_ = obj.positionIsomap_;
    texIsomap_ = obj.texIsomap_;
    normalIsomap_ = obj.normalIsomap_;
    n_verts_ = obj.n_verts_;
    ;
    n_tri_ = obj.n_tri_;
    albedo_ = obj.albedo_;
    return *this;
  }

  ObjMesh& operator=(ObjMesh&& obj) {
    fbegin = std::move(fbegin);
    fend = std::move(fend);
    position_ = std::move(obj.position_);
    color_ = std::move(obj.position_);
    albedo_ = std::move(obj.albedo_);
    normal_ = std::move(obj.normal_);
    tex_coord_ = std::move(obj.tex_coord_);
    face_normal_ = std::move(obj.face_normal_);
    tri_list_ = std::move(obj.tri_list_);
    tri_listTex_ = std::move(obj.tri_listTex_);
    tri_listNormal_ = std::move(obj.tri_listNormal_);
    positionIsomap_ = std::move(obj.positionIsomap_);
    texIsomap_ = std::move(obj.texIsomap_);
    normalIsomap_ = std::move(obj.normalIsomap_);
    n_verts_ = obj.n_verts_;
    obj.n_verts_ = 0;
    n_tri_ = obj.n_tri_;
    obj.n_tri_ = 0;
    return *this;
  }

  bool getNearestBarycentricPoint(Eigen::MatrixXf& result,
                                  Eigen::MatrixXf& vertex, int vid, float* dis);
  // vertices - position & color & normal & tex coordinate

  std::vector<int> fbegin, fend;
  std::vector<finfo> fvLookUpTable;

  Eigen::MatrixXf position_;
  Eigen::MatrixXf geoposition_;

  Eigen::MatrixXf color_;
  Eigen::MatrixXf albedo_;
  Eigen::MatrixXf normal_;
  Eigen::MatrixXf tex_coord_;
  // triangle - face normal & triangle list
  Eigen::MatrixXf face_normal_;
  Eigen::MatrixXi tri_list_;
  Eigen::MatrixXi tri_listTex_;
  Eigen::MatrixXi tri_listNormal_;
  // + for isomap generation
  Eigen::MatrixXf positionIsomap_;
  Eigen::MatrixXf texIsomap_;
  Eigen::MatrixXf normalIsomap_;

  // number of vertices & triangles
  int n_verts_;
  int n_tri_;
  // io prefix
  bool request_position_;
  bool request_color_;
  bool request_normal_;
  bool request_tex_coord_;
  bool request_face_normal_;
  bool request_tri_list_;
  bool save_color_;
  bool save_normal_;
  bool save_tex_coord_;
  virtual void init() {}
  void build_fvLookUpTable();
};

#ifdef USE_CUDA
class cudaObjMesh;
void cudaUpdateNormal(pcl::gpu::DeviceArray<float3> normal,
                      const pcl::gpu::DeviceArray<float3> x,
                      const pcl::gpu::DeviceArray<int3> tri_list,
                      const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                      const pcl::gpu::DeviceArray<int1> fbegin,
                      const pcl::gpu::DeviceArray<int1> fend);

void cudaUpdateFrontVertices(pcl::gpu::DeviceArray<unsigned short> is_front,
                             const pcl::gpu::DeviceArray<float3> position,
                             const pcl::gpu::DeviceArray<float> rotation,
                             const pcl::gpu::DeviceArray<float> translation,
                             cudaTextureObject_t depth_map,
                             const msfr::intrinsics camera, const int width,
                             const int height);

void cudaUpdateBoundary(pcl::gpu::DeviceArray<unsigned short> is_boundary,
                        const pcl::gpu::DeviceArray<int3> tri_list,
                        const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                        const pcl::gpu::DeviceArray<int1> fbegin,
                        const pcl::gpu::DeviceArray<int1> fend);

class cudaObjMesh : public ObjMesh {
 public:
  cudaObjMesh() : ObjMesh() {}

  cudaObjMesh(const cudaObjMesh& obj)
      : ObjMesh(obj),
        cuda_position_(obj.n_verts_),
        cuda_color_(obj.n_verts_),
        cuda_normal_(obj.n_verts_),
        cuda_tri_list_(obj.n_tri_),
        cudafvLookUpTable(3 * obj.n_tri_),
        cudafBegin(obj.n_verts_),
        cudafEnd(obj.n_verts_),
        cuda_is_front_now(n_verts_) {
    cuda_position_ = obj.cuda_position_;
    cuda_color_ = obj.cuda_color_;
    cuda_normal_ = obj.cuda_normal_;
    cuda_tri_list_ = obj.cuda_tri_list_;
    cudafvLookUpTable = obj.cudafvLookUpTable;
    cudafBegin = obj.cudafBegin;
    cudafEnd = obj.cudafEnd;
  }

  cudaObjMesh(const ObjMesh& obj) : ObjMesh(obj) {
    cuda_position_.upload(reinterpret_cast<const float3*>(obj.position_.data()),
                          obj.n_verts_);
    cuda_color_.upload(reinterpret_cast<const float3*>(obj.color_.data()),
                       obj.n_verts_);  // because float3, so that's right
    cuda_normal_.upload(reinterpret_cast<const float3*>(obj.normal_.data()),
                        obj.n_verts_);
    cuda_tri_list_.upload(reinterpret_cast<const int3*>(obj.tri_list_.data()),
                          obj.n_tri_);
    std::vector<int2> host_fvLookUpTable(3 * n_tri_);
    std::vector<int> host_fv_idx(3 * n_tri_);
    for (int i = 0; i < 3 * n_tri_; i++) {
      host_fvLookUpTable[i].x = fvLookUpTable[i].fid;
      host_fvLookUpTable[i].y = fvLookUpTable[i].vid;
      host_fv_idx[i] = fvLookUpTable[i].idx;
    }
    cudafvLookUpTable.upload(
        reinterpret_cast<const int2*>(host_fvLookUpTable.data()), 3 * n_tri_);
    cuda_fv_idx.upload(host_fv_idx);
    cudafBegin.upload(reinterpret_cast<const int1*>(fbegin.data()),
                      obj.n_verts_);
    cudafEnd.upload(reinterpret_cast<const int1*>(fend.data()), obj.n_verts_);
    cuda_is_front_now.create(obj.n_verts_);
  }

  // virtual void load_obj(std::string filename);

  virtual void init() {
    cuda_position_.create(n_verts_);
    cuda_color_.create(n_verts_);
    cuda_normal_.create(n_verts_);
    cuda_tri_list_.create(n_tri_);
    cudafvLookUpTable.create(3 * n_tri_);
    cudafBegin.create(n_verts_);
    cudafEnd.create(n_verts_);
    cuda_is_front_now.create(n_verts_);
    tri_list_.transposeInPlace();
    position_.transposeInPlace();

    color_.transposeInPlace();
    cuda_position_.upload(reinterpret_cast<const float3*>(position_.data()),
                          n_verts_);

    cuda_color_.upload(reinterpret_cast<const float3*>(color_.data()),
                       n_verts_);
    cuda_normal_.upload(reinterpret_cast<const float3*>(normal_.data()),
                        n_verts_);
    cuda_tri_list_.upload(reinterpret_cast<const int3*>(tri_list_.data()),
                          n_tri_);
    std::vector<int2> host_fvLookUpTable(3 * n_tri_);
    std::vector<int> host_fv_idx(3 * n_tri_);
    for (int i = 0; i < 3 * n_tri_; i++) {
      host_fvLookUpTable[i].x = fvLookUpTable[i].fid;
      host_fvLookUpTable[i].y = fvLookUpTable[i].vid;
      host_fv_idx[i] = fvLookUpTable[i].idx;
    }
    cudafvLookUpTable.upload(
        reinterpret_cast<const int2*>(host_fvLookUpTable.data()), 3 * n_tri_);
    cuda_fv_idx.upload(host_fv_idx);
    cudafBegin.upload(reinterpret_cast<const int1*>(fbegin.data()), n_verts_);
    cudafEnd.upload(reinterpret_cast<const int1*>(fend.data()), n_verts_);
    tri_list_.transposeInPlace();
    position_.transposeInPlace();

    color_.transposeInPlace();
    cuda_is_boundary_.create(n_verts_);
    update_boundary();
  }

  void set_position(pcl::gpu::DeviceArray<float3>& new_position) {
    if (new_position.size() != n_verts_) {
      std::cerr << __FUNCTION__ << ": "
                << "Vertices numbers mismatch!" << std::endl;
      return;
    }
    cuda_position_ = new_position;
  }

  void set_color(pcl::gpu::DeviceArray<float3>& new_color) {
    if (new_color.size() != n_verts_) {
      std::cerr << __FUNCTION__ << ": "
                << "Vertices numbers mismatch!" << std::endl;
      return;
    }
    cuda_color_ = new_color;
  }

  void update_normal() {
    cudaUpdateNormal(cuda_normal_, cuda_position_, cuda_tri_list_,
                     cudafvLookUpTable, cudafBegin, cudafEnd);
  }

  void updateOtherNormal(pcl::gpu::DeviceArray<float3> cudaNormal,
                         pcl::gpu::DeviceArray<float3> cudaPosition) const {
    cudaUpdateNormal(cudaNormal, cudaPosition, cuda_tri_list_,
                     cudafvLookUpTable, cudafBegin, cudafEnd);
  }

  void getFontVertices(const pcl::gpu::DeviceArray<float> rotation,
                       const pcl::gpu::DeviceArray<float> translation,
                       cudaTextureObject_t depth_map,
                       const msfr::intrinsics& camera, const int width,
                       const int height) {
    cudaUpdateFrontVertices(cuda_is_front_now, position(), rotation,
                            translation, depth_map, camera, width, height);
  }

  void download_position(const pcl::gpu::DeviceArray<float3> device_position);

  virtual const pcl::gpu::DeviceArray<float3>& position() const {
    return cuda_position_;
  }

  virtual const pcl::gpu::DeviceArray<float3>& normal() const {
    return cuda_normal_;
  }

 public:
  pcl::gpu::DeviceArray<float3> cuda_position_;

  pcl::gpu::DeviceArray<float3> cuda_color_;
  pcl::gpu::DeviceArray<float3> cuda_normal_;
  pcl::gpu::DeviceArray<int3> cuda_tri_list_;
  pcl::gpu::DeviceArray<int2> cudafvLookUpTable;
  pcl::gpu::DeviceArray<int1> cudafBegin, cudafEnd;
  pcl::gpu::DeviceArray<unsigned short> cuda_is_front_now;
  pcl::gpu::DeviceArray<unsigned short> cuda_is_boundary_;
  pcl::gpu::DeviceArray<int> cuda_fv_idx;
  void estimate_geoposition();

 private:
  void update_boundary();
};

typedef cudaObjMesh Mesh;
#else
typedef ObjMesh Mesh;
#endif  // USE_CUDA
