#include "ObjMesh.h"

#include <omp.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <fstream>
#include <iostream>

#include "GraphUtil.h"

void ObjMesh::load_obj(std::string filename) {
  // prepare all the prefixs
  std::vector<double> coords;
  std::vector<int> tris;
  std::vector<int> trisTexCoords;
  std::vector<int> trisNormals;
  std::vector<double> vColor;
  std::vector<double> vNormal;
  std::vector<double> tex_coords;

  printf("Loading OBJ file %s...\n", filename.c_str());
  double x, y, z, r, g, b, nx, ny, nz;
  double tx, ty, tz;
  std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
  // open the file, return if open fails
  FILE *file = fopen(filename.c_str(), "r");
  if (file == NULL) {
    printf(
        "Impossible to open the file ! Are you in the right path ? See "
        "Tutorial 1 for details\n");
    getchar();
    return;
  }
  // prepare prefix
  int pref_cnt = 0;
  if (request_position_) pref_cnt++;
  if (request_normal_) pref_cnt++;
  if (request_tex_coord_) pref_cnt++;

  while (1) {
    char lineHeader[128];
    // read the first word of the line
    int res = fscanf(file, "%s", lineHeader);
    if (res == EOF) break;
    // parse the file
    if (strcmp(lineHeader, "v") == 0) {
      if (request_position_ && request_color_) {
        fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &x, &y, &z, &r, &g, &b);
        coords.push_back(x);
        coords.push_back(y);
        coords.push_back(z);
        vColor.push_back(r);
        vColor.push_back(g);
        vColor.push_back(b);
      } else {
        fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &x, &y, &z);
        coords.push_back(x);
        coords.push_back(y);
        coords.push_back(z);
      }
    } else if (strcmp(lineHeader, "vn") == 0) {
      fscanf(file, "%lf %lf %lf\n", &nx, &ny, &nz);
      vNormal.push_back(nx);
      vNormal.push_back(ny);
      vNormal.push_back(nz);
    } else if (strcmp(lineHeader, "vt") == 0) {
      fscanf(file, "%lf %lf\n", &tx, &ty);
      tex_coords.push_back(tx);
      tex_coords.push_back(ty);

      // fscanf(file, "%lf %lf %lf\n", &tx, &ty, &tz);
      // tex_coords.push_back(tx);
      // tex_coords.push_back(ty);
      // tex_coord.push_back(tz);
    } else if (strcmp(lineHeader, "f") == 0) {
      std::string vertex1, vertex2, vertex3;
      unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
      if (pref_cnt == 1)
        fscanf(file, "%d %d %d\n", &vertexIndex[0], &vertexIndex[1],
               &vertexIndex[2]);
      else if (pref_cnt == 2) {
        if (request_normal_)
          fscanf(file, "%d//%d %d//%d %d//%d\n", &vertexIndex[0],
                 &normalIndex[0], &vertexIndex[1], &normalIndex[1],
                 &vertexIndex[2], &normalIndex[2]);
        else if (request_tex_coord_)
          fscanf(file, "%d/%d %d/%d %d/%d\n", &vertexIndex[0], &uvIndex[0],
                 &vertexIndex[1], &uvIndex[1], &vertexIndex[2], &uvIndex[2]);
      } else if (pref_cnt == 3)
        fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0],
               &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1],
               &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);

      tris.push_back(vertexIndex[0] - 1);
      tris.push_back(vertexIndex[1] - 1);
      tris.push_back(vertexIndex[2] - 1);

      if (request_tex_coord_) {
        trisTexCoords.push_back(uvIndex[0] - 1);
        trisTexCoords.push_back(uvIndex[1] - 1);
        trisTexCoords.push_back(uvIndex[2] - 1);
      }
      if (request_normal_) {
        trisNormals.push_back(normalIndex[0] - 1);
        trisNormals.push_back(normalIndex[1] - 1);
        trisNormals.push_back(normalIndex[2] - 1);
      }

    } else {
      // Probably a comment, eat up the rest of the line
      char stupidBuffer[1000];
      fgets(stupidBuffer, 1000, file);
    }
  }
  fclose(file);
  // post process
  this->n_tri_ = tris.size() / 3;
  this->n_verts_ = coords.size() / 3;
  this->position_ = Eigen::MatrixXf(n_verts_, 3);
  this->position_.setZero();
  this->color_ = Eigen::MatrixXf(n_verts_, 3);
  this->color_.setOnes(); /*this->color_ = this->color_ * 255.f;*/
  this->normal_ = Eigen::MatrixXf(n_verts_, 3);
  this->normal_.setZero();
  this->tex_coord_ = Eigen::MatrixXf(n_verts_, 2);
  this->tex_coord_.setZero();
  this->positionIsomap_ = Eigen::MatrixXf(n_verts_, 2);
  this->positionIsomap_.setZero();
  this->texIsomap_ = Eigen::MatrixXf(n_verts_, 2);
  this->texIsomap_.setZero();
  this->normalIsomap_ = Eigen::MatrixXf(n_verts_, 3);
  this->normalIsomap_.setZero();

  this->tri_list_ = Eigen::MatrixXi(n_tri_, 3);
  this->tri_list_.setZero();
  this->tri_listTex_ = Eigen::MatrixXi(n_tri_, 3);
  this->tri_listTex_.setZero();
  this->tri_listNormal_ = Eigen::MatrixXi(n_tri_, 3);
  this->tri_listNormal_.setZero();

  this->face_normal_ = Eigen::MatrixXf(n_tri_, 3);
  this->face_normal_.setZero();
  // copy data
  int off3, off2;
  for (int kk = 0; kk < n_verts_; kk++) {
    off3 = 3 * kk;
    off2 = 2 * kk;
    if (request_position_) this->position_(kk, 0) = coords[off3];
    this->position_(kk, 1) = coords[off3 + 1];
    this->position_(kk, 2) = coords[off3 + 2];
    if (request_color_) {
      this->color_(kk, 0) = vColor[off3];
      this->color_(kk, 1) = vColor[off3 + 1];
      this->color_(kk, 2) = vColor[off3 + 2];
    }
    if (request_normal_) {
      this->normal_(kk, 0) = vNormal[off3];
      this->normal_(kk, 1) = vNormal[off3 + 1];
      this->normal_(kk, 2) = vNormal[off3 + 2];
    }
    if (request_tex_coord_) {
      this->tex_coord_(kk, 0) = tex_coords[off2];
      this->tex_coord_(kk, 1) = tex_coords[off2 + 1];

      // adjust to -1 ~ 1
      this->positionIsomap_(kk, 0) = tex_coords[off2] * 2 - 1;
      this->positionIsomap_(kk, 1) = tex_coords[off2 + 1] * 2 - 1;
    }
  }
  for (int kk = 0; kk < this->n_tri_; kk++) {
    off3 = 3 * kk;
    if (request_tri_list_) this->tri_list_(kk, 0) = tris[off3];
    this->tri_list_(kk, 1) = tris[off3 + 1];
    this->tri_list_(kk, 2) = tris[off3 + 2];

    if (request_tex_coord_) {
      this->tri_listTex_(kk, 0) = trisTexCoords[off3];
      this->tri_listTex_(kk, 1) = trisTexCoords[off3 + 1];
      this->tri_listTex_(kk, 2) = trisTexCoords[off3 + 2];
    }

    if (request_normal_) {
      this->tri_listNormal_(kk, 0) = trisNormals[off3];
      this->tri_listNormal_(kk, 1) = trisNormals[off3 + 1];
      this->tri_listNormal_(kk, 2) = trisNormals[off3 + 2];
    }
  }
  albedo_ = color_;
  build_fvLookUpTable();
  init();
  return;
}

void ObjMesh::load_off(std::string filename) {
  // prepare all the prefixes
  std::vector<double> coords;
  std::vector<int> tris;
  std::vector<int> trisTexCoords;
  std::vector<int> trisNormals;
  std::vector<double> vColor;
  std::vector<double> vNormal;
  std::vector<double> tex_coords;

  printf("Loading OBJ file %s...\n", filename.c_str());
  double x, y, z, r, g, b, nx, ny, nz;
  double tx, ty, tz;
  std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
  // open the file, return if open fails
  FILE *file = fopen(filename.c_str(), "r");
  if (file == NULL) {
    printf(
        "Impossible to open the file ! Are you in the right path ? See "
        "Tutorial 1 for details\n");
    getchar();
    return;
  }
  // prepare prefix
  int pref_cnt = 0;
  if (request_position_) pref_cnt++;
  if (request_normal_) pref_cnt++;
  if (request_tex_coord_) pref_cnt++;

  int n, m, t;
  char lineHeader[128];
  // read the first word of the line
  int res = fscanf(file, "%s", lineHeader);
  if (res == EOF) return;
  fscanf(file, "%d %d %d", &n, &m, &t);
  std::vector<std::vector<double>> pos;
  std::vector<double> element;
  if (request_color_)
    element.resize(7, 0);
  else
    element.resize(4, 0);
  pos.resize(n, element);
  for (int i = 0; i < n; i++) {
    if (request_color_) {
      fscanf(file, "%lf %lf %lf %lf %lf %lf 255\n", &pos[i][0], &pos[i][1],
             &pos[i][2], &pos[i][4], &pos[i][5], &pos[i][6]);

    } else {
      fscanf(file, "%lf %lf %lf\n", &pos[i][0], &pos[i][1], &pos[i][2]);
    }
    pos[i][3] = i + 0.1;
  }
  struct {
    bool operator()(std::vector<double> a, std::vector<double> b) const {
      return a[2] < b[2];
    }
  } customLess;
  struct {
    bool operator()(std::vector<double> a, std::vector<double> b) const {
      if (a[1] < b[1]) return true;
      if (a[1] > b[1]) return false;
      return a[2] < b[2];
    }
  } customLess1;
  struct {
    bool operator()(std::vector<double> a, std::vector<double> b) const {
      if (a[0] < b[0]) return true;
      if (a[0] > b[0]) return false;
      if (a[1] < b[1]) return true;
      if (a[1] > b[1]) return false;
      return a[2] < b[2];
    }
  } customLess2;
  std::sort(pos.begin(), pos.end(), customLess);
  std::sort(pos.begin(), pos.end(), customLess1);
  std::sort(pos.begin(), pos.end(), customLess2);
  int cnt = 0;
  std::vector<int> newID;
  newID.resize(n, -1);
  coords.push_back(pos[0][0]);
  coords.push_back(pos[0][1]);
  coords.push_back(pos[0][2]);
  if (request_color_) {
    vColor.push_back(pos[0][4] / 255);
    vColor.push_back(pos[0][5] / 255);
    vColor.push_back(pos[0][6] / 255);
  }
  newID[(int)pos[0][3]] = 0;
  for (int i = 1; i < n; i++)
    if (pos[i][0] != pos[i - 1][0] || pos[i][1] != pos[i - 1][1] ||
        pos[i][2] != pos[i - 1][2]) {
      cnt++;
      coords.push_back(pos[i][0]);
      coords.push_back(pos[i][1]);
      coords.push_back(pos[i][2]);
      if (request_color_) {
        vColor.push_back(pos[i][4] / 255);
        vColor.push_back(pos[i][5] / 255);
        vColor.push_back(pos[i][6] / 255);
      }
      newID[(int)pos[i][3]] = cnt;
    } else {
      newID[(int)pos[i][3]] = cnt;
    }
  cnt++;
  fvLookUpTable.clear();
  for (int i = 0; i < m; i++) {
    fscanf(file, "%s", lineHeader);
    if (strcmp(lineHeader, "3") == 0) {
      finfo tempFinfo;
      unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
      fscanf(file, "%d %d %d\n", &vertexIndex[0], &vertexIndex[1],
             &vertexIndex[2]);
      tris.push_back(newID[vertexIndex[0]]);
      tris.push_back(newID[vertexIndex[1]]);
      tris.push_back(newID[vertexIndex[2]]);
      tempFinfo.vid = newID[vertexIndex[0]];
      tempFinfo.fid = i;
      fvLookUpTable.push_back(tempFinfo);

      tempFinfo.vid = newID[vertexIndex[1]];
      tempFinfo.fid = i;
      fvLookUpTable.push_back(tempFinfo);

      tempFinfo.vid = newID[vertexIndex[2]];
      tempFinfo.fid = i;
      fvLookUpTable.push_back(tempFinfo);
    }
  }
  fclose(file);
  pos.clear();
  newID.clear();

  std::sort(fvLookUpTable.begin(), fvLookUpTable.end());
  fbegin.resize(cnt);
  fend.resize(cnt);
  fbegin[0] = 0;
  for (int i = 1; i < fvLookUpTable.size(); i++)
    if (fvLookUpTable[i].vid != fvLookUpTable[i - 1].vid) {
      fbegin[fvLookUpTable[i].vid] = i;
      fend[fvLookUpTable[i - 1].vid] = i;
    }
  fend[cnt - 1] = fvLookUpTable.size();

  // post process
  this->n_tri_ = tris.size() / 3;
  this->n_verts_ = coords.size() / 3;
  this->position_ = Eigen::MatrixXf(n_verts_, 3);
  this->position_.setZero();
  this->color_ = Eigen::MatrixXf(n_verts_, 3);
  this->color_.setOnes(); /*this->color_ = this->color_ * 255.f;*/
  this->normal_ = Eigen::MatrixXf(n_verts_, 3);
  this->normal_.setZero();
  this->tex_coord_ = Eigen::MatrixXf(n_verts_, 2);
  this->tex_coord_.setZero();
  this->positionIsomap_ = Eigen::MatrixXf(n_verts_, 2);
  this->positionIsomap_.setZero();
  this->texIsomap_ = Eigen::MatrixXf(n_verts_, 2);
  this->texIsomap_.setZero();
  this->normalIsomap_ = Eigen::MatrixXf(n_verts_, 3);
  this->normalIsomap_.setZero();

  this->tri_list_ = Eigen::MatrixXi(n_tri_, 3);
  this->tri_list_.setZero();
  this->tri_listTex_ = Eigen::MatrixXi(n_tri_, 3);
  this->tri_listTex_.setZero();
  this->tri_listNormal_ = Eigen::MatrixXi(n_tri_, 3);
  this->tri_listNormal_.setZero();

  this->face_normal_ = Eigen::MatrixXf(n_tri_, 3);
  this->face_normal_.setZero();
  // copy data
  int off3, off2;
  for (int kk = 0; kk < n_verts_; kk++) {
    off3 = 3 * kk;
    off2 = 2 * kk;
    if (request_position_) this->position_(kk, 0) = coords[off3];
    this->position_(kk, 1) = coords[off3 + 1];
    this->position_(kk, 2) = coords[off3 + 2];
    if (request_color_) {
      this->color_(kk, 0) = vColor[off3];
      this->color_(kk, 1) = vColor[off3 + 1];
      this->color_(kk, 2) = vColor[off3 + 2];
    }
    if (request_normal_) {
      this->normal_(kk, 0) = vNormal[off3];
      this->normal_(kk, 1) = vNormal[off3 + 1];
      this->normal_(kk, 2) = vNormal[off3 + 2];
    }
    if (request_tex_coord_) {
      this->tex_coord_(kk, 0) = tex_coords[off2];
      this->tex_coord_(kk, 1) = tex_coords[off2 + 1];

      // adjust to -1 ~ 1
      this->positionIsomap_(kk, 0) = tex_coords[off2] * 2 - 1;
      this->positionIsomap_(kk, 1) = tex_coords[off2 + 1] * 2 - 1;
    }
  }
  for (int kk = 0; kk < this->n_tri_; kk++) {
    off3 = 3 * kk;
    if (request_tri_list_) this->tri_list_(kk, 0) = tris[off3];
    this->tri_list_(kk, 1) = tris[off3 + 1];
    this->tri_list_(kk, 2) = tris[off3 + 2];

    if (request_tex_coord_) {
      this->tri_listTex_(kk, 0) = trisTexCoords[off3];
      this->tri_listTex_(kk, 1) = trisTexCoords[off3 + 1];
      this->tri_listTex_(kk, 2) = trisTexCoords[off3 + 2];
    }

    if (request_normal_) {
      this->tri_listNormal_(kk, 0) = trisNormals[off3];
      this->tri_listNormal_(kk, 1) = trisNormals[off3 + 1];
      this->tri_listNormal_(kk, 2) = trisNormals[off3 + 2];
    }
  }
  albedo_ = color_;
  build_fvLookUpTable();
  init();
  return;
}

void ObjMesh::write_obj(std::string filename) {
  std::ofstream obj_file(filename);
  int pref_cnt = 1;
  if (save_normal_) pref_cnt++;
  if (save_tex_coord_) pref_cnt++;
  // write vertices
  if (save_color_) {
    for (std::size_t i = 0; i < position_.rows(); ++i) {
      obj_file << "v " << position_(i, 0) << " " << position_(i, 1) << " "
               << position_(i, 2) << " " << color_(i, 0) << " " << color_(i, 1)
               << " " << color_(i, 2) << " " << std::endl;
    }
  } else {
    for (std::size_t i = 0; i < position_.rows(); ++i) {
      obj_file << "v " << position_(i, 0) << " " << position_(i, 1) << " "
               << position_(i, 2) << " " << std::endl;
    }
  }
  // write normal
  if (save_normal_) {
    for (std::size_t i = 0; i < normal_.rows(); ++i) {
      obj_file << "vn " << normal_(i, 0) << " " << normal_(i, 1) << " "
               << normal_(i, 2) << " " << std::endl;
    }
  }
  // write tex coord
  if (save_tex_coord_) {
    for (std::size_t i = 0; i < tex_coord_.rows(); ++i) {
      obj_file << "vt " << tex_coord_(i, 0) << " " << tex_coord_(i, 1) << " "
               << std::endl;
    }
  }
  // write triangles
  for (int kk = 0; kk < tri_list_.rows(); kk++) {
    // Add one because obj starts counting triangle indices at 1
    if (pref_cnt == 1) {
      obj_file << "f " << tri_list_(kk, 0) + 1 << " " << tri_list_(kk, 1) + 1
               << " " << tri_list_(kk, 2) + 1 << std::endl;
    } else if (pref_cnt == 2) {
      obj_file << "f " << tri_list_(kk, 0) + 1 << "/" << tri_list_(kk, 0) + 1
               << " " << tri_list_(kk, 1) + 1 << "/" << tri_list_(kk, 1) + 1
               << " " << tri_list_(kk, 2) + 1 << "/" << tri_list_(kk, 2) + 1
               << std::endl;
    } else if (pref_cnt == 3) {
      obj_file << "f " << tri_list_(kk, 0) + 1 << "/" << tri_list_(kk, 0) + 1
               << "/" << tri_list_(kk, 0) + 1 << " " << tri_list_(kk, 1) + 1
               << "/" << tri_list_(kk, 1) + 1 << "/" << tri_list_(kk, 1) + 1
               << " " << tri_list_(kk, 2) + 1 << "/" << tri_list_(kk, 2) + 1
               << "/" << tri_list_(kk, 2) + 1 << std::endl;
    }
  }
  return;
}

void ObjMesh::print_summary() {
  std::cout << std::endl;
  std::cout << "Summary of Mesh Model: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  std::cout << "n vertices is " << n_verts_ << std::endl;
  std::cout << "n triangles is " << n_tri_ << std::endl;
  std::cout << "position of vertice-0 is ( " << position_(0, 0) << " , "
            << position_(0, 1) << " , " << position_(0, 2) << " ) "
            << std::endl;

  std::cout << "color of vertice-0 is ( " << color_(0, 0) << " , "
            << color_(0, 1) << " , " << color_(0, 2) << " ) " << std::endl;

  std::cout << "normal of vertice-0 is ( " << normal_(0, 0) << " , "
            << normal_(0, 1) << " , " << normal_(0, 2) << " ) " << std::endl;

  std::cout << "texcoord of vertice-0 is ( " << tex_coord_(0, 0) << " , "
            << tex_coord_(0, 1) << " ) " << std::endl;

  std::cout << "texcoord of vertice-1 is ( " << tex_coord_(1, 0) << " , "
            << tex_coord_(1, 1) << " ) " << std::endl;

  std::cout << "index of triangle-0 is ( " << tri_list_(0, 0) << " , "
            << tri_list_(0, 1) << " , " << tri_list_(0, 2) << " ) "
            << std::endl;

  std::cout << "index of triangle-1 is ( " << tri_list_(1, 0) << " , "
            << tri_list_(1, 1) << " , " << tri_list_(1, 2) << " ) "
            << std::endl;

  std::cout << "End of Summary: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            << std::endl;
  return;
}

void ObjMesh::update_normal() {
  this->normal_.resize(n_verts_, 3);
  this->face_normal_.resize(n_tri_, 3);
  this->normal_.setZero();
  this->face_normal_.setZero();
  std::vector<float> area_sum(this->n_verts_, 0.f);
  omp_lock_t writelock;
  omp_init_lock(&writelock);

#pragma omp parallel for

  for (int i = 0; i < n_tri_; ++i) {
    auto vidx0 = tri_list_(i, 0);
    auto vidx1 = tri_list_(i, 1);
    auto vidx2 = tri_list_(i, 2);

    auto v0 = Eigen::Vector3f(position_.row(vidx0));
    auto v1 = Eigen::Vector3f(position_.row(vidx1));
    auto v2 = Eigen::Vector3f(position_.row(vidx2));

    auto v0v1 = v1 - v0;
    auto v0v2 = v2 - v0;
    auto n = v0v1.cross(v0v2);
    double area = n.norm();

    omp_set_lock(&writelock);
    this->normal_.row(vidx0) += n;
    this->normal_.row(vidx1) += n;
    this->normal_.row(vidx2) += n;
    area_sum[vidx0] += area;
    area_sum[vidx1] += area;
    area_sum[vidx2] += area;
    omp_unset_lock(&writelock);
    n.normalize();
    this->face_normal_.row(i) = n;
  }
  omp_destroy_lock(&writelock);

#pragma omp parallel for
  for (int i = 0; i < n_verts_; ++i) {
    this->normal_.row(i) /= area_sum[i];
  }
}

void ObjMesh::normalize_model() {
  float mean_x = this->position_.col(0).sum() / this->n_verts_;
  float mean_y = this->position_.col(1).sum() / this->n_verts_;
  float mean_z = this->position_.col(2).sum() / this->n_verts_;

  this->position_.col(0) = this->position_.col(0).array() - mean_x;
  this->position_.col(1) = this->position_.col(1).array() - mean_y;
  this->position_.col(2) = this->position_.col(2).array() - mean_z;

  float max_x = this->position_.col(0).maxCoeff();
  float max_y = this->position_.col(1).maxCoeff();
  float max_z = this->position_.col(2).maxCoeff();

  float min_x = this->position_.col(0).minCoeff();
  float min_y = this->position_.col(1).minCoeff();
  float min_z = this->position_.col(2).minCoeff();

  float scale_x = 2 / (max_x - min_x + 0.00001);
  float scale_y = 2 / (max_y - min_y + 0.00001);
  float scale_z = 2 / (max_z - min_z + 0.00001);

  float scale = scale_x < scale_y ? scale_x : scale_y;
  scale = scale_z < scale ? scale_z : scale;
  this->position_.col(0) = (this->position_.col(0).array() - min_x) * scale;
  this->position_.col(1) = (this->position_.col(1).array() - min_y) * scale;
  this->position_.col(2) = (this->position_.col(2).array() - min_z) * scale;

  this->position_ = this->position_.array() - 1.0f;
  return;
}

void ObjMesh::center_model() {
  float mean_x = this->position_.col(0).sum() / this->n_verts_;
  float mean_y = this->position_.col(1).sum() / this->n_verts_;
  float mean_z = this->position_.col(2).sum() / this->n_verts_;

  this->position_.col(0) = this->position_.col(0).array() - mean_x;
  this->position_.col(1) = this->position_.col(1).array() - mean_y;
  this->position_.col(2) = this->position_.col(2).array() - mean_z;

  return;
}

inline Eigen::Matrix3f &computeV(const ObjMesh &obj, int id0, int id1,
                                 int id2) {
  Eigen::Matrix3f Vs;
  Eigen::Matrix3f V;
  Eigen::Vector3f v0 = obj.position_.row(id0).transpose();
  Eigen::Vector3f v1 = obj.position_.row(id1).transpose();
  Eigen::Vector3f v2 = obj.position_.row(id2).transpose();
  V.col(0) = v1 - v0;
  V.col(1) = v2 - v0;
  Eigen::Vector3f v3 = (v1 - v0).cross(v2 - v0);
  V.col(2) = v3 / v3.norm();
  return V;
}

bool ObjMesh::deform_transfer(const ObjMesh &source_undeformed,
                              const ObjMesh &source_deformed,
                              ObjMesh &target_deformed) const {
  // Check if source undeformed and source deformed has the same topology
  {
    bool is_different_topology = false;
    if (source_deformed.n_tri_ != source_undeformed.n_tri_ ||
        source_deformed.n_verts_ != source_undeformed.n_verts_ ||
        n_tri_ != source_deformed.n_tri_ ||
        n_verts_ !=
            source_deformed
                .n_verts_)  // check the numbers of the triangles and vertices
    {
      is_different_topology = true;
    }
    if (!is_different_topology)  // check details information
    {
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < n_tri_; i++) {
          if (source_deformed.tri_list_(i, j) !=
                  source_undeformed.tri_list_(i, j) ||
              tri_list_(i, j) != source_deformed.tri_list_(i, j)) {
            is_different_topology = true;
            break;
          }
        }
        if (is_different_topology) break;
      }
    }
    if (is_different_topology) {
      std::cerr << "ERROR: Different topology in deformation transfer!"
                << std::endl;
      return is_different_topology;
    }
  }

  // deformation transfer
  target_deformed = *this;
  Eigen::SparseMatrix<float> A(9 * n_tri_, 3 * (n_tri_ + n_verts_));
  typedef Eigen::Triplet<float> T;
  std::vector<T> triplet_list;
  Eigen::VectorXf b(9 * n_tri_);
  Eigen::MatrixXf cMatrix(4, 3);
  cMatrix << -1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1;
  // compute A and b
  for (int i = 0; i < n_tri_; i++) {
    auto id0 = tri_list_(i, 0);
    auto id1 = tri_list_(i, 1);
    auto id2 = tri_list_(i, 2);
    // init b
    Eigen::Matrix3f Vs = computeV(source_undeformed, id0, id1, id2);
    Eigen::Matrix3f Vs_ = computeV(source_deformed, id0, id1, id2);
    Eigen::Matrix3f Vt = computeV(*this, id0, id1, id2);

    Eigen::Matrix3f S = Vs_ * Vs.inverse();
    b.block(i * 9, 0, 3, 1) = S.col(0);
    b.block(i * 9 + 3, 0, 3, 1) = S.col(1);
    b.block(i * 9 + 6, 0, 3, 1) = S.col(2);
    // init A
    Eigen::MatrixXf A_ = cMatrix * Vt.inverse();
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        triplet_list.push_back(T(i * 9 + j * 3 + k, id0 * 3 + k, A_(0, j)));
        triplet_list.push_back(T(i * 9 + j * 3 + k, id1 * 3 + k, A_(1, j)));
        triplet_list.push_back(T(i * 9 + j * 3 + k, id2 * 3 + k, A_(2, j)));
        triplet_list.push_back(
            T(i * 9 + j * 3 + k, n_verts_ * 3 + i * 3 + k, A_(3, j)));
      }
    }
  }
  A.setFromTriplets(triplet_list.begin(), triplet_list.end());
  Eigen::SparseMatrix<float> AtA = A.transpose() * A;
  Eigen::VectorXf Atb = A.transpose() * b;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
  solver.compute(AtA);
  if (solver.info() != Eigen::Success) {
    // decomposition failed
    return false;
  }
  Eigen::VectorXf x = solver.solve(Atb);
  if (solver.info() != Eigen::Success) {
    // solving failed
    return false;
  }
  for (int i = 0; i < n_verts_; i++) {
    target_deformed.position_.row(i) = x.block(i * 3, 0, 3, 1).transpose();
  }
  target_deformed.update_normal();
  return true;
}

bool ObjMesh::getNearestBarycentricPoint(Eigen::MatrixXf &result,
                                         Eigen::MatrixXf &vertex, int vid,
                                         float *dis) {
  bool found = false;
  float dist =
      (vertex.transpose() - this->position_.block(vid, 0, 1, 3)).norm();
  float neg = 0;
  float min_neg = -1e8;
  float min_dist = dist;
  result = vertex.transpose();
  *dis = dist;
  for (int i = fbegin[vid]; i < fend[vid]; i++) {
    auto id0 = this->tri_list_(fvLookUpTable[i].fid, 0);
    auto id1 = this->tri_list_(fvLookUpTable[i].fid, 1);
    auto id2 = this->tri_list_(fvLookUpTable[i].fid, 2);
    Eigen::Vector3f v0 = this->position_.block(id0, 0, 1, 3).transpose();
    Eigen::Vector3f v1 = this->position_.block(id1, 0, 1, 3).transpose();
    Eigen::Vector3f v2 = this->position_.block(id2, 0, 1, 3).transpose();
    auto normal = (v1 - v0).cross(v2 - v0).normalized();
    float area_tri = (v1 - v0).cross(v2 - v0).dot(normal);
    float area_12 = (v1 - vertex).cross(v2 - vertex).dot(normal);
    float area_20 = (v2 - vertex).cross(v0 - vertex).dot(normal);
    // assert(abs(area_tri) > 1e-10);
    if (area_tri > 1e-10) {
      float c0 = area_12 / area_tri;
      float c1 = area_20 / area_tri;
      float c2 = 1 - c0 - c1;
      if (c0 >= 0 && c0 <= 1 && c1 >= 0 && c1 <= 1 && c2 >= 0 && c2 <= 1) {
        found = true;
        dist = (v0 * c0 + v1 * c1 + v2 * c2 - vertex).norm();
        if (dist < min_dist) {
          *dis = dist;
          min_dist = dist;
          result = (v0 * c0 + v1 * c1 + v2 * c2).transpose();
        }
      }
    }
    /*else if (!found)
    {
            neg = 0;
            if (c0 < 0)
                    neg += c0;
            if (c1 < 0)
                    neg += c1;
            if (c2 < 0)
                    neg += c2;
            if (neg > min_neg)
            {
                    min_neg = neg;
                    result = v0*c0 + v1*c1 + v2*c2;
            }
    }*/
  }
  return true;
}

//************************************
// Method:    build_fvLookUpTable
// FullName:  ObjMesh::build_fvLookUpTable
// Access:    private
// Returns:   void
//  This function can build a look up table which can make it convenient to find
//  the triangles corresponding to the given vertex.
//************************************
void ObjMesh::build_fvLookUpTable() {
  fvLookUpTable.clear();
  for (int i = 0; i < n_tri_; i++) {
    finfo tempFinfo;
    // einfo tempEinfo;

    tempFinfo.vid = tri_list_(i, 0);
    tempFinfo.fid = i;
    tempFinfo.idx = 0;
    fvLookUpTable.push_back(tempFinfo);
    tempFinfo.vid = tri_list_(i, 1);
    tempFinfo.idx = 1;
    fvLookUpTable.push_back(tempFinfo);
    tempFinfo.vid = tri_list_(i, 2);
    tempFinfo.idx = 2;
    tempFinfo.fid = i;
    fvLookUpTable.push_back(tempFinfo);
  }

  std::sort(fvLookUpTable.begin(), fvLookUpTable.end());
  fbegin.resize(n_verts_);
  fend.resize(n_verts_);
  fbegin[0] = 0;
  for (int i = 1; i < fvLookUpTable.size(); i++)
    if (fvLookUpTable[i].vid != fvLookUpTable[i - 1].vid) {
      fbegin[fvLookUpTable[i].vid] = i;
      fend[fvLookUpTable[i - 1].vid] = i;
    }
  fend[n_verts_ - 1] = fvLookUpTable.size();
}

void ObjMesh::write_textured_obj(std::string filename) {
  assert((position_.rows() == color_.rows() || color_.rows() == 0) &&
         !(tex_coord_.rows() == 0));
  std::ofstream obj_file(filename);
  std::string mtl_filename = filename + ".mtl";
  obj_file << "mtllib " << mtl_filename
           << std::endl;  // first line of the obj file
  if (color_.rows() == 0) {
    for (std::size_t i = 0; i < position_.rows(); ++i) {
      obj_file << "v " << position_(i, 0) << " " << position_(i, 1) << " "
               << position_(i, 2) << " " << std::endl;
    }
  } else {
    for (std::size_t i = 0; i < position_.rows(); ++i) {
      obj_file << "v " << position_(i, 0) << " " << position_(i, 1) << " "
               << position_(i, 2) << " " << color_(i, 0) << " " << color_(i, 1)
               << " " << color_(i, 2) << " " << std::endl;
    }
  }
  for (std::size_t i = 0; i < tex_coord_.rows(); ++i) {
    // obj_file << "vt " << tex_coord_(i, 0) << " " << 1.0f - tex_coord_(i, 1)
    // << std::endl;

    obj_file << "vt " << tex_coord_(i, 0) << " " << tex_coord_(i, 1)
             << std::endl;
  }
  obj_file
      << "usemtl FaceTexture"
      << std::endl;  // the name of our texture (material) will be 'FaceTexture'
  for (int kk = 0; kk < tri_list_.rows(); kk++) {
    // Add one because obj starts counting triangle indices at 1
    obj_file << "f " << tri_list_(kk, 0) + 1 << "/" << tri_listTex_(kk, 0) + 1
             << " " << tri_list_(kk, 1) + 1 << "/" << tri_listTex_(kk, 1) + 1
             << " " << tri_list_(kk, 2) + 1 << "/" << tri_listTex_(kk, 2) + 1
             << std::endl;
  }
  obj_file.close();

  std::ofstream mtl_file(mtl_filename);
  std::string texture_filename = filename + ".isomap.png";
  mtl_file << "newmtl FaceTexture" << std::endl;
  mtl_file << "map_Kd " << texture_filename << std::endl;
  mtl_file.close();
  return;
};

#ifdef USE_CUDA

// void cudaObjMesh::load_obj(std::string filename) {
//
//  // prepare all the prefixs
//  std::vector<double> coords;
//  std::vector<int> tris;
//  std::vector<int> trisTexCoords;
//  std::vector<int> trisNormals;
//  std::vector<double> vColor;
//  std::vector<double> vNormal;
//  std::vector<double> tex_coords;
//
//  printf("Loading OBJ file %s...\n", filename.c_str());
//  double x, y, z, r, g, b, nx, ny, nz;
//  double tx, ty, tz;
//  std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
//  // open the file, return if open fails
//  FILE * file = fopen(filename.c_str(), "r");
//  if (file == NULL) {
//    printf("Impossible to open the file ! Are you in the right path ? See
//    Tutorial 1 for details\n"); getchar(); return;
//  }
//  // prepare prefix
//  int pref_cnt = 0;
//  if (request_position_) pref_cnt++;
//  if (request_normal_) pref_cnt++;
//  if (request_tex_coord_) pref_cnt++;
//
//  while (1) {
//
//    char lineHeader[128];
//    // read the first word of the line
//    int res = fscanf(file, "%s", lineHeader);
//    if (res == EOF)
//      break;
//    // parse the file
//    if (strcmp(lineHeader, "v") == 0) {
//      if (request_position_ && request_color_) {
//        fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &x, &y, &z, &r, &g, &b);
//        coords.push_back(x); coords.push_back(y); coords.push_back(z);
//        vColor.push_back(r); vColor.push_back(g); vColor.push_back(b);
//      }
//      else {
//        fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &x, &y, &z);
//        coords.push_back(x); coords.push_back(y); coords.push_back(z);
//      }
//    }
//    else if (strcmp(lineHeader, "vn") == 0) {
//      fscanf(file, "%lf %lf %lf\n", &nx, &ny, &nz);
//      vNormal.push_back(nx); vNormal.push_back(ny); vNormal.push_back(nz);
//    }
//    else if (strcmp(lineHeader, "vt") == 0) {
//      fscanf(file, "%lf %lf\n", &tx, &ty);
//      tex_coords.push_back(tx);
//      tex_coords.push_back(ty);
//
//      //fscanf(file, "%lf %lf %lf\n", &tx, &ty, &tz);
//      //tex_coords.push_back(tx);
//      //tex_coords.push_back(ty);
//      // tex_coord.push_back(tz);
//    }
//    else if (strcmp(lineHeader, "f") == 0) {
//      std::string vertex1, vertex2, vertex3;
//      unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
//      if (pref_cnt == 1)
//        fscanf(file, "%d %d %d\n", &vertexIndex[0], &vertexIndex[1],
//        &vertexIndex[2]);
//      else if (pref_cnt == 2) {
//        if (request_normal_)
//          fscanf(file, "%d/%d %d/%d %d/%d\n", &vertexIndex[0],
//          &normalIndex[0], &vertexIndex[1], &normalIndex[1], &vertexIndex[2],
//          &normalIndex[2]);
//        else if (request_tex_coord_)
//          fscanf(file, "%d/%d %d/%d %d/%d\n", &vertexIndex[0], &uvIndex[0],
//          &vertexIndex[1], &uvIndex[1], &vertexIndex[2], &uvIndex[2]);
//      }
//      else if (pref_cnt == 3)
//        fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0],
//        &uvIndex[0], &normalIndex[0],
//          &vertexIndex[1], &uvIndex[1], &normalIndex[1],
//          &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
//
//      tris.push_back(vertexIndex[0] - 1);
//      tris.push_back(vertexIndex[1] - 1);
//      tris.push_back(vertexIndex[2] - 1);
//
//      if (request_tex_coord_) {
//        trisTexCoords.push_back(uvIndex[0] - 1);
//        trisTexCoords.push_back(uvIndex[1] - 1);
//        trisTexCoords.push_back(uvIndex[2] - 1);
//      }
//      if (request_normal_) {
//        trisNormals.push_back(normalIndex[0] - 1);
//        trisNormals.push_back(normalIndex[1] - 1);
//        trisNormals.push_back(normalIndex[2] - 1);
//      }
//
//    }
//    else {
//      // Probably a comment, eat up the rest of the line
//      char stupidBuffer[1000];
//      fgets(stupidBuffer, 1000, file);
//    }
//  }
//  fclose(file);
//  // post process
//  this->n_tri_ = tris.size() / 3;
//  this->n_verts_ = coords.size() / 3;
//  this->position_ = Eigen::MatrixXf(n_verts_, 3);
//  this->position_.setZero(); this->color_ = Eigen::MatrixXf(n_verts_, 3);
//  this->color_.setOnes(); /*this->color_ = this->color_ * 255.f;*/
//  this->normal_ = Eigen::MatrixXf(n_verts_, 3); this->normal_.setZero();
//  this->tex_coord_ = Eigen::MatrixXf(n_verts_, 2);
//  this->tex_coord_.setZero(); this->positionIsomap_ =
//  Eigen::MatrixXf(n_verts_, 2);	this->positionIsomap_.setZero();
//  this->texIsomap_ = Eigen::MatrixXf(n_verts_, 2); this->texIsomap_.setZero();
//  this->normalIsomap_ = Eigen::MatrixXf(n_verts_, 3);
//  this->normalIsomap_.setZero();
//
//  this->tri_list_ = Eigen::MatrixXi(n_tri_, 3); this->tri_list_.setZero();
//  this->tri_listTex_ = Eigen::MatrixXi(n_tri_, 3);
//  this->tri_listTex_.setZero(); this->tri_listNormal_ =
//  Eigen::MatrixXi(n_tri_, 3); this->tri_listNormal_.setZero();
//
//  this->face_normal_ = Eigen::MatrixXf(n_tri_, 3);
//  this->face_normal_.setZero();
//  // copy data
//  int off3, off2;
//  for (int kk = 0; kk < n_verts_; kk++) {
//    off3 = 3 * kk; off2 = 2 * kk;
//    if (request_position_)
//      this->position_(kk, 0) = coords[off3]; this->position_(kk, 1) =
//      coords[off3 + 1]; this->position_(kk, 2) = coords[off3 + 2];
//    if (request_color_) {
//      this->color_(kk, 0) = vColor[off3]; this->color_(kk, 1) = vColor[off3 +
//      1]; this->color_(kk, 2) = vColor[off3 + 2];
//    }
//    if (request_normal_) {
//      this->normal_(kk, 0) = vNormal[off3]; this->normal_(kk, 1) =
//      vNormal[off3 + 1]; this->normal_(kk, 2) = vNormal[off3 + 2];
//    }
//    if (request_tex_coord_) {
//      this->tex_coord_(kk, 0) = tex_coords[off2]; this->tex_coord_(kk, 1) =
//      tex_coords[off2 + 1];
//
//      // adjust to -1 ~ 1
//      this->positionIsomap_(kk, 0) = tex_coords[off2] * 2 - 1;
//      this->positionIsomap_(kk, 1) = tex_coords[off2 + 1] * 2 - 1;
//    }
//  }
//  for (int kk = 0; kk < this->n_tri_; kk++) {
//    off3 = 3 * kk;
//    if (request_tri_list_)
//      this->tri_list_(kk, 0) = tris[off3]; this->tri_list_(kk, 1) = tris[off3
//      + 1]; this->tri_list_(kk, 2) = tris[off3 + 2];
//
//    if (request_tex_coord_) {
//      this->tri_listTex_(kk, 0) = trisTexCoords[off3];
//      this->tri_listTex_(kk, 1) = trisTexCoords[off3 + 1];
//      this->tri_listTex_(kk, 2) = trisTexCoords[off3 + 2];
//    }
//
//    if (request_normal_) {
//      this->tri_listNormal_(kk, 0) = trisNormals[off3];
//      this->tri_listNormal_(kk, 1) = trisNormals[off3 + 1];
//      this->tri_listNormal_(kk, 2) = trisNormals[off3 + 2];
//    }
//
//  }
//  albedo_ = color_;
//  build_fvLookUpTable();
//  init();
//  return;
//}

void cudaObjMesh::download_position(
    const pcl::gpu::DeviceArray<float3> device_position) {
  if (device_position.size() != n_verts_) {
    LOG(ERROR) << "Download position from device error!" << std::endl;
  }
  cudaStreamSynchronize(0);
  position_.transposeInPlace();
  device_position.download(reinterpret_cast<float3 *>(position_.data()));
  position_.transposeInPlace();
}

void cudaObjMesh::update_boundary() {
  cudaUpdateBoundary(cuda_is_boundary_, cuda_tri_list_, cudafvLookUpTable,
                     cudafBegin, cudafEnd);
}

void cudaObjMesh::estimate_geoposition() {
  geoposition_.resize(n_verts_, 3);
  std::vector<int> coorPoints;
  // coorPoints.push_back(0); coorPoints.push_back(1); coorPoints.push_back(2);
  coorPoints.push_back(8004);
  coorPoints.push_back(4638);
  coorPoints.push_back(11315);
  GraphUtil graphUtil(*this, coorPoints);
  graphUtil.FindNearest();
}
#endif  // USE_CUDA
