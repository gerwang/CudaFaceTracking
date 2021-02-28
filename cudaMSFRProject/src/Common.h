#pragma once
#define LAPLACIAN_SMOOTH
#define CUDA_GET_LAST_ERROR_AND_SYNC 0

#define USE_FACESHIFT_EXP
#define CORRECTIVE_THRESHOLD 0.0f
#define ONE_MINUS_CORRECTIVE_THRESHOLD 1.0f
#define NUM_IDENTITY_3DMM 199
#define NUM_ALBEDO_3DMM 199
#define FACESHAPE_PNTNUM 78
#define NUM_CONTOUR_STRIP 96
#define NUM_COLORTRANSFORM_3DMM 7
#define NUM_EXPRESSION_PCA_3DMM 29
#define NUM_EXPRESSION_INTEL_3DMM 45
#define NUM_3DMM_VERTEX 34508
#define NUM_3DMM_VERTEX_REMAIN 4508
#define NUM_3DMM_VERTEX_BLOCK 15000
#define NUM_3DMM_FACE 68256
#define ONLINE_SAVE_SEQ 0

#define ALBEDO_TEXTURE_RES 512

#ifdef USE_EXPR_PCA
#define NUM_EXPRESSION_3DMM 29
#else
#define NUM_EXPRESSION_3DMM 80
#endif

#define MORPH_DEFAULT_RIGID_INDEX "data/model/rigidIdx.txt"
#define MORPH_ARCSOFT_CONTOUR_LANDMARK "data/model/contour_landmark.txt"
#define MORPH_DEFAULT_INTEL_MODEL_PATH "data/model/Model.mat"
#define MORPH_DEFAULT_INTEL_TEMPLATE_MODEL "data/model/meanUV.obj"
#define MORPH_DEFAULT_INTEL_INDEX3D "data/model/index.txt"
#define LANDMARK_DEFAULT "data/model/landmark.txt"
#define FUSIONRESULTS_DIR "fusionResults"
#define NUM_FACE_SEG 4

//#define TEST_MODE
#define OPTIMIZER_FIXA
#define USE_CUDA

#ifndef M_PI
#define M_PI 3.14159265358979323846

namespace msfr {
struct intrinsics {
  float fx, fy, cx, cy;
  intrinsics() {}
  intrinsics(float fx_, float fy_, float cx_, float cy_)
      : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

  intrinsics operator()(int level_index) const {
    int div = 1 << level_index;
    return (intrinsics(fx / div, fy / div, cx / div, cy / div));
  }
};
}  // namespace msfr

#endif