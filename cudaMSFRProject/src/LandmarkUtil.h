#pragma once

#include <vector>
#include <string>
#include "Common.h"
#include "ObjMesh.h"
#include "BaselModel.h"
#include "Parameters.h"
#include "pcl\gpu\containers\device_array.h"


/// A tool for Landmarks
class LandmarkUtil {
public:
  int landmarkNum;
  int contourLandmarkNum;
  LandmarkUtil() { LoadLandmark(); }
  std::vector<int2> landmarkIndex, contourLandmarkIndex;
    void LoadLandmark();
};