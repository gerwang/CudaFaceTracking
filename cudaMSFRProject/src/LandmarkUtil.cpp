#include <LandmarkUtil.h>
#include <fstream>


void LandmarkUtil::LoadLandmark()
{
  std::string path(LANDMARK_DEFAULT);
  std::ifstream File(path);
  File >> landmarkNum >> contourLandmarkNum;

  landmarkIndex.resize(landmarkNum);
  for (int i = 0; i < landmarkNum; i++) {
    File >> landmarkIndex[i].x >> landmarkIndex[i].y;
  }
  contourLandmarkIndex.resize(contourLandmarkNum);
  for (int i = 0; i < contourLandmarkNum; ++i)
  {
    File >> contourLandmarkIndex[i].x;
    contourLandmarkIndex[i].y = -1;
  }
  File.close();
  std::sort(contourLandmarkIndex.begin(), contourLandmarkIndex.end(),
    [](int2 a, int2 b)
  {
    return a.x < b.x;
  });
}
