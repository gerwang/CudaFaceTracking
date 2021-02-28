#include "GenModels_CUDA.h"
#ifdef USE_CUDA

cudaGenModels::~cudaGenModels() {
  std::map<std::string, Model>::iterator it;
  for (it = GameModelList.begin(); it != GameModelList.end(); ++it) {
    // delete VAO and VBOs (if many)
    unsigned int* p = &it->second.vao;
    glDeleteVertexArrays(1, p);
    glDeleteBuffers(it->second.vbos.size(), &it->second.vbos[0]);
    it->second.vbos.clear();
  }
  GameModelList.clear();
}

void cudaGenModels::createMesh(const std::string& gameModelName,
                               const cudaObjMesh& ref_model) {
  Model myModel;
  myModel.vao;
  myModel.cudaResourcesBuf.resize(1);
  myModel.vbos.resize(1);

  glGenVertexArrays(1, &myModel.vao);
  if (myModel.vao != NULL) glBindVertexArray(myModel.vao);
  std::vector<VertexFormatOpt> vertices(ref_model.n_tri_ * 3);
  nTriFrontList[gameModelName] = ref_model.n_tri_;

  glGenBuffers(1, &myModel.vbos[0]);
  glBindBuffer(GL_ARRAY_BUFFER, myModel.vbos[0]);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(VertexFormatOpt),
               &vertices[0], GL_DYNAMIC_COPY);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFormatOpt),
                        (void*)0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexFormatOpt),
                        (void*)12);

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFormatOpt),
                        (void*)28);

  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFormatOpt),
                        reinterpret_cast<void*>(40));

  glEnableVertexAttribArray(4);
  glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(VertexFormatOpt),
                        reinterpret_cast<void*>(52));

  /*
  glm::vec3 position;
  glm::vec4 albedo;
  glm::vec3 normal;
  glm::vec3 weight;
  float     tri;
  */

  cudaSafeCall(cudaGraphicsGLRegisterBuffer(&myModel.cudaResourcesBuf[0],
                                            myModel.vbos[0],
                                            cudaGraphicsRegisterFlagsNone));

  GameModelList[gameModelName] = std::move(myModel);
  updateMesh(gameModelName, ref_model);
}

void cudaGenModels::updateMesh(const std::string& gameModelName,
                               const pcl::gpu::DeviceArray<float3> position,
                               const pcl::gpu::DeviceArray<float3> color,
                               const pcl::gpu::DeviceArray<float3> normal,
                               const pcl::gpu::DeviceArray<int3> tri_list) {
  size_t size;
  float* verticsInfo;
  cudaSafeCall(cudaGraphicsMapResources(
      1, &GameModelList[gameModelName].cudaResourcesBuf[0], 0));
  cudaSafeCall(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&verticsInfo), &size,
      GameModelList[gameModelName].cudaResourcesBuf[0]));
  pcl::gpu::DeviceArray<float> vInfoArray(verticsInfo, size / sizeof(float));
  cudaUpdateMesh(vInfoArray, position, color, normal, tri_list);
  cudaSafeCall(cudaGraphicsUnmapResources(
      1, &GameModelList[gameModelName].cudaResourcesBuf[0], 0));
}

void cudaGenModels::updateMesh2(const std::string& gameModelName,
                                const pcl::gpu::DeviceArray<float3> position,
                                const pcl::gpu::DeviceArray<float3> color,
                                const pcl::gpu::DeviceArray<float3> normal,
                                const pcl::gpu::DeviceArray<int2> fvLookUpTable,
                                const pcl::gpu::DeviceArray<int1> fbegin,
                                const pcl::gpu::DeviceArray<int1> fend,
                                const pcl::gpu::DeviceArray<int> fv_idx) {
  size_t size;
  float* verticsInfo;
  cudaSafeCall(cudaGraphicsMapResources(
      1, &GameModelList[gameModelName].cudaResourcesBuf[0], 0));
  cudaSafeCall(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&verticsInfo), &size,
      GameModelList[gameModelName].cudaResourcesBuf[0]));
  pcl::gpu::DeviceArray<float> vInfoArray(verticsInfo, size / sizeof(float));
  cudaUpdateMesh2(vInfoArray, position, color, normal, fvLookUpTable, fbegin,
                  fend, fv_idx);
  cudaSafeCall(cudaGraphicsUnmapResources(
      1, &GameModelList[gameModelName].cudaResourcesBuf[0], 0));
}

void cudaGenModels::updateMesh(const std::string& gameModelName,
                               const cudaObjMesh& ref_model) {
  size_t size;
  float* verticsInfo;
  cudaSafeCall(cudaGraphicsMapResources(
      1, &GameModelList[gameModelName].cudaResourcesBuf[0], 0));
  cudaSafeCall(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&verticsInfo), &size,
      GameModelList[gameModelName].cudaResourcesBuf[0]));
  pcl::gpu::DeviceArray<float> vInfoArray(verticsInfo, size / sizeof(float));
  cudaUpdateMesh(vInfoArray, ref_model.position(), ref_model.cuda_color_,
                 ref_model.normal(), ref_model.cuda_tri_list_);
  cudaSafeCall(cudaGraphicsUnmapResources(
      1, &GameModelList[gameModelName].cudaResourcesBuf[0], 0));
}

void cudaGenModels::createImage(const std::string& imageName, const int width,
                                const int height) {
  Model myModel;
  myModel.cudaResourcesBuf.resize(10);
  myModel.vbos.resize(11);
  myModel.width = width;
  myModel.height = height;

  glGenTextures(1, &myModel.vbos[0]);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[0]);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  // glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R32F, width, height, 0,
  // GL_RED, GL_FLOAT, 0);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0,
               GL_RGBA, GL_FLOAT, 0);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[0], myModel.vbos[0], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[1]);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[1]);
  // glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
  // GL_LINEAR); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
  // GL_LINEAR); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S,
  // GL_CLAMP); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T,
  // GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R32F, width, height, 0, GL_RED,
               GL_FLOAT, 0);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[1], myModel.vbos[1], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[2]);  // input color
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[2]);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  // glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R32F, width, height, 0,
  // GL_RED, GL_FLOAT, 0);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0,
               GL_RGBA, GL_FLOAT, 0);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[2], myModel.vbos[2], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[3]);  // input depth map
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[3]);
  // glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
  // GL_LINEAR); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
  // GL_LINEAR); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S,
  // GL_CLAMP); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T,
  // GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R32F, width, height, 0, GL_RED,
               GL_FLOAT, 0);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[3], myModel.vbos[3], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[4]);  // input aligned depth map
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[4]);
  // glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
  // GL_LINEAR); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
  // GL_LINEAR); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S,
  // GL_CLAMP); glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T,
  // GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R32F, width, height, 0, GL_RED,
               GL_FLOAT, 0);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[4], myModel.vbos[4], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[5]);  // tri
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[5]);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0,
               GL_RGBA, GL_FLOAT, nullptr);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[5], myModel.vbos[5], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[6]);  // normal
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[6]);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0,
               GL_RGBA, GL_FLOAT, nullptr);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[6], myModel.vbos[6], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[7]);  // debug albedo
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[7]);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0,
               GL_RGBA, GL_FLOAT, nullptr);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[7], myModel.vbos[7], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[8]);  // weight
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[8]);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0,
               GL_RGBA, GL_FLOAT, nullptr);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[8], myModel.vbos[8], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenTextures(1, &myModel.vbos[9]);  // backup input color
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[9]);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  // glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R32F, width, height, 0,
  // GL_RED, GL_FLOAT, 0);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width, height, 0,
               GL_RGBA, GL_FLOAT, 0);
  cudaSafeCall(cudaGraphicsGLRegisterImage(
      &myModel.cudaResourcesBuf[9], myModel.vbos[9], GL_TEXTURE_RECTANGLE_ARB,
      cudaGraphicsRegisterFlagsWriteDiscard));

  glGenFramebuffersEXT(1, &myModel.vao);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, myModel.vao);
  glGenRenderbuffersEXT(1, &myModel.vbos[myModel.vbos.size() - 1]);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT,
                        myModel.vbos[myModel.vbos.size() - 1]);
  glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32F, width,
                           height);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                            GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[0], 0);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,
                            GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[1], 0);
  // glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT,
  // GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[2], 0);
  // glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT,
  // GL_TEXTURE_RECTANGLE_ARB, myModel.vbos[3], 0);
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                               GL_RENDERBUFFER_EXT,
                               myModel.vbos[myModel.vbos.size() - 1]);
  GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
  if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
    printf("glCheckFramebufferStatusEXT failed!\n");
    exit(1);
  }
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

  myModel.size = width * height;
  GameModelList[imageName] = std::move(myModel);
}

void cudaGenModels::getImage(const std::string& imageName, const int id,
                             pcl::gpu::DeviceArray<float>& dst) {
  cudaArray* imageAdd;
  cudaSafeCall(cudaGraphicsMapResources(
      1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
  cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(
      &imageAdd, GameModelList[imageName].cudaResourcesBuf[id], 0, 0));
  cudaSafeCall(cudaMemcpyFromArray(
      reinterpret_cast<void*>(dst.ptr()), imageAdd, 0, 0,
      GameModelList[imageName].size * dst.elem_size, cudaMemcpyDeviceToDevice));
  cudaSafeCall(cudaGraphicsUnmapResources(
      1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
}

void cudaGenModels::getImage(const std::string& imageName, const int id,
                             pcl::gpu::DeviceArray2D<float>& dst) {
  cudaArray* imageAdd;
  cudaSafeCall(cudaGraphicsMapResources(
      1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
  cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(
      &imageAdd, GameModelList[imageName].cudaResourcesBuf[id], 0, 0));
  cudaSafeCall(cudaMemcpyFromArray(
      reinterpret_cast<void*>(dst.ptr()), imageAdd, 0, 0,
      GameModelList[imageName].size * dst.elem_size, cudaMemcpyDeviceToDevice));
  cudaSafeCall(cudaGraphicsUnmapResources(
      1, &GameModelList[imageName].cudaResourcesBuf[id], 0));
}

void cudaGenModels::DeleteModel(const std::string& gameModelName) {
  Model model = GameModelList[gameModelName];
  unsigned int p = model.vao;
  glDeleteVertexArrays(1, &p);
  glDeleteBuffers(model.vbos.size(), &model.vbos[0]);
  model.vbos.clear();
  GameModelList.erase(gameModelName);
}

unsigned int cudaGenModels::GetModel(const std::string& gameModelName) {
  return GameModelList[gameModelName].vao;
}

#endif  // USE_CUDA
