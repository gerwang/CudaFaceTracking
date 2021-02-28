// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#pragma once
#include <GL/glew.h>
#include <stdio.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>
#include <GenModels_CUDA.h>
#include <Parameters.h>

#include <librealsense2\rsutil.h>
#include <iostream>
#include <sstream>
#include <string>
//////////////////////////////
// Basic Data Types         //
//////////////////////////////
// typedef struct rs2_intrinsics
//{
//  int           width;     /**< Width of the image in pixels */
//  int           height;    /**< Height of the image in pixels */
//  float         ppx;       /**< Horizontal coordinate of the principal point
//  of the image, as a pixel offset from the left edge */ float         ppy;
//  /**< Vertical coordinate of the principal point of the image, as a pixel
//  offset from the top edge */ float         fx;        /**< Focal length of
//  the image plane, as a multiple of pixel width */ float         fy; /**<
//  Focal length of the image plane, as a multiple of pixel height */ float
//  coeffs[5]; /**< Distortion coefficients, order: k1, k2, p1, p2, k3 */
//} rs2_intrinsics;
typedef rs2_intrinsics camera_intrinsics;

namespace renderer {
enum render_mode {
  NORMAL = 0,
  WHITE = 5,
  ALBEDO = 6,
  WEIGHT = 1,
  ICP = 4,
  DEPTH = 3,
  ERR = 2,
  RENDER = 7,
  SH_FALSE = 8
};
struct float3 {
  float x, y, z;
};
struct float2 {
  float x, y;
};
struct renderInfo {
  camera_intrinsics camera_;
  int width_, height_;
  render_mode rmode_ = WHITE;
  GLenum emode_;
  FrameParameters* parameters_;
};
}  // namespace renderer

inline glm::mat4 intrisic_to_projection(
    const camera_intrinsics& colorIntrinsic) {
  double point2pixel[4] = {colorIntrinsic.fx, colorIntrinsic.fy,
                           colorIntrinsic.ppx, colorIntrinsic.ppy};
  glm::mat4 projection = glm::scale(
      glm::mat4(1.0), glm::vec3(colorIntrinsic.fx, colorIntrinsic.fy, 1.0f));
  projection[3][3] = 0.0f;
  projection[2][2] = 1.0f;
  projection[2][3] = 1.0f;
  projection[3][2] = -0.001f;
  glm::mat4 proj_mat =
      glm::ortho(0.0f, (float)colorIntrinsic.width,
                 (float)colorIntrinsic.height, 0.0f, -10.0f, 10.0f) *
      glm::translate(glm::fmat4(1.0),
                     glm::fvec3(colorIntrinsic.ppx, colorIntrinsic.ppy, 0)) *
      projection;  // Parameters.projection_;
  projection =
      glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f),
                  glm::vec3(0.0f, -1.0f, 0.0f));
  return projection * proj_mat;
  // return glm::ortho(0.0f, (float)colorIntrinsic.width,
  // (float)colorIntrinsic.height, 0.0f, 0.0f, 100.0f) *
  // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f),
  // glm::vec3(0.0f, -1.0f, 0.0f));
}

struct rect {
  float x, y;
  float w, h;

  // Create new rect within original boundaries with give aspect ration
  rect adjust_ratio(renderer::float2 size) const {
    auto H = static_cast<float>(h), W = static_cast<float>(h) * size.x / size.y;
    if (W > w) {
      auto scale = w / W;
      W *= scale;
      H *= scale;
    }

    return {x + (w - W) / 2, y + (h - H) / 2, W, H};
  }
};

//////////////////////////////
// Simple font loading code //
//////////////////////////////

#include "./stb_easy_font.h"

inline void draw_text(int x, int y, const char* text) {
  char buffer[60000];  // ~300 chars
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(2, GL_FLOAT, 16, buffer);
  glDrawArrays(GL_QUADS, 0,
               4 * stb_easy_font_print((float)x, (float)(y - 7), (char*)text,
                                       nullptr, buffer, sizeof(buffer)));
  glDisableClientState(GL_VERTEX_ARRAY);
}

////////////////////////
// Image display code //
////////////////////////
class texture {
 public:
  void render(float* frame, const rect& r_, int width_, int height_,
              GLuint gl_handle_, int app_width, int app_height,
              const char* text) {
    render(frame, r_, width_, height_, gl_handle_, app_width, app_height);
    rect r = r_.adjust_ratio({float(width), float(height)});
    // draw_text((int)r.x + 15, (int)r.y + 20, text);
  }

  void renderShBall(cudaGenModels& model_list, const std::string& canvas,
                    const double* sh, GLuint shader,
                    const renderer::renderInfo& renderInfo,
                    bool useCameraView = false) const {
    const int winWidth = renderInfo.width_;
    const int winHeight = renderInfo.height_;
    glBindFramebuffer(GL_FRAMEBUFFER, model_list.GetModel(canvas));
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_RECTANGLE_ARB,
                              model_list.GameModelList[canvas].vbos[0], 0);
    GLenum buffers[] = {GL_COLOR_ATTACHMENT0_EXT};
    glDrawBuffers(1, buffers);
    glm::mat4 modelViewMat = glm::mat4(1.0f);
    if (useCameraView) {
      modelViewMat = renderInfo.parameters_->cameraParams.GetViewMatrix();
    }
    glViewport(0, 0, winWidth, winHeight);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "modelViewMatrix"), 1,
                       GL_FALSE, glm::value_ptr(modelViewMat));
    glUniform1dv(glGetUniformLocation(shader, "sh"), 9, sh);
    GLUquadricObj* obj = gluNewQuadric();
    gluQuadricNormals(obj, GLU_SMOOTH);
    gluSphere(obj, 0.1f, 200, 200);  // todo
    gluDeleteQuadric(obj);
    glUseProgram(0);
    glFinish();
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void renderModel(cudaGenModels& model_list, const std::string& canvas,
                   const std::string& model,
                   const renderer::renderInfo& render_info, GLuint shader,
                   bool is_use_camera_view = false) {
    int winWidth = model_list.GameModelList[canvas].width;
    int winHeight = model_list.GameModelList[canvas].height;
    float aspect = (float)winWidth / (float)winHeight;
    // std::cout << model_list.GetModel("canvas") << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, model_list.GetModel(canvas));
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_RECTANGLE_ARB,
                              model_list.GameModelList[canvas].vbos[0], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,
                              GL_TEXTURE_RECTANGLE_ARB,
                              model_list.GameModelList[canvas].vbos[1], 0);
    // glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT,
    // GL_TEXTURE_RECTANGLE_ARB, model_list.GameModelList[canvas].vbos[2], 0);
    glFramebufferTexture2DEXT(
        GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT4_EXT, GL_TEXTURE_RECTANGLE_ARB,
        model_list.GameModelList[canvas].vbos[8], 0);  // weight
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT5_EXT,
                              GL_TEXTURE_RECTANGLE_ARB,
                              model_list.GameModelList[canvas].vbos[5], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT6_EXT,
                              GL_TEXTURE_RECTANGLE_ARB,
                              model_list.GameModelList[canvas].vbos[6], 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT7_EXT,
                              GL_TEXTURE_RECTANGLE_ARB,
                              model_list.GameModelList[canvas].vbos[7], 0);

    GLenum buffers[] = {GL_COLOR_ATTACHMENT0_EXT,
                        GL_COLOR_ATTACHMENT1_EXT,
                        GL_NONE,
                        GL_NONE,
                        GL_COLOR_ATTACHMENT4_EXT,
                        GL_COLOR_ATTACHMENT5_EXT,
                        GL_COLOR_ATTACHMENT6_EXT,
                        GL_COLOR_ATTACHMENT7_EXT};
    // prevbug: misunderstand glDrawBuffers's meaning
    glDrawBuffers(8, buffers);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shader);
    glm::mat4 modelview_mat =
        glm::scale(glm::mat4(1.0), glm::vec3(1.0f, 1.0f, 1.0f));
    if (is_use_camera_view) {
      modelview_mat =
          render_info.parameters_->cameraParams
              .GetViewMatrix();  // *render_info.parameters_->UpdateModelMat();
    }
    glm::mat4 proj_mat = intrisic_to_projection(render_info.camera_);
    // double point2pixel[4] = { render_info.camera_.fx, render_info.camera_.fy,
    // render_info.camera_.ppx, render_info.camera_.ppy

    glViewport(0, 0, winWidth, winHeight);
    glUniformMatrix4fv(glGetUniformLocation(shader, "modelViewMatrix"), 1,
                       GL_FALSE, glm::value_ptr(modelview_mat));
    glUniformMatrix4fv(glGetUniformLocation(shader, "projectionMatrix"), 1,
                       GL_FALSE, glm::value_ptr(proj_mat));
    glUniform1f(glGetUniformLocation(shader, "colorMode"), render_info.rmode_);
    // glUniform1dv(glGetUniformLocation(shader, "point2pixel"), 4,
    // point2pixel);
    glUniform1dv(glGetUniformLocation(shader, "SHparas"), 27,
                 render_info.parameters_->modelParams.shParams);
    glUniform1dv(glGetUniformLocation(shader, "SHparas_false"), 27,
                 render_info.parameters_->modelParams.shParams_false);
    glUniform1dv(glGetUniformLocation(shader, "SH"), 9,
                 render_info.parameters_->SH);

    glBindVertexArray(model_list.GetModel(model));
    glDrawArrays(render_info.emode_, 0, 3 * model_list.nTriFrontList[model]);
    glBindVertexArray(0);

    glUseProgram(0);

    glFinish();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
  }
  void render(const rect& r_, int width_, int height_, GLuint gl_handle_,
              int app_width, int app_height, const char* text,
              bool upside_down = false) {
    width = width_;
    height = height_;

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glViewport(0, 0, app_width, app_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, app_width, app_height, 0, -1, +1);

    rect r = r_.adjust_ratio({float(width), float(height)});

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, gl_handle_);
    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBegin(GL_QUAD_STRIP);
    if (upside_down) {
      glTexCoord2f(0.f, 0.f);
      glVertex2f(r.x, r.y + r.h);
      glTexCoord2f(0.f, height);
      glVertex2f(r.x, r.y);
      glTexCoord2f(width, 0.f);
      glVertex2f(r.x + r.w, r.y + r.h);
      glTexCoord2f(width, height);
      glVertex2f(r.x + r.w, r.y);
    } else {
      glTexCoord2f(0.f, height);
      glVertex2f(r.x, r.y + r.h);
      glTexCoord2f(0.f, 0.0f);
      glVertex2f(r.x, r.y);
      glTexCoord2f(width, height);
      glVertex2f(r.x + r.w, r.y + r.h);
      glTexCoord2f(width, 0.0f);
      glVertex2f(r.x + r.w, r.y);
    }
    glEnd();
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glDisable(GL_BLEND);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    draw_text((int)r.x + 15, (int)r.y + 20, text);
  }

  void render(const rect& r_, int width_, int height_, GLuint gl_handle_,
              int app_width, int app_height) {
    width = width_;
    height = height_;

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, app_width, app_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, app_width, app_height, 0, -1, +1);

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, gl_handle_);
    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBegin(GL_QUAD_STRIP);
    rect r = r_.adjust_ratio({float(width), float(height)});
    glTexCoord2f(0.f, 0.f);
    glVertex2f(r.x, r.y + r.h);
    glTexCoord2f(0.f, height);
    glVertex2f(r.x, r.y);
    glTexCoord2f(width, 0.f);
    glVertex2f(r.x + r.w, r.y + r.h);
    glTexCoord2f(width, height);
    glVertex2f(r.x + r.w, r.y);
    glEnd();
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
  }

  void render(unsigned char* frame, const rect& r_, int width_, int height_,
              int app_width, int app_height) {
    width = width_;
    height = height_;
    glDisable(GL_DEPTH_TEST);
    glBindTexture(GL_TEXTURE_2D, gl_handle);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, frame);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, app_width, app_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, app_width, app_height, 0, -1, +1);

    glBindTexture(GL_TEXTURE_2D, gl_handle);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUAD_STRIP);
    rect r = r_.adjust_ratio({float(width), float(height)});
    glTexCoord2f(0.f, 0.f);
    glVertex2f(r.x, r.y + r.h);
    glTexCoord2f(0.f, 1.f);
    glVertex2f(r.x, r.y);
    glTexCoord2f(1.f, 0.f);
    glVertex2f(r.x + r.w, r.y + r.h);
    glTexCoord2f(1.f, 1.f);
    glVertex2f(r.x + r.w, r.y);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void render(float* frame, const rect& r_, int width_, int height_,
              GLuint gl_handle_, int app_width, int app_height) {
    width = width_;
    height = height_;
    glDisable(GL_DEPTH_TEST);
    glBindTexture(GL_TEXTURE_2D, gl_handle_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA,
                 GL_FLOAT, frame);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, app_width, app_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, app_width, app_height, 0, -1, +1);

    glBindTexture(GL_TEXTURE_2D, gl_handle_);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUAD_STRIP);
    rect r = r_.adjust_ratio({float(width), float(height)});
    glTexCoord2f(0.f, 0.f);
    glVertex2f(r.x, r.y + r.h);
    glTexCoord2f(0.f, 1.f);
    glVertex2f(r.x, r.y);
    glTexCoord2f(1.f, 0.f);
    glVertex2f(r.x + r.w, r.y + r.h);
    glTexCoord2f(1.f, 1.f);
    glVertex2f(r.x + r.w, r.y);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void show(const rect& r, float app_width, float app_height,
            const char* text) const {
    if (!gl_handle) return;

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, app_width, app_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, app_width, app_height, 0, -1, +1);

    glBindTexture(GL_TEXTURE_2D, gl_handle);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUAD_STRIP);
    glTexCoord2f(0.f, 0.f);
    glVertex2f(r.x, r.y + r.h);
    glTexCoord2f(0.f, 1.f);
    glVertex2f(r.x, r.y);
    glTexCoord2f(1.f, 0.f);
    glVertex2f(r.x + r.w, r.y + r.h);
    glTexCoord2f(1.f, 1.f);
    glVertex2f(r.x + r.w, r.y);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    // draw_text((int)r.x + 15, (int)r.y + 20, text);
  }

  GLuint get_gl_handle() { return gl_handle; }

  void show(const rect& r, float app_width, float app_height) const {
    if (!gl_handle) return;

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, app_width, app_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, app_width, app_height, 0, -1, +1);

    glBindTexture(GL_TEXTURE_2D, gl_handle);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUAD_STRIP);
    glTexCoord2f(0.f, 0.f);
    glVertex2f(r.x, r.y + r.h);
    glTexCoord2f(0.f, 1.f);
    glVertex2f(r.x, r.y);
    glTexCoord2f(1.f, 0.f);
    glVertex2f(r.x + r.w, r.y + r.h);
    glTexCoord2f(1.f, 1.f);
    glVertex2f(r.x + r.w, r.y);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    // draw_text((int)r.x + 15, (int)r.y + 20, rs2_stream_to_string(stream));
  }

 private:
  GLuint gl_handle = 0;
  int width = 0;
  int height = 0;
};

class window {
 public:
  static constexpr float cameraSpeed = 0.05f;
  std::function<void(bool)> on_left_mouse = [](bool) {};
  std::function<void(double, double)> on_mouse_scroll = [](double, double) {};
  std::function<void(double, double)> on_mouse_move = [](double, double) {};
  std::function<void(int)> on_key_release = [](int) {};
  std::function<void(float)> ON_PRESS_W;
  std::function<void(float)> ON_PRESS_A;
  std::function<void(float)> ON_PRESS_S;
  std::function<void(float)> ON_PRESS_D;
  std::function<void()> ON_PRESS_R;
  std::function<void()> ON_PRESS_ENTER;
  std::function<void()> ON_PRESS_SPACE;
  std::function<void()> ON_PRESS_ESCAPE;

  window(int width, int height, const char* title)
      : _width(width), _height(height) {
    lastX = _width / 2.0f;
    lastY = _height / 2.0f;
    glfwInit();
    win = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!win)
      throw std::runtime_error(
          "Could not open OpenGL window, please check your graphic drivers or "
          "use the textual SDK tools");
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);  // Enable vsync
    glewInit();

    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard
    // Controls io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable
    // Gamepad Controls

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    const char* glsl_version = "#version 430";
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
    ImGui::StyleColorsDark();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glfwSetWindowUserPointer(win, this);
    glfwSetMouseButtonCallback(
        win, [](GLFWwindow* win, int button, int action, int mods) {
          auto s = (window*)glfwGetWindowUserPointer(win);
          if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
              s->mouse_left_pressed = true;
              s->firstMouse = true;
            }
          }
          if (action = GLFW_RELEASE) {
            s->mouse_left_pressed = false;
          }
        });

    //     glfwSetScrollCallback(win, [](GLFWwindow * win, double xoffset,
    //     double yoffset)
    //     {
    //         auto s = (window*)glfwGetWindowUserPointer(win);
    //         s->on_mouse_scroll(xoffset, yoffset);
    //     });

    glfwSetCursorPosCallback(
        win, [](GLFWwindow* win, double xpos, double ypos) {
          auto s = (window*)glfwGetWindowUserPointer(win);
          if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT)) {
            if (s->firstMouse) {
              s->lastX = xpos;
              s->lastY = ypos;
              s->firstMouse = false;
            }

            float xoffset = xpos - s->lastX;
            float yoffset =
                s->lastY -
                ypos;  // reversed since y-coordinates go from bottom to top

            s->lastX = xpos;
            s->lastY = ypos;
            s->on_mouse_move(xoffset, yoffset);
          }
        });

    glfwSetKeyCallback(
        win, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
          auto s = (window*)glfwGetWindowUserPointer(win);
          if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_R) {
              s->ON_PRESS_R();
            } else if (key == GLFW_KEY_ENTER) {
              s->ON_PRESS_ENTER();
            } else if (key == GLFW_KEY_SPACE) {
              s->ON_PRESS_SPACE();
            } else if (key == GLFW_KEY_ESCAPE) {
              s->ON_PRESS_ESCAPE();
            }
          }
          if (0 == action)  // on key release
          {
            s->on_key_release(key);
          }
        });
  }

  float width() const { return float(_width); }
  float height() const { return float(_height); }

  operator bool() {
    float currentFrame = glfwGetTime();
    delay_time = currentFrame - lastFrame;
    processInput();
    // glPopMatrix();
    glfwSwapBuffers(win);

    auto res = !glfwWindowShouldClose(win);

    glfwPollEvents();
    glfwGetFramebufferSize(win, &_width, &_height);

    // Clear the framebuffer
    glClearColor(150.0f / 255.0f, 150.0f / 255.0f, 150.0f / 255.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // glViewport(0, 0, _width, _height);

    // Draw the images
    // glPushMatrix();
    glfwGetWindowSize(win, &_width, &_height);
    // glOrtho(0, _width, _height, 0, -1, +1);

    return res;
  }

  void processInput() {
    auto s = (window*)glfwGetWindowUserPointer(win);
    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) s->ON_PRESS_W(delay_time);
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) s->ON_PRESS_S(delay_time);
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) s->ON_PRESS_A(delay_time);
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) s->ON_PRESS_D(delay_time);
  }
  ~window() {
    glfwDestroyWindow(win);
    glfwTerminate();
  }

  operator GLFWwindow*() { return win; }

 private:
  GLFWwindow* win;
  int _width, _height;
  float delay_time = 0.0f, lastFrame = 0.0f;
  float lastX;
  float lastY;
  bool firstMouse = false;
  bool mouse_left_pressed = false;
};

// Struct for managing rotation of pointcloud view
struct glfw_state {
  glfw_state()
      : yaw(15.0),
        pitch(15.0),
        last_x(0.0),
        last_y(0.0),
        ml(false),
        offset_x(2.f),
        offset_y(2.f),
        tex() {}
  double yaw;
  double pitch;
  double last_x;
  double last_y;
  bool ml;
  float offset_x;
  float offset_y;
  texture tex;
};
