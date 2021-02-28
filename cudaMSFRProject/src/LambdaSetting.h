#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#define USE_IMGUI
#ifdef USE_IMGUI
#include "imgui.h"
//#include "imgui_impl_glfw.h"
//#include "imgui_impl_opengl3.h"
#endif

typedef std::map<std::string, std::map<std::string, float>> lambdaList;

class LambdaSetting {
 public:
  ~LambdaSetting() { saveLambdaSetting(); }

  LambdaSetting(const LambdaSetting&) = delete;

  LambdaSetting& operator=(const LambdaSetting&) = delete;

  static LambdaSetting& get_instance() {
    static LambdaSetting instance;
    return instance;
  }
  void saveLambdaSetting() {
    for (auto& mode : lambdas) {
      auto mode_name = mode.first.c_str();
      rapidjson::Value& mode_value = document[mode_name];
      for (auto& lambda : mode.second) {
        rapidjson::Value& lambda_value = mode_value[lambda.first.c_str()];
        lambda_value.SetFloat(lambda.second);
        std::cout << mode_name << ": " << lambda.first << " "
                  << lambda_value.GetFloat() << std::endl;
      }
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    std::ofstream output_file("./new_lambda_setting.json");
    output_file << buffer.GetString() << std::endl;
  }

  void saveLambdaSetting(const std::string output_dir) {
    for (auto& mode : lambdas) {
      auto mode_name = mode.first.c_str();
      rapidjson::Value& mode_value = document[mode_name];
      for (auto& lambda : mode.second) {
        rapidjson::Value& lambda_value = mode_value[lambda.first.c_str()];
        lambda_value.SetFloat(lambda.second);
        std::cout << mode_name << ": " << lambda.first << " "
                  << lambda_value.GetFloat() << std::endl;
      }
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    std::ofstream output_file(output_dir + "/lambda_setting.json");
    output_file << buffer.GetString() << std::endl;
  }

  float& operator()(const std::string& mode_name,
                    const std::string& lambda_name) {
    return lambdas.at(mode_name).at(lambda_name);
  }

  void show() {
#ifndef USE_IMGUI
    for (auto& mode : lambdas) {
      auto mode_name = mode.first.c_str();
      for (auto& lambda : mode.second) {
        std::cout << mode_name << ": " << lambda.first << " " << lambda.second
                  << std::endl;
      }
    }
#else
    if (ImGui::TreeNode("Lambda Setting")) {
      for (auto& mode : lambdas) {
        auto mode_name = mode.first;
        if (ImGui::TreeNode(mode_name.c_str())) {
          for (auto& lambda : mode.second) {
            ImGui::DragFloat((mode_name + " " + lambda.first).c_str(),
                             &lambda.second, 0.01);
          }
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
#endif
  }

 private:
  rapidjson::Document document;
  lambdaList lambdas;
  LambdaSetting() {
    std::cout << "Loading lambda setting." << std::endl;
    std::ifstream t("./lambda_setting.json");
    std::string str((std::istreambuf_iterator<char>(t)),
                    std::istreambuf_iterator<char>());
    document.Parse(str.c_str());

    // Iterating object members
    static const char* kTypeNames[] = {"Null",  "False",  "True",  "Object",
                                       "Array", "String", "Number"};

    for (rapidjson::Value::ConstMemberIterator itr = document.MemberBegin();
         itr != document.MemberEnd(); ++itr) {
      if (kTypeNames[itr->value.GetType()]) {
        lambdas[itr->name.GetString()] = std::map<std::string, float>();
        auto& lambdas_i = lambdas[itr->name.GetString()];
        for (rapidjson::Value::ConstMemberIterator itr_i =
                 itr->value.MemberBegin();
             itr_i != itr->value.MemberEnd(); ++itr_i) {
          lambdas_i[itr_i->name.GetString()] = itr_i->value.GetFloat();
          printf("  Value of member %s is %f\n", itr_i->name.GetString(),
                 itr_i->value.GetFloat());
        }
      }
    }
  }
};