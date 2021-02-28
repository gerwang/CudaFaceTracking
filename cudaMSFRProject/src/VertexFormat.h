#pragma once

#include <glm/glm.hpp>

struct VertexFormatGraph {

  glm::vec3 position;
  glm::vec4 color;
  glm::vec3 normal;

  VertexFormatGraph(): position(0.0f), color(0.0f), normal(0.0f) {
  }

  VertexFormatGraph(const glm::vec3& iPos,
                    const glm::vec4& iColor,
                    const glm::vec3& iNormal
  ) {
    position = iPos;
    color = iColor;
    normal = iNormal;
  }
};

struct VertexFormatOpt{

	glm::vec3 position;
	glm::vec4 color;
	glm::vec3 normal;
  glm::vec3 weight;
  float     tri;

    VertexFormatOpt(): position(0.0f), color(0.0f), normal(0.0f), weight(0.0f), tri(0.0f) {
    }
    VertexFormatOpt(const glm::vec3 &iPos,
      const glm::vec4 &iColor,
      const glm::vec3 &iNormal,
      const glm::vec3 &iWeight,
      const float     &iTri
    ){
		position = iPos;
		color = iColor;
    normal = iNormal;
    weight = iWeight;
    tri = iTri;
	}
};

struct VertexFormatIsomap{

	glm::vec3 position;
	glm::vec2 tex;

	VertexFormatIsomap(const glm::vec3 &iPos, const glm::vec2 &iTex){
		position = iPos;
		tex = iTex;
	}

};
