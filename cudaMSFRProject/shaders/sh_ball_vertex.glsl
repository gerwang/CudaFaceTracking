#version 430 core

uniform mat4 modelViewMatrix;
out vec4 normal_R;

void main() {
  normal_R = transpose(inverse(modelViewMatrix)) * vec4(gl_Normal, 0.0);
}