#version 430 core

in vec4 normal_R;

layout(location = 0) out vec4 outColor;

uniform double SH[9];

double calcShLight(double x, double y, double z, double* sh) {
  double res = sh[0];
  res += sh[1] * z;
  res += sh[2] * y;
  res += sh[3] * x;
  res += sh[4] * (2 * z * z - x * x - y * y);
  res += sh[5] * y * z;
  res += sh[6] * x * z;
  res += sh[7] * x * y;
  res += sh[8] * (x * x - y * y);
  return res;
}

void main() {
  vec3 normal = normalize(normal_R);
  double light = calcShLight(normal.x, normal.y, normal.z, SH);
  outColor = vec4(light, light, light, 1);
}