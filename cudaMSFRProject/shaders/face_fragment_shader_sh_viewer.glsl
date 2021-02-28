#version 430 core

in vec3 fs_position;
in vec4 fs_albedo;
in vec3 fs_normal;
in vec4 fs_weight;
in vec4 fs_tri;

uniform double SHparas[27];
uniform double SHparas_false[27];
uniform double SH[9];
uniform double CT[7];

vec3 CalcSHLight(vec3 rgb, vec3 normal);
vec3 CalcSHLight_false(vec3 rgb, vec3 normal);
vec3 CalcDirLight(vec3 i_rgb, vec3 normal, vec3 viewDir, vec3 lightDir_);
vec3 jet_colormap(float v);
uniform sampler2D texture1;
uniform float texMapping;
uniform float colorMode;

layout(location = 0) out vec4 out_color;
layout(location = 1) out float out_depth;
layout(location = 2) out float out_depth_2;
layout(location = 4) out vec4 out_weight;
layout(location = 5) out vec4 out_tri;
layout(location = 6) out vec4 out_normal;
layout(location = 7) out vec4 out_albedo;

vec3 lightPoint[3] = {vec3(-0.4f, 0.1f, 0.4f), vec3(0.4f, 0.1f, 0.4f),
                      vec3(0.0f, 0.3f, 1.0f)};
float lightPointCoeff[3] = {0.35f, 0.35f, 0.6f};

const float MAX_ERROR = 10e-2f;

void main() {
  vec3 normal = normalize(fs_normal);
  // normal.z=-normal.z;
  vec3 viewDir = normalize(vec3(0, 0, 0) - fs_position);
  vec3 mRGB;
  vec3 result;

  if (colorMode == 0.f) {  /// NOMRAL
    // mRGB = fs_albedo.xyz;
    // result = CalcSHLight(mRGB, normal);
    // result = CalcSHLight_false(mRGB, normal) / result;
    result = (-normal + vec3(1.0f, 1.0f, 1.0f)) / 2;
  } else if (colorMode == 1.f) {  /// SH Lighting show
    mRGB = vec3(0.6, 0.6, 0.6);
    // result = CalcDirLight(mRGB, normal, viewDir);
    result = CalcSHLight(mRGB, normal);
  } else if (colorMode == 2.f) {  /// ERROR
    if (fs_albedo.x < 0.0f)
      out_color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    else {
      result = jet_colormap(fs_albedo.z / MAX_ERROR);
      out_color = vec4(CalcSHLight(result, normal), 1.0f);
    }
  } else if (colorMode == 3.f) {  /// Depth
    result = vec3(10 * (fs_position.z - 0.35f), 0.0f, 0.0f);
  } else if (colorMode == 4.f) {  /// ICP
    if (fs_albedo.x < 0.0f)
      mRGB = vec3(0.0f, 1.0f, 0.0f);
    else if (fs_albedo.y < 0.0f)  /// pos.z < corr.z
      mRGB = vec3(1.0f, 0.0f, 0.0f);
    else
      mRGB = vec3(0.0f, 0.0f, 1.0f);  /// pos.z >= corr.z
    result = mRGB;
    // result = fs_albedo.xyz;//CalcDirLight(mRGB, normal, viewDir);
  } else if (colorMode == 5.f) {  /// White
    mRGB = vec3(0.75, 0.75, 0.75);
    // mRGB = vec3(fs_albedo.r, fs_albedo.g, fs_albedo.b);
    // result = 0.25 * CalcDirLight(mRGB, normal, viewDir, vec3(-1.0f,
    // 1.0f, 2.5f)) + 0.25 * CalcDirLight(mRGB, normal, viewDir, vec3(1.0f,
    // 1.0f, 2.5f)) + result = 0.35 * CalcDirLight(mRGB, normal, viewDir,
    // vec3(-0.7f, 0.1f, -0.4f)) + 0.35 * CalcDirLight(mRGB, normal, viewDir,
    // vec3(0.7f, 0.1f, -0.4f)) +
    // 0.6 * CalcDirLight(mRGB, normal, viewDir, vec3(0.0f, 0.3f, 1.0f)) ;
    result = lightPointCoeff[0] *
             CalcDirLight(mRGB, normal, viewDir, lightPoint[0] - fs_position);
    result += lightPointCoeff[1] *
              CalcDirLight(mRGB, normal, viewDir, lightPoint[1] - fs_position);
    result += lightPointCoeff[2] *
              CalcDirLight(mRGB, normal, viewDir, lightPoint[2] - fs_position);
    // result =
    // vec3(3*(fs_position.z-0.4),3*(fs_position.z-0.4),3*(fs_position.z-0.4));
    // result = vec3(fs_position.x/640,fs_position.y/720,3*(fs_position.z-0.4));
    // result = vec3(fs_position.x/640,0,0);
  } else if (colorMode == 6.f) {  // Albedo, no lighting
    result = vec3(fs_albedo.r, fs_albedo.g, fs_albedo.b);
  } else if (colorMode == 7.f) {  // Render albedo with light
    mRGB = vec3(fs_albedo.r, fs_albedo.g, fs_albedo.b);
    result = CalcSHLight(mRGB, normal);
  } else if (colorMode == 8.f) {
    mRGB = vec3(0.6, 0.6, 0.6);  // * 0.95;
    // result = CalcDirLight(mRGB, normal, viewDir);
    result = CalcSHLight_false(mRGB, normal);
  }

  // color transfer
  // vec3 offset = vec3(CT[3], CT[4], CT[5]);
  // mat3 CTM = mat3(CT[0] * (CT[6] + 0.3*(1 - CT[6])), CT[1] * (0.3*(1 -
  // CT[6])), CT[2] * (0.3*(1 - CT[6])), CT[0] * (0.59*(1 - CT[6])), CT[1] *
  // (CT[6] + 0.59*(1 - CT[6])), CT[2] * (0.59*(1 - CT[6])), CT[0] * (0.11*(1 -
  // CT[6])), CT[1] * (0.11*(1 - CT[6])), CT[2] * (CT[6] + 0.11*(1 - CT[6])));
  // result = CTM*result + offset;

  if (colorMode != 2.0f) {
    out_color = vec4(result, 1.0);
  }
  // out_color = vec4((fs_position.z - 0.35)*5, 0.0, 0.0, 1.0);
  out_depth = fs_position.z;
  out_depth_2 = fs_position.z;

  out_weight = fs_weight;
  out_tri = fs_tri;
  out_normal = vec4(fs_normal, 1.0);

  out_albedo = fs_albedo;
}

vec3 CalcDirLight(vec3 rgb, vec3 normal, vec3 viewDir, vec3 lightDir_) {
  vec3 lightDir = -normalize(lightDir_);
  float diff = max(dot(normal, lightDir), 0.0);
  // specular shading
  vec3 reflectDir = normalize(reflect(lightDir, normal));
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 20.f);
  float spec1 = pow(max(dot(viewDir, reflectDir), 0.0), 5.f);
  // ambient & diffuse
  vec3 ambient = vec3(0.1f, 0.1f, 0.1f) * rgb;
  vec3 diffuse = vec3(0.7f, 0.7f, 0.7f) * diff * rgb;
  vec3 specular = vec3(0.5f, 0.5f, 0.5f) * (0.4f * spec + 0.4f * spec1);
  return (ambient + diffuse + specular);
}

vec3 CalcSHLight(vec3 i_rgb, vec3 normal) {
  double pi_ = 3.1415926;
  vec3 o_rgb =
      vec3(SH[0], SH[0], SH[0]) * vec3(SHparas[0], SHparas[1], SHparas[2]);
  o_rgb = o_rgb + vec3(SH[1] * normal.z, SH[1] * normal.z, SH[1] * normal.z) *
                      vec3(SHparas[3], SHparas[4], SHparas[5]);
  o_rgb = o_rgb + vec3(SH[2] * normal.y, SH[2] * normal.y, SH[2] * normal.y) *
                      vec3(SHparas[6], SHparas[7], SHparas[8]);
  o_rgb = o_rgb + vec3(SH[3] * normal.x, SH[3] * normal.x, SH[3] * normal.x) *
                      vec3(SHparas[9], SHparas[10], SHparas[11]);
  o_rgb = o_rgb + vec3(SH[4] * (3 * normal.z * normal.z - 1),
                       SH[4] * (3 * normal.z * normal.z - 1),
                       SH[4] * (3 * normal.z * normal.z - 1)) *
                      vec3(SHparas[12], SHparas[13], SHparas[14]);
  o_rgb = o_rgb + vec3(SH[5] * normal.y * normal.z, SH[5] * normal.y * normal.z,
                       SH[5] * normal.y * normal.z) *
                      vec3(SHparas[15], SHparas[16], SHparas[17]);
  o_rgb = o_rgb + vec3(SH[6] * normal.x * normal.z, SH[6] * normal.x * normal.z,
                       SH[6] * normal.x * normal.z) *
                      vec3(SHparas[18], SHparas[19], SHparas[20]);
  o_rgb = o_rgb + vec3(SH[7] * normal.x * normal.y, SH[7] * normal.x * normal.y,
                       SH[7] * normal.x * normal.y) *
                      vec3(SHparas[21], SHparas[22], SHparas[23]);
  o_rgb = o_rgb + vec3(SH[8] * (normal.x * normal.x - normal.y * normal.y),
                       SH[8] * (normal.x * normal.x - normal.y * normal.y),
                       SH[8] * (normal.x * normal.x - normal.y * normal.y)) *
                      vec3(SHparas[24], SHparas[25], SHparas[26]);
  o_rgb = o_rgb * i_rgb;
  return o_rgb;
}

vec3 CalcSHLight_false(vec3 i_rgb, vec3 normal) {
  double pi_ = 3.1415926;
  vec3 o_rgb = vec3(SH[0], SH[0], SH[0]) *
               vec3(SHparas_false[0], SHparas_false[1], SHparas_false[2]);
  o_rgb =
      o_rgb + vec3(SH[1] * normal.z, SH[1] * normal.z, SH[1] * normal.z) *
                  vec3(SHparas_false[3], SHparas_false[4], SHparas_false[5]);
  o_rgb =
      o_rgb + vec3(SH[2] * normal.y, SH[2] * normal.y, SH[2] * normal.y) *
                  vec3(SHparas_false[6], SHparas_false[7], SHparas_false[8]);
  o_rgb =
      o_rgb + vec3(SH[3] * normal.x, SH[3] * normal.x, SH[3] * normal.x) *
                  vec3(SHparas_false[9], SHparas_false[10], SHparas_false[11]);
  o_rgb =
      o_rgb + vec3(SH[4] * (3 * normal.z * normal.z - 1),
                   SH[4] * (3 * normal.z * normal.z - 1),
                   SH[4] * (3 * normal.z * normal.z - 1)) *
                  vec3(SHparas_false[12], SHparas_false[13], SHparas_false[14]);
  o_rgb =
      o_rgb + vec3(SH[5] * normal.y * normal.z, SH[5] * normal.y * normal.z,
                   SH[5] * normal.y * normal.z) *
                  vec3(SHparas_false[15], SHparas_false[16], SHparas_false[17]);
  o_rgb =
      o_rgb + vec3(SH[6] * normal.x * normal.z, SH[6] * normal.x * normal.z,
                   SH[6] * normal.x * normal.z) *
                  vec3(SHparas_false[18], SHparas_false[19], SHparas_false[20]);
  o_rgb =
      o_rgb + vec3(SH[7] * normal.x * normal.y, SH[7] * normal.x * normal.y,
                   SH[7] * normal.x * normal.y) *
                  vec3(SHparas_false[21], SHparas_false[22], SHparas_false[23]);
  o_rgb =
      o_rgb + vec3(SH[8] * (normal.x * normal.x - normal.y * normal.y),
                   SH[8] * (normal.x * normal.x - normal.y * normal.y),
                   SH[8] * (normal.x * normal.x - normal.y * normal.y)) *
                  vec3(SHparas_false[24], SHparas_false[25], SHparas_false[26]);
  o_rgb = o_rgb * i_rgb;
  return o_rgb;
}

float interp(float v, float y0, float x0, float y1, float x1) {
  if (v < x0)
    return y0;
  else if (v > x1)
    return y1;
  else
    return (v - x0) * (y1 - y0) / (x1 - x0) + y0;
}

float jet_base(float v) {
  if (v <= -0.75f)
    return 0.0f;
  else if (v <= -0.25f)
    return interp(v, 0.0, -0.75, 1.0, -0.25);
  else if (v <= 0.25f)
    return 1.0;
  else if (v <= 0.75f)
    return interp(v, 1.0, 0.25, 0.0, 0.75);
  else
    return 0.0f;
}

vec3 jet_colormap(float v) {
  float r = jet_base(v * 2.0f - 1.5f);
  float g = jet_base(v * 2.0f - 1.0f);
  float b = jet_base(v * 2.0f - 0.5f);
  return vec3(r, g, b);
}
