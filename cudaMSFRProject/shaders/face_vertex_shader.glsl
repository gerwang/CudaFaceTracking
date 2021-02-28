#version 430 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_albedo;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec3 in_weight;
layout(location = 4) in float in_tri;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform double point2pixel[4];

out vec3 fs_position;
out vec4 fs_albedo;
out vec3 fs_normal;
out vec4 fs_weight;
out vec4 fs_tri;


void main()
{
	vec4 real_position = modelViewMatrix*vec4(in_position, 1.0f);
	gl_Position = projectionMatrix*real_position;
	//gl_Position = projectionMatrix*vec4(real_position.x/real_position.z * point2pixel[0] + point2pixel[2], real_position.y/real_position.z* point2pixel[1] + point2pixel[3], real_position.z, 1.0);
	//vec4 m_pos = vec4(real_position.x/real_position.z * point2pixel[0] + point2pixel[2], real_position.y/real_position.z* point2pixel[1] + point2pixel[3], real_position.z, 1.0);
	vec4 m_pos = real_position;
	vec4 m_normal = transpose(inverse(modelViewMatrix)) * vec4(in_normal, 0.0);

	fs_position = vec3(real_position.x, real_position.y, real_position.z);
	fs_albedo = in_albedo;
  fs_normal = m_normal.xyz;
  fs_weight = vec4(in_weight, 1.0);
  fs_tri = vec4(in_tri, 0.0, 0.0, 1.0);
}

