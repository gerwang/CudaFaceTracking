#version 430 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_albedo;
layout (location = 2) in vec3 in_weight;
layout(location = 3) in vec2 in_tex;
layout(location = 4) in vec3 in_normal;
layout(location = 5) in vec2 in_tri;

uniform mat4 projectionMatrix;

out vec4 fs_position;
out vec4 fs_albedo;

void main()
{
	gl_Position = vec4(in_tex, 0.0f, 1.0f);
	fs_position = gl_Position;
	fs_albedo = in_albedo;
}

