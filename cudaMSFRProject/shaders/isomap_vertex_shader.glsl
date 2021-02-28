#version 430 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex;

uniform mat4 projectionMatrix;

out vec4 fs_position;
out vec2 fs_tex;

void main(){

	gl_Position = vec4(in_position, 1.0f);

	fs_position = gl_Position;
	fs_tex = in_tex;
}