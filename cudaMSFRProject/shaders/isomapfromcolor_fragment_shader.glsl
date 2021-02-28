#version 430 core

in vec4 fs_albedo;
in vec4 fs_position;


layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_position;

void main(){

	out_color = fs_albedo;
	out_position = fs_position;
}