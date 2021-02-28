#version 430 core


in vec2 fs_tex;
in vec4 fs_position;

uniform sampler2D texture1;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_position;

void main(){

	vec3 result;

	if (fs_tex.x >= 0 && fs_tex.y >= 0) {
		result = vec3(texture(texture1, fs_tex));
		//out_color = vec4(1.0, 0.5, 0.3, 1.0);
	}
	else result = vec3(1.0, 1.0, 1.0);

	out_color = vec4(result, 1.0);
	out_position = fs_position;

}