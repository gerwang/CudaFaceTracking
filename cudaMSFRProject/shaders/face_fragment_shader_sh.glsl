#version 430 core

in vec3 fs_position;
in vec4 fs_albedo;
in vec3 fs_normal;


uniform double SHparas[27];
uniform double SHparas_false[27];
uniform double SH[9];
uniform double CT[7];

vec3 CalcSHLight(vec3 rgb, vec3 normal);
vec3 CalcSHLight_false(vec3 rgb, vec3 normal);
vec3 CalcDirLight(vec3 i_rgb,vec3 normal,vec3 viewDir);
uniform sampler2D texture1;
uniform float texMapping;
uniform float colorMode;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_computing;
layout(location = 2) out vec4 out_normal;
layout(location = 3) out vec4 out_position;
layout(location = 4) out vec4 out_weight;
layout(location = 5) out vec4 out_albedo;


void main(){

	vec3 normal=normalize(fs_normal);
	//normal.z=-normal.z;
	vec3 viewDir=normalize(vec3(0,0,0)-fs_position);
	vec3 mRGB;
  	vec3 result;

	if (colorMode == 0.f) {
		mRGB = vec3(1.0, 1.0, 1.0);
		result = CalcSHLight(mRGB, normal);
		result = CalcSHLight_false(mRGB, normal) / result;
	}
	else if (colorMode == 1.f) {
		mRGB = vec3(0.7, 0.7, 0.7);
		//result = CalcDirLight(mRGB, normal, viewDir);
		result = CalcSHLight(mRGB, normal);
	}
	else if (colorMode == 2.f) {
		result = vec3(fs_albedo.r, fs_albedo.g, fs_albedo.b);
	}
	else if (colorMode == 3.f) {
		mRGB = vec3(0.7, 0.7, 0.7);
		result = CalcSHLight_false(mRGB, normal);
	}
	else if (colorMode == 4.f) {
		mRGB = vec3(fs_position.x, fs_position.y, fs_position.z);
		//result = CalcSHLight(mRGB, normal);
	}
	else {
		mRGB = vec3(0.7, 0.7, 0.7)*vec3(fs_albedo.r, fs_albedo.g, fs_albedo.b);
		result = CalcDirLight(mRGB, normal, viewDir);
		//result = vec3(3*(fs_position.z-0.4),3*(fs_position.z-0.4),3*(fs_position.z-0.4));
		//result = vec3(fs_position.x/640,fs_position.y/720,3*(fs_position.z-0.4));
		//result = vec3(fs_position.x/640,0,0);
	}

	//color transfer
	//vec3 offset = vec3(CT[3], CT[4], CT[5]);
	//mat3 CTM = mat3(CT[0] * (CT[6] + 0.3*(1 - CT[6])), CT[1] * (0.3*(1 - CT[6])), CT[2] * (0.3*(1 - CT[6])),
	//CT[0] * (0.59*(1 - CT[6])), CT[1] * (CT[6] + 0.59*(1 - CT[6])), CT[2] * (0.59*(1 - CT[6])),
	//CT[0] * (0.11*(1 - CT[6])), CT[1] * (0.11*(1 - CT[6])), CT[2] * (CT[6] + 0.11*(1 - CT[6])));
	//result = CTM*result + offset;

	out_color=vec4(result,1.0);
	out_computing = vec3(fs_position.x, fs_position.y, fs_position.z);
	out_normal = vec4(normal.x, normal.y, normal.z,1.0);
	out_position = vec4(fs_position, 1.0);
	out_weight = vec4(1.0, 1.0, 1.0, 1.0);
	out_albedo = fs_albedo;
}

vec3 CalcDirLight(vec3 rgb,vec3 normal,vec3 viewDir)
{
	vec3 lightDir = -normalize(vec3(0.0f,0.05f,0.866025f));
	float diff = max(dot(normal,lightDir),0.0);
	//specular shading
	vec3 reflectDir = normalize(reflect(lightDir,normal));
	float spec=pow(max(dot(viewDir,reflectDir),0.0),40.f);
	//ambient & diffuse
	vec3 ambient = vec3(0.4f,0.4f,0.4f)*rgb;
	vec3 diffuse = vec3(0.5f,0.5f,0.5f)*diff*rgb;
	vec3 specular = vec3(0.5f,0.5f,0.5f)*0.06f*spec;
	return (ambient+diffuse+specular);
}

vec3 CalcSHLight(vec3 i_rgb, vec3 normal)
{
	double pi_ = 3.1415926;
	vec3 o_rgb = vec3(SH[0],SH[0],SH[0])*vec3(SHparas[0], SHparas[1], SHparas[2]);
	o_rgb = o_rgb + vec3(SH[1]*normal.z, SH[1]*normal.z, SH[1]*normal.z)*vec3(SHparas[3], SHparas[4], SHparas[5]);
	o_rgb = o_rgb + vec3(SH[2]*normal.y, SH[2]*normal.y, SH[2]*normal.y)*vec3(SHparas[6], SHparas[7], SHparas[8]);
	o_rgb = o_rgb + vec3(SH[3]*normal.x, SH[3]*normal.x, SH[3]*normal.x)*vec3(SHparas[9], SHparas[10], SHparas[11]);
	o_rgb = o_rgb + vec3(SH[4]*(3*normal.z*normal.z-1), SH[4]*(3*normal.z*normal.z-1), SH[4]*(3*normal.z*normal.z-1))*vec3(SHparas[12], SHparas[13], SHparas[14]);
	o_rgb = o_rgb + vec3(SH[5]*normal.y*normal.z, SH[5]*normal.y*normal.z, SH[5]*normal.y*normal.z)*vec3(SHparas[15], SHparas[16], SHparas[17]);
	o_rgb = o_rgb + vec3(SH[6]*normal.x*normal.z, SH[6]*normal.x*normal.z, SH[6]*normal.x*normal.z)*vec3(SHparas[18], SHparas[19], SHparas[20]);
	o_rgb = o_rgb + vec3(SH[7]*normal.x*normal.y, SH[7]*normal.x*normal.y, SH[7]*normal.x*normal.y)*vec3(SHparas[21], SHparas[22], SHparas[23]);
	o_rgb = o_rgb + vec3(SH[8]*(normal.x*normal.x-normal.y*normal.y), SH[8]*(normal.x*normal.x-normal.y*normal.y), SH[8]*(normal.x*normal.x-normal.y*normal.y))*vec3(SHparas[24], SHparas[25], SHparas[26]);
	o_rgb = o_rgb*i_rgb;
	return o_rgb;
}

vec3 CalcSHLight_false(vec3 i_rgb, vec3 normal)
{
	double pi_ = 3.1415926;
	vec3 o_rgb = vec3(SH[0],SH[0],SH[0])*vec3(SHparas_false[0], SHparas_false[1], SHparas_false[2]);
	o_rgb = o_rgb + vec3(SH[1]*normal.z, SH[1]*normal.z, SH[1]*normal.z)*vec3(SHparas_false[3], SHparas_false[4], SHparas_false[5]);
	o_rgb = o_rgb + vec3(SH[2]*normal.y, SH[2]*normal.y, SH[2]*normal.y)*vec3(SHparas_false[6], SHparas_false[7], SHparas_false[8]);
	o_rgb = o_rgb + vec3(SH[3]*normal.x, SH[3]*normal.x, SH[3]*normal.x)*vec3(SHparas_false[9], SHparas_false[10], SHparas_false[11]);
	o_rgb = o_rgb + vec3(SH[4]*(3*normal.z*normal.z-1), SH[4]*(3*normal.z*normal.z-1), SH[4]*(3*normal.z*normal.z-1))*vec3(SHparas_false[12], SHparas_false[13], SHparas_false[14]);
	o_rgb = o_rgb + vec3(SH[5]*normal.y*normal.z, SH[5]*normal.y*normal.z, SH[5]*normal.y*normal.z)*vec3(SHparas_false[15], SHparas_false[16], SHparas_false[17]);
	o_rgb = o_rgb + vec3(SH[6]*normal.x*normal.z, SH[6]*normal.x*normal.z, SH[6]*normal.x*normal.z)*vec3(SHparas_false[18], SHparas_false[19], SHparas_false[20]);
	o_rgb = o_rgb + vec3(SH[7]*normal.x*normal.y, SH[7]*normal.x*normal.y, SH[7]*normal.x*normal.y)*vec3(SHparas_false[21], SHparas_false[22], SHparas_false[23]);
	o_rgb = o_rgb + vec3(SH[8]*(normal.x*normal.x-normal.y*normal.y), SH[8]*(normal.x*normal.x-normal.y*normal.y), SH[8]*(normal.x*normal.x-normal.y*normal.y))*vec3(SHparas_false[24], SHparas_false[25], SHparas_false[26]);
	o_rgb = o_rgb*i_rgb;
	return o_rgb;
}


