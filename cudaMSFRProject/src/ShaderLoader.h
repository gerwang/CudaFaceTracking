#pragma once

#include <GL\glew.h>
#include <GLFW/glfw3.h>
//#include <GL\freeglut.h>
#include <iostream>

class ShaderLoader
{
	private:

		std::string ReadShader(char *filename);
		GLuint CreateShader(GLenum shaderType,std::string source,char* shaderName);

	public:
		ShaderLoader(void);
		~ShaderLoader(void);
		GLuint CreateProgram(char* VertexShaderFilename,char* FragmentShaderFilename);

};

