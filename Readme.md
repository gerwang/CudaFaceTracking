# cudaMSFRProject

### 安装说明

项目在windows系统上运行，使用Visual Studio 2019编译。

- 安装CUDA 10.2
- 从git上下载代码后，进入文件夹
- 将网盘（https://cloud.tsinghua.edu.cn/d/4e49f63f17a242d48912/，密码oppo-thu）中的文件放到./cudaMSFRProject同目录下
- 用Visual Studio 2019打开cudaMSFRProject.sln
- 用Ctrl+F5编译运行程序

### 运行说明

编辑configuration.json修改输入文件的路径：

- 修改`avatar_folder`改变被驱动的avatar。默认使用的是`./input/avatar/atiqur_rahman`。
- 修改`color_folder`改变输入图片序列的位置。