# Cu-SIM
1. Project Overview
This project focuses on implementing fast 3D SIM (Structured Illumination Microscopy) reconstruction based on the Cu-SIM algorithm. By migrating the core computations to the CUDA side for processing, we fully leverage the parallel computing capabilities of the GPU, significantly enhancing the speed and efficiency of 3D SIM reconstruction. This enables more efficient handling of relevant microscopic image data.

2. Prerequisites
Software Requirements
Visual Studio: This project requires Visual Studio with a version of 2017 or higher. You can download the latest version from the official Microsoft website.
CUDA: For GPU-based accelerated computing, a CUDA version of 10.2 or above is necessary. The CUDA Toolkit can be obtained from the NVIDIA website. Please ensure that your GPU is compatible with the installed CUDA version.
Eigen: An Eigen library version of 3.4.0 or higher is required. Eigen is a C++ template library for linear algebra. You can download it from the official Eigen website and integrate it into your project.
OpenCV: The project uses OpenCV version 4.1.2. OpenCV is a popular computer vision library. You can install it via package managers like apt-get (on Linux systems) or download the pre-built binaries from the official OpenCV website.
MATLAB Runtime: A MATLAB Runtime version of R2021a or higher is required. If your application interacts with MATLAB-based code, this runtime environment enables the execution of MATLAB code without the need for a full MATLAB installation. You can obtain it from the MathWorks website.
Hardware Requirements
GPU: A GPU with 24GB of video memory and a model of NVIDIA 3090 or higher is required to ensure sufficient computing resources and video memory to support the core computations on the CUDA side and handle large-scale 3D SIM reconstruction data.

3. Installation
Step-by-Step Installation Guide
Install Visual Studio:
Visit the official Microsoft Visual Studio download page.
Select the appropriate version (Community, Professional, or Enterprise) and click the download button.
Run the installer and follow the on-screen instructions. Make sure to select the necessary components for C++ development, such as the C++ compiler and build tools.
Install CUDA:
Visit the NVIDIA CUDA Toolkit download page.
Select the appropriate CUDA version (10.2 or above) according to your GPU model and operating system.
Download the installer and run it. Follow the steps in the installation wizard, which may include accepting the license agreement, choosing the installation directory, and installing CUDA samples for testing.
Install Eigen:
Go to the official Eigen website.
Download the latest version (3.4.0 or higher).
Extract the downloaded archive to a directory in your project workspace. In your Visual Studio project, add the Eigen include directory to the project's include paths.
Install OpenCV:
For Linux systems (using apt-get):
Open a terminal and run sudo apt-get update.
Then run sudo apt-get install libopencv-dev.
For Windows systems:
Download the pre-built OpenCV binaries from the official website.
Extract the archive. In your Visual Studio project, add the OpenCV include directories to the project's include paths and the OpenCV library directories to the library paths. Also, add the necessary OpenCV DLL files to the project's output directory or the system path.
Install MATLAB Runtime:
Log in to the MathWorks website.
Navigate to the MATLAB Runtime download section.
Download the installer suitable for your operating system and MATLAB version (R2021a or higher).
Run the installer and follow the installation instructions.

4. Usage
Input Data Preparation
Before running the 3D SIM reconstruction program, you need to prepare the original 3D SIM image data that meets the requirements. Ensure that the format of the data is consistent with the format supported by the project code.
Running the Program
Open Visual Studio and load the solution file of this project.
Ensure that the project configuration (such as debug/release mode, target platform, etc.) meets your needs.
Compile and run the project. The program will automatically allocate the core computation tasks to the CUDA side for processing.
Output Results
After the program finishes running, the reconstructed 3D SIM image data will be generated. According to the settings in the project code, you can save the results as files in the specified format for subsequent analysis and processing.

5. Contribution Guidelines
Coding Style
Please follow the existing coding style and specifications of the project to ensure the consistency and readability of the code. When writing code, add necessary comments to make it easier for other developers to understand the functionality and logic of the code.

6. Contact Information
If you encounter any problems or have any suggestions while using this project, you are welcome to contact us in the following ways:
Email: hongyuwang@pku.edu.cn
GitHub Issue Tracking: Create a new Issue in the GitHub repository of this project and describe your problem or suggestion in detail.
