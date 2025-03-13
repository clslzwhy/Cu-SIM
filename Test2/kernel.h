
#pragma once
#include <vector>
#include <list>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cublas_v2.h"
#include "cufft.h" 
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "mclmcrrt.h"
#include "libMatlab_deconvlucy.h"
#include "libMatlab_edgetaper.h"
//#include "libMatlab_ifft2.h"

#define pi_Loco 3.1415926535897

#define epsy 2.220446049250313e-14
#define ceil_Rec 0.001
//#define ImageCutTop 24
//#define ImageCutTop 28
//#define ImageBefore 784
//#define ImageAfter 1568
//#define ImageCutTop 32
//#define ImageBefore 1024
//#define ImageAfter 2048
using namespace std;


struct dataparaCpu
{
	double numangles = 3;
	double numsteps = 5;
	double numpixelsx;
	double numpixelsy;


	double NA = 1.49;
	double lambda;
	double nrBands = 3;

	double rawpixelsize[3] = { 65,65,125 };
	double upsampling[3] = { 2,2,1 };
	double refmed = 1.47;
	double refcov = 1.512;
	double refimm = 1.515;
	double exwavelength;
	double emwavelength;
	double fwd = 120000;
	double depth = 0;
	double xemit = 0;// x - position focal point
	double yemit = 0;// y - position focal point
	double zemit = 0;// z - position focal point
	double attStrength = 0;
	double maxorder = 3;
	double cyclesPerMicron;
	int sampleLateral;
	double cutoff;
	double estimateAValue;
	double ImageSizex;
	double ImageSizey;
	double attFWHM = 1.0;
	double Nx;
	double Ny;
	double Nz;
	std::vector<cv::Mat> Snoisy;
	cv::Mat* allftorderims;
	cv::Mat* OTFshift0;


	double notchwidthxy1;
	double notchwidthxy2;
	double	notchdips1 = 0.92;
	double	notchdips2 = 0.98;
	double notchheight = 3.0;
	double lambdaregul1 = 0.5;
	double lambdaregul2 = 0.1;
	double ImageCut;
	double ImageCutBlock;
	double ImageCutThread;

};
struct dataparaGPU
{




};

struct GPUHandle
{
	cufftHandle fftn2;
};


struct sCudaPara
{
	dataparaCpu dataparamsCpu;
	dataparaGPU dataparamsGpu;
	GPUHandle dataparamsHandle;
};



class  CalFuntion
{
public:
	string MainPath;
	int ImageSizeCut;
	int ImgSizeBefore;
	int ImgSizeAfter;
	int hengxiang = 0;
	int zongxiang = 0;
	cv::Mat double2Mat16(cv::Mat tes, int h, int w);
	void get_pupil_matrix(cufftDoubleComplex* result, sCudaPara& para, int Raw_Size_Q);
	void get_pupil_matrix_C(cv::Mat* result_x, cv::Mat* result_y, double NA, double lambda, double refimm, double refmed, double refcov, double fwd, double depth, double pixelsize, int Raw_Size_Q);
	void prechirpz(cufftDoubleComplex* Dfft_G, cufftDoubleComplex* A_G, cufftDoubleComplex* B_G, double xsize, double qsize, double N, double M, sCudaPara& para);

	void get_throughfocusotf_C(cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By,
		double PupilSize, double Nx, double Ny, double* rawpixelsize, int Raw_Size_Q, sCudaPara& para);

	void get_rimismatchpars(double& zvals_1, double& zvals_2, double& zvals_3, double refimm, double  refimmnom, double refmed, double fwd, double depth, double NA);
	void get_modelOTF_G(cufftDoubleComplex* dst, sCudaPara& para, cufftDoubleComplex* wavevector, int Nz, int Raw_Size_Q);
	void get_field_matrix(double* dst, cufftDoubleComplex* wavevector, sCudaPara& para);
	void get_field_matrix_C(cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By,
		double PupilSize, double Nx, double Ny, double* rawpixelsize, int Raw_Size_Q, sCudaPara& para);

	void get_throughfocusotf(cufftDoubleComplex* dst, double* PFS, sCudaPara& para);
	void get_field_matrix_G(double* dst, sCudaPara& para, cufftDoubleComplex* wavevector, cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By, int Nz, int Raw_Size_Q);
	void transpose_cztfunc(cufftDoubleComplex* result, cufftDoubleComplex* PupilFunction, cufftDoubleComplex* A, cufftDoubleComplex* B, cufftDoubleComplex* D, int Raw_Size_Q, sCudaPara& para);
	void get_psf(double* dst, double* PSFin, int Nz, sCudaPara& para, int Raw_Size_Q);
	void get_throughfocusotf_G(cufftDoubleComplex* dst, double Nx, double* PSFslice, cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By, int Nz, sCudaPara& para, int Raw_Size_Q);
	void get_otf3d(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int Nz, int Raw_Size_Q);
	void do_OTFmasking3D(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int Nz, int Raw_Size_Q);
	void Recon3DSIM(cv::Mat* SimResult, sCudaPara& para, int dataH, int dataW);
	void ShowDouble(double* paradouble, int dataH, int dataW, int id_start, int id_end, int jd_start, int jd_end, int Z_start, int Z_end);
	void ShowComplex(cufftDoubleComplex* paradouble, int dataH, int dataW, int id_start, int id_end, int jd_start, int jd_end, int zi, int zj);
	void get_normalization(double& normint_free, cv::Mat* PupilMatrix_x, cv::Mat* PupilMatrix_y, double pixelsize, double  NA, double lambda);
	void find_illumination_pattern(double* peak_kx, double* peak_ky, double* Snoisy, double SIMparams_SIMpixelsize, cufftDoubleComplex* dst, double* Module, sCudaPara& para);
	void SimOtfProvider(double* dst, sCudaPara& para, int Raw_Size_Q);
	void deconvlucy(double* result, string fileName, sCudaPara& para, int dataH, int dataW);
	void edgetaper(std::vector<cv::Mat> &Snoisy);
	void Mat_edgetaper(cv::Mat& src);
	void Mat_deconvlucy(cv::Mat* dst, string srcName, int dataH, int dataW);
	mxArray* TifMat2mwArray_Double(cv::Mat src);
	mxArray* MatReal_2mwArray_Double(cv::Mat src);
	mxArray* Double2mwArray(double Num);
	cv::Mat mwArry2doubleMat(mxArray* QQ);
	void FFT2D_15(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int inverse);
	void find_illumination_pattern_Cal_C0C3(cufftDoubleComplex* result_fft, cufftDoubleComplex* result_Norm, cufftDoubleComplex* result, cufftDoubleComplex* src, double* NotchFilter1, sCudaPara& para, int Num, int MaxNum);
	void FFT2D_1(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int inverse);
	cv::Mat cufftDoubleComplex2Mat(cufftDoubleComplex* src, int dataH, int dataW);
	void Mat2cufftDoubleComplex(cufftDoubleComplex* output, const cv::Mat& input);
	mxArray* Mat2mwArray_Double(cv::Mat src);
	void fitPeak(double& newKx, double& newKy, cufftDoubleComplex* c0_Norma, cufftDoubleComplex* c3_Norma, cufftDoubleComplex* c0_Mask, cufftDoubleComplex* c3_Mask, double* otf, sCudaPara& para, double kx, double ky, double weightLimit, double search);
	void commonRegion(cufftDoubleComplex* newb0, cufftDoubleComplex* newb1, cufftDoubleComplex* band0, cufftDoubleComplex* band1, cufftDoubleComplex* c0_Mask, cufftDoubleComplex* c3_Mask, double* otf, sCudaPara& para, double kx, double ky, double dist, double weightLimit);
	void getPeak(cufftDoubleComplex* dst, cufftDoubleComplex* src1, cufftDoubleComplex* src2, cufftDoubleComplex* c0_Mask, cufftDoubleComplex* c3_Mask, double* otf, sCudaPara& para, double kx, double ky, double weightLimit);
	double peak_kx[3];//目前分析是3不是Nz
	double peak_ky[3];
	void process_data(cufftDoubleComplex* dst1, double* dst2, double* kx, double* ky, double* allimages_in, cufftDoubleComplex* OTFem, cufftDoubleComplex* find_illumination_pattern_p1, double* Module, sCudaPara& para);
	void Inv(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para);
	void invert_device(cufftDoubleComplex* cu_5_a, cufftDoubleComplex* cu_5_o, sCudaPara& para);
	void shift_2D(cufftDoubleComplex* dst, cufftDoubleComplex* src, double kx, double ky, int Testi2, sCudaPara& para);

	void shift_3D(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Nz, double kx, double ky, double kz, sCudaPara& para, int Raw_Size_Q);
	void FuctionGetFilter(double* Filter1, double* Filter2, double* OTFshiftfinal, double lambdaregul1, double lambdaregul2, double* peak_kx, double* peak_ky, int Npola, int Nazim, sCudaPara& para);
	void filter_3D(double* Result, cufftDoubleComplex* sum_fft, sCudaPara& para);


	int Npola;
	int Nazim;//3不是Nz

};
