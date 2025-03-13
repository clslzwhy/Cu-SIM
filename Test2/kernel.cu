
#include "cuda_runtime.h" //提供了时间计算的功能函数
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "kernel.h"
#include <stdio.h>
#include <time.h>
dim3 blocks;
dim3 threads;
dim3 blocks2;
dim3 threads2;
dim3 blocks3;
dim3 threads3;

dim3 blocks6;
dim3 blocks8;
dim3 blocks9;
dim3 blocks7;

dim3 blocks10;








__device__ cufftDoubleComplex DeviceComplexMulty(cufftDoubleComplex src1, cufftDoubleComplex src2)
{
	cufftDoubleComplex dst;
	dst.x = src1.x * src2.x - src1.y * src2.y;
	dst.y = src1.x * src2.y + src1.y * src2.x;
	return dst;

}




__global__ void Abs_C_type4(double* dst, cufftDoubleComplex* src)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;

	dst[i] = sqrt(src[i].x * src[i].x + src[i].y * src[i].y);

}
__global__ void Add_type2(double* dst, double* src1, double* src2)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;

	dst[i] = src1[i] + src2[i];

}
__global__ void CudaCircshiftThreeDimension(double* dst, double* src, int direction, int Nz, int dimshift, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int X, Y;
	X = i0 / Raw_Size_Q;
	Y = i0 % Raw_Size_Q;
	int X_new, Y_new, Z_new;
	if (direction == 0)
	{
		X_new = X + dimshift;
		if (X_new >= Raw_Size_Q)
		{
			X_new = X_new - Raw_Size_Q;
		}
		if (X_new < 0)
		{
			X_new = X_new + Raw_Size_Q;
		}
		Y_new = Y;
	}
	if (direction == 1)
	{
		X_new = X;
		Y_new = Y + dimshift;

		if (Y_new >= Raw_Size_Q)
		{
			Y_new = Y_new - Raw_Size_Q;
		}
		if (Y_new < 0)
		{
			Y_new = Y_new + Raw_Size_Q;
		}
	}
	if (direction == 2)
	{
		X_new = X;
		Y_new = Y;
	}


	for (int Z = 0; Z < Nz; Z++)
	{
		if (direction == 2)
		{
			Z_new = Z + dimshift;
		}
		else
		{
			Z_new = Z;
		}
		if (Z_new >= Nz)
		{
			Z_new = Z_new - Nz;
		}
		if (Z_new < 0)
		{
			Z_new = Z_new + Nz;
		}
		int i = i0 + Z * Raw_Size_Q * Raw_Size_Q;
		int j = Y_new + Raw_Size_Q * X_new + Z_new * Raw_Size_Q * Raw_Size_Q;
		dst[j] = src[i];
	}


}

__global__ void CudaCalFlip(double* dst, double* src, int direction, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int X, Y;
	X = i0 / Raw_Size_Q;
	Y = i0 % Raw_Size_Q;
	int X_new, Y_new, Z_new;
	if (direction == 0)
	{
		X_new = Raw_Size_Q - 1 - X;
		Y_new = Y;
	}
	if (direction == 1)
	{
		X_new = X;
		Y_new = Raw_Size_Q - 1 - Y;
	}
	if (direction == 2)
	{
		X_new = X;
		Y_new = Y;
	}


	for (int Z = 0; Z < Nz; Z++)
	{
		if (direction == 2)
		{
			Z_new = Nz - Z - 1;
		}
		else
		{
			Z_new = Z;
		}
		int i = i0 + Z * Raw_Size_Q * Raw_Size_Q;
		int j = Y_new + Raw_Size_Q * X_new + Z_new * Raw_Size_Q * Raw_Size_Q;
		dst[j] = src[i];
	}



}
__global__ void CudaCalFilterStepOne(double* dst, double* src)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	double Apo = src[i];
	if (Apo < epsy)
	{
		Apo = 0;
	}
	dst[i] = pow(Apo, 0.4);


}
__global__ void CudaCalFilterStepThree(double* dst, double* Apo, double* OTFshiftfinal, double lambdaregul, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	if ((OTFshiftfinal[i] + lambdaregul) != 0)
	{
		dst[i] = Apo[i] / (OTFshiftfinal[i] + lambdaregul);
	}
	else
	{
		dst[i] = epsy;
	}


}
__global__ void Add_type3(double* dst, double* src1, double* src2, int Num)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;

	dst[i] = src1[i] + src2[Num];

}
__global__ void CudaCalFilterStepTwo(double* dst, double * cutoff, double SIMpixelsize_1, double SIMpixelsize_2, double SIMpixelsize_3, int z, int Nz, int  Npola, int Nazim, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;


	int X, Y;
	X = i % Raw_Size_Q;
	Y = i / Raw_Size_Q;
	double qx, qy, qz;
	qx = (double(i % Raw_Size_Q) - double(Raw_Size_Q / 2)) * (1.0 / double(Raw_Size_Q) / SIMpixelsize_1);
	qy = (double(i / Raw_Size_Q) - double(Raw_Size_Q / 2)) * (1.0 / double(Raw_Size_Q) / SIMpixelsize_2);
	qz = (double(z) - floor(double(Nz) / 2.0)) * (1.0 / double(Nz) / SIMpixelsize_3);
	double qrad = sqrt(qx * qx + qy * qy + qz * qz);
	double qcospol = qz / qrad;

	if ((z == (Nz / 2)) && (X == (Raw_Size_Q / 2)) && (Y == (Raw_Size_Q / 2)))
	{
		qcospol = 0;
	}
	double qphi = atan2(qy, qx);

	int  alljazi = ceil((pi_Loco + qphi) * double(Nazim) / 2.0 / pi_Loco - ceil_Rec) - 1.0;
	int  alljpol = ceil((1.0 + qcospol) * double(Npola) / 2.0 - ceil_Rec) - 1.0;
	if (alljpol < 0)
	{
		alljpol = 0;
	}
	double cutoffmap = cutoff[/*alljazi * Npola + */alljpol* Nazim + alljazi];

	dst[i] = 1.0 - qrad / cutoffmap;



}

__global__ void CudaCalFilterStepCalCutoff(double* dst, double NAl, cufftDoubleComplex NBl, double q02, double q0ex,
	double SIMpixelsize_1, double  pkx1, double pkx2, double pkx3, double pky1, double pky2, double pky3, int Npola, int Nazim, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int X, Y;
	Y = i % Nazim;
	X = i / Nazim;

	double delqr = 4.0 * NAl / (4.0 * Raw_Size_Q);
	double allazim = (2.0 * (Y + 1.0) - 1.0 - Nazim) * pi_Loco / Nazim;
	double allcosazim = cos(allazim);
	double allsinazim = sin(allazim);
	double allcospola = (2.0 * (X + 1.0) - 1.0 - Npola) / Npola;
	double allsinpola = sqrt(1 - allcospola * allcospola);


	double dirx = allsinpola * allcosazim;
	double diry = allsinpola * allsinazim;
	double dirz = allcospola;
	int centerorder = (5 + 1) / 2;

	double  patternangles[3];
	patternangles[0] = atan(pky1 / pkx1);
	patternangles[1] = atan(pky2 / pkx2);
	patternangles[2] = atan(pky3 / pkx3);


	double patternpitch[3];
	patternpitch[0] = 2 * SIMpixelsize_1 * double(Raw_Size_Q) / (sqrt(pky1 * pky1 + pkx1 * pkx1));
	patternpitch[1] = 2 * SIMpixelsize_1 * double(Raw_Size_Q) / (sqrt(pky2 * pky2 + pkx2 * pkx2));
	patternpitch[2] = 2 * SIMpixelsize_1 * double(Raw_Size_Q) / (sqrt(pky3 * pky3 + pkx3 * pkx3));
	//dst[i] = patternpitch[0];
	double sum = 0;
	for (int qri = 0; qri < Raw_Size_Q * 8 + 1; qri++)
	{
		int Badd = 0;
		for (int jangle = 0; jangle < 3; jangle++)//3Numangle,必须是3，不是Nz
		{
			double qvector_1 = cos(patternangles[jangle]) / patternpitch[jangle];
			double qvector_2 = sin(patternangles[jangle]) / patternpitch[jangle];
			double axialshift = q0ex - sqrt(q0ex * q0ex - 1.0 / (patternpitch[jangle] * patternpitch[jangle]));

			for (int jorder = 0; jorder < 5; jorder++)//numsteps=5;
			{

				int mm = jorder - 3 + 1;

				double qr = double(qri - Raw_Size_Q * 4) * delqr;
				double qpar = sqrt((qr * dirx - mm * qvector_1) * (qr * dirx - mm * qvector_1) + (qr * diry - mm * qvector_2) * (qr * diry - mm * qvector_2));
				double qax = qr * dirz;
				if (mm % 2 == 0)//0,-3,1   1,-2,0   mod(mm,2)==0
				{
					cufftDoubleComplex comp_lo1;
					comp_lo1.x = qax - NBl.x;
					comp_lo1.y = -NBl.y;
					double maskmn = DeviceComplexMulty(comp_lo1, comp_lo1).x + (qpar - NAl) * (qpar - NAl);
					comp_lo1.x = qax + NBl.x;
					comp_lo1.y = NBl.y;
					double maskpl = DeviceComplexMulty(comp_lo1, comp_lo1).x + (qpar - NAl) * (qpar - NAl);
					if (maskmn <= q02 && maskpl <= q02)
					{
						Badd++;
					}
				}
				else
				{
					cufftDoubleComplex comp_lo1;
					comp_lo1.x = qax + axialshift - NBl.x;
					comp_lo1.y = -NBl.y;
					double maskplmn = DeviceComplexMulty(comp_lo1, comp_lo1).x + +(qpar - NAl) * (qpar - NAl);
					comp_lo1.x = qax + axialshift + NBl.x;
					comp_lo1.y = NBl.y;
					double maskplpl = DeviceComplexMulty(comp_lo1, comp_lo1).x + (qpar - NAl) * (qpar - NAl);
					comp_lo1.x = qax - axialshift - NBl.x;
					comp_lo1.y = -NBl.y;
					double maskmnmn = DeviceComplexMulty(comp_lo1, comp_lo1).x + (qpar - NAl) * (qpar - NAl);
					comp_lo1.x = qax - axialshift + NBl.x;
					comp_lo1.y = NBl.y;
					double maskmnpl = DeviceComplexMulty(comp_lo1, comp_lo1).x + (qpar - NAl) * (qpar - NAl);
					if ((maskplmn <= q02 && maskplpl <= q02) || (maskmnmn <= q02 && maskmnpl <= q02))
					{
						Badd++;
					}

				}


			}

		}
		if (Badd > 0)
		{
			sum = sum + 1;
		}

	}
	dst[i] = sum * delqr / 2.0;


}


__global__ void CalMask_type2(cufftDoubleComplex* dst, cufftDoubleComplex* src, double MinNum)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;



	if (src[i].x < MinNum)
	{
		dst[i].x = 0;
		dst[i].y = 0;
	}
	else
	{
		dst[i] = src[i];
	}
}

__global__ void Mul_type8(cufftDoubleComplex* dst, cufftDoubleComplex* src1, double* src2, int Raw_Size_Q)//src1*src2[0]
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	cufftDoubleComplex shiftOperatorX;
	shiftOperatorX.x = cos(src2[0]);
	shiftOperatorX.y = sin(src2[0]);


	double temp;
	temp = src1[i].x * shiftOperatorX.x - src1[i].y * shiftOperatorX.y;
	dst[i].y = src1[i].y * shiftOperatorX.x + src1[i].x * shiftOperatorX.y;
	dst[i].x = temp;
}
__global__ void Cal_sum_angle(double* result, cufftDoubleComplex* src, int Num, int resultNum)//src1*conjsrc2
{
	double sum_x = 0, sum_y = 0;

	for (int i = 0; i < Num; i++)
	{
		sum_x = sum_x + src[i].x;
		sum_y = sum_y + src[i].y;
	}
	/*result[resultNum].x = sum_x;
	result[resultNum].y = sum_y;*/
	result[resultNum] = atan2(sum_y, sum_x);
}
__global__ void Mul_type7(cufftDoubleComplex* dst, cufftDoubleComplex* src1, cufftDoubleComplex* src2, int Raw_Size_Q)//src1*conjsrc2
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	double temp;
	temp = src1[i].x * src2[i].x + src1[i].y * src2[i].y;
	dst[i].y = src1[i].y * src2[i].x - src1[i].x * src2[i].y;
	dst[i].x = temp;
}
__global__ void Abs_Add_type1(double* dst, double* src1, cufftDoubleComplex* src2)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	dst[i] = src1[i] + sqrt(src2[i].x * src2[i].x + src2[i].y * src2[i].y);

}
__global__ void Mul_type6(cufftDoubleComplex* dst, cufftDoubleComplex* src1, cufftDoubleComplex* src2, int Raw_Size_Q)//src1*src2
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	double temp;
	temp = src1[i].x * src2[i].x - src1[i].y * src2[i].y;
	dst[i].y = src1[i].y * src2[i].x + src1[i].x * src2[i].y;
	dst[i].x = temp;
}
__global__ void Cal_shiftedFFT(cufftDoubleComplex* dst, cufftDoubleComplex* src, double kx, double ky, double kz, int Num, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	double qx, qy, qz;
	qx = kx * double(i % Raw_Size_Q) - double(Raw_Size_Q / 2) / double(Raw_Size_Q);
	qy = ky * double(i / Raw_Size_Q) - double(Raw_Size_Q / 2) / double(Raw_Size_Q);
	qz = floor(kz)*(double(Num) - double(Nz) / 2.0) / double(Nz);
	cufftDoubleComplex shiftOperatorX, shiftOperatorY, shiftOperatorZ;

	shiftOperatorX.x = cos(2 * pi_Loco * qx);
	shiftOperatorX.y = sin(2 * pi_Loco * qx);

	shiftOperatorY.x = cos(2 * pi_Loco * qy);
	shiftOperatorY.y = sin(2 * pi_Loco *qy);

	shiftOperatorZ.x = cos(2 * pi_Loco *  qz);
	shiftOperatorZ.y = sin(2 * pi_Loco * qz);
	cufftDoubleComplex test = src[i];
	cufftDoubleComplex result = DeviceComplexMulty(DeviceComplexMulty(DeviceComplexMulty(test, shiftOperatorX), shiftOperatorY), shiftOperatorZ);
	dst[i] = result;

	//dst[i].x = /*src[i].x * */qz/* - src[i].y * shiftOperatorZ.y*/;
	//dst[i].y =/* src[i].x **/ floor(kz) /*+ src[i].y * shiftOperatorZ.x*/;


	//dst[i].x = result.x;
	//dst[i].y = test.x;
}

__global__ void ifftShfit_3D_C(cufftDoubleComplex* dst, cufftDoubleComplex* src, int nx)//多维
{//到这 do_upsample fftShfit_3D_R 20240105写到这
	int m = blockIdx.z % 4;
	int n = (blockIdx.z - blockIdx.z % 4) / 4;
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + m * gridDim.x * gridDim.y;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + nx * nx * n;
	int gz = gridDim.z / 4;

	int x, y;
	int threadId_2D1 = i0 % nx;
	int blockId_2D1 = (i0 - threadId_2D1) / nx;

	int z_new;
	if (blockId_2D1 < nx / 2)
	{
		if (threadId_2D1 > nx / 2 - 1)
		{
			x = threadId_2D1 - nx / 2;
		}
		else
		{
			x = threadId_2D1 + nx / 2;
		}
		y = blockId_2D1 + nx / 2;
	}
	else
	{
		if (threadId_2D1 > nx / 2 - 1)
		{
			x = threadId_2D1 - nx / 2;
		}
		else
		{
			x = threadId_2D1 + nx / 2;
		}
		y = blockId_2D1 - nx / 2;
	}


	if (n > (gz / 2 - 1))
	{
		z_new = n - gz / 2;

	}
	else
	{
		if (gz % 2 == 1)
		{
			z_new = n + gz / 2 + 1;
		}
		else
		{
			z_new = n + gz / 2;
		}
	}
	int j = x + nx * y + z_new * nx * nx;
	dst[j].x = src[i].x;
	dst[j].y = src[i].y;



}
__global__ void fftShfit_3D_C_2(cufftDoubleComplex* dst, cufftDoubleComplex* src, int nx)//多维
{//到这 do_upsample fftShfit_3D_R 20240105写到这
	int m = blockIdx.z % 4;
	int n = (blockIdx.z - blockIdx.z % 4) / 4;
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + m * gridDim.x * gridDim.y;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + nx * nx * n;
	int gz = gridDim.z / 4;

	int x, y;
	int threadId_2D1 = i0 % nx;
	int blockId_2D1 = (i0 - threadId_2D1) / nx;

	int z_new;
	if (blockId_2D1 < nx / 2)
	{
		if (threadId_2D1 > nx / 2 - 1)
		{
			x = threadId_2D1 - nx / 2;
		}
		else
		{
			x = threadId_2D1 + nx / 2;
		}
		y = blockId_2D1 + nx / 2;
	}
	else
	{
		if (threadId_2D1 > nx / 2 - 1)
		{
			x = threadId_2D1 - nx / 2;
		}
		else
		{
			x = threadId_2D1 + nx / 2;
		}
		y = blockId_2D1 - nx / 2;
	}


	if (n > (gz / 2 - 1))
	{
		z_new = n - gz / 2;

	}
	else
	{
		if (gz % 2 == 1)
		{
			z_new = n + gz / 2 + 1;
		}
		else
		{
			z_new = n + gz / 2;
		}
	}
	int j = x + nx * y + z_new * nx * nx;
	dst[i].x = src[j].x;
	dst[i].y = src[j].y;



}

__global__ void ifftR2RRec_2(cufftDoubleComplex* dst, cufftDoubleComplex* src, double Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	dst[i].x = src[i].x / (Raw_Size_Q);
	dst[i].y = src[i].y / (Raw_Size_Q);
}
__global__ void Cal_tempout(cufftDoubleComplex* dst, cufftDoubleComplex* src, double kx, double ky, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	double x, y;
	x = double(i % Raw_Size_Q) - double(Raw_Size_Q / 2);
	y = double(i / Raw_Size_Q) - double(Raw_Size_Q / 2);

	cufftDoubleComplex tem;
	tem.x = cos(2.0 * pi_Loco * (kx / double(Raw_Size_Q) * x + ky / double(Raw_Size_Q) * y));
	tem.y = sin(2.0 * pi_Loco* (kx / double(Raw_Size_Q) * x + ky / double(Raw_Size_Q) * y));

	double temp;

	temp = src[i].x * tem.x - src[i].y * tem.y;

	dst[i].y = src[i].x * tem.y + src[i].y * tem.x;
	dst[i].x = temp;
}
__global__ void Add_type1(cufftDoubleComplex* dst, cufftDoubleComplex* src1, cufftDoubleComplex* src2)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;

	dst[i].x = src1[i].x + src2[i].x;
	dst[i].y = src1[i].y + src2[i].y;
}
__global__ void CudaMultyComplexWithOnedouble(cufftDoubleComplex* dst, cufftDoubleComplex* src1, double src2, int Raw_Size_Q) {
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	dst[i].x = src1[i].x * src2;
	dst[i].y = src1[i].y * src2;
}

__global__ void CudaCopyTypeOne(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Raw_Size_L)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	//int threadId_2D1 = Raw_Size_L / 2.0f + threadId_2D;
	//int blockId_2D1 = Raw_Size_L / 2.0f + blockId_2D;
	//int j = threadId_2D1 + 2 * (blockDim.x * blockDim.y) * blockId_2D1 + 4 * blockIdx.z * Raw_Size_L * Raw_Size_L;
	dst[i].x = src[i].x;
	dst[i].y = src[i].y;
}
__global__ void extend(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Raw_Size_L)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D + blockIdx.z * Raw_Size_L * Raw_Size_L;

	int threadId_2D1 = Raw_Size_L / 2.0f + threadId_2D;
	int blockId_2D1 = Raw_Size_L / 2.0f + blockId_2D;
	int j = threadId_2D1 + 2 * (blockDim.x * blockDim.y) * blockId_2D1 + 4 * blockIdx.z * Raw_Size_L * Raw_Size_L;
	dst[j].x = src[i].x;
	dst[j].y = src[i].y;
}
__global__ void fftShfit_3D_C(cufftDoubleComplex* dst, cufftDoubleComplex* src)//没有考虑2048*2048*3情况，只能做2048*2048，
{//到这 do_upsample fftShfit_3D_R 20240105写到这
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int blockId_2D0 = blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D + blockId_2D0;
	int x, y;
	int z_new;
	if (blockId_2D < blockDim.x * blockDim.y / 2)
	{
		if (threadId_2D > blockDim.x * blockDim.y / 2 - 1)
		{
			x = threadId_2D - blockDim.x * blockDim.y / 2;
		}
		else
		{
			x = threadId_2D + blockDim.x * blockDim.y / 2;
		}
		y = blockId_2D + gridDim.x * gridDim.y / 2;
	}
	else
	{
		if (threadId_2D > blockDim.x * blockDim.y / 2 - 1)
		{
			x = threadId_2D - blockDim.x * blockDim.y / 2;
		}
		else
		{
			x = threadId_2D + blockDim.x * blockDim.y / 2;
		}
		y = blockId_2D - gridDim.x * gridDim.y / 2;
	}


	if (blockIdx.z > (gridDim.z / 2 - 1))
	{
		z_new = blockIdx.z - gridDim.z / 2;

	}
	else
	{
		if (gridDim.z % 2 == 1)
		{
			z_new = blockIdx.z + gridDim.z / 2 + 1;
		}
		else
		{
			z_new = blockIdx.z + gridDim.z / 2;
		}
	}




	int blockId_2D1 = z_new * gridDim.y * gridDim.x * blockDim.x * blockDim.y;

	int j = x + (blockDim.x * blockDim.y) * y + blockId_2D1;
	dst[i].x = src[j].x;
	dst[i].y = src[j].y;




}
__global__ void fftn3D_1(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int x = i0 % Raw_Size_Q;
	int y = i0 / Raw_Size_Q;

	int j = y * Raw_Size_Q * Nz + x * Nz;


	cufftDoubleComplex datain;//>Nz

	for (int Testi = 0; Testi < Nz; Testi++)
	{


		int i = i0 + Testi * Raw_Size_Q * Raw_Size_Q;

		dst[i].x = src[j + Testi].x;
		dst[i].y = src[j + Testi].y;

	}





}
__global__ void fftn3D_0(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int x = i0 % Raw_Size_Q;
	int y = i0 / Raw_Size_Q;

	int j = y * Raw_Size_Q * Nz + x * Nz;


	cufftDoubleComplex datain;//>Nz

	for (int Testi = 0; Testi < Nz; Testi++)
	{


		int i = i0 + Testi * Raw_Size_Q * Raw_Size_Q;


		dst[j + Testi].x = src[i].x;
		dst[j + Testi].y = src[i].y;
	}





}
__global__ void ifftShfit_3D_R(double* dst, double* src)//一维
{//到这 do_upsample fftShfit_3D_R 20240105写到这

	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int blockId_2D0 = blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D + blockId_2D0;
	int x, y;
	int z_new;
	if (blockId_2D < blockDim.x * blockDim.y / 2)
	{
		if (threadId_2D > blockDim.x * blockDim.y / 2 - 1)
		{
			x = threadId_2D - blockDim.x * blockDim.y / 2;
		}
		else
		{
			x = threadId_2D + blockDim.x * blockDim.y / 2;
		}
		y = blockId_2D + gridDim.x * gridDim.y / 2;
	}
	else
	{
		if (threadId_2D > blockDim.x * blockDim.y / 2 - 1)
		{
			x = threadId_2D - blockDim.x * blockDim.y / 2;
		}
		else
		{
			x = threadId_2D + blockDim.x * blockDim.y / 2;
		}
		y = blockId_2D - gridDim.x * gridDim.y / 2;
	}


	if (blockIdx.z > (gridDim.z / 2 - 1))
	{
		z_new = blockIdx.z - gridDim.z / 2;

	}
	else
	{
		if (gridDim.z % 2 == 1)
		{
			z_new = blockIdx.z + gridDim.z / 2 + 1;
		}
		else
		{
			z_new = blockIdx.z + gridDim.z / 2;
		}
	}
	int blockId_2D1 = z_new * gridDim.y * gridDim.x * blockDim.x * blockDim.y;

	int j = x + (blockDim.x * blockDim.y) * y + blockId_2D1;
	dst[j] = src[i];





}
__global__ void Cal_do_upsampleimage_in(double* dst, double* src, int Num, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int j = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;
	int i = i0 + (blockIdx.z * 15 + Num) * Raw_Size_Q * Raw_Size_Q;
	dst[j] = src[i];


}
__global__ void Normalization_type5(double* dst, double* src, int Nz, double maxNum, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	for (int Nznum = 0; Nznum < Nz; Nznum++)
	{
		int i = i0 + Nznum * Raw_Size_Q * Raw_Size_Q;
		dst[i] = src[i] / maxNum;
	}



}
__device__ double Cal_notch_D(double kz, double qvector_xy, double notchheight, int z, int Nz, int i, int Raw_Size_Q)
{
	double dst;
	double qx, qy;
	qx = (i % Raw_Size_Q) - (Raw_Size_Q / 2);
	qy = (i / Raw_Size_Q) - (Raw_Size_Q / 2);

	double qz;
	//Tempz = qz * kz ;

	qz = (double(z) - floor(double(Nz) / 2.0));
	dst = (qx / qvector_xy) * (qx / qvector_xy) + (qy / qvector_xy) * (qy / qvector_xy) + (qz / kz/ notchheight) * (qz / kz/ notchheight)/**/;

	return dst;

}

__global__ void Cal_notch(double* dst, cufftDoubleComplex* OTFem, int z, int Nz, double kz, double qvector_xy, double notchdips, double notchwidthxy, double notchheight, double MaskNum, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int j = i + z * Raw_Size_Q * Raw_Size_Q;
	double qradsq = Cal_notch_D(kz, qvector_xy, notchheight, z, Nz, i, Raw_Size_Q);


	double OTFemAbs = sqrt(OTFem[i].x * OTFem[i].x + OTFem[i].y * OTFem[i].y);
	/**/
	dst[i] =  1 - notchdips * exp(-qradsq / 2.0 / (notchwidthxy* notchwidthxy));
	if (OTFemAbs < MaskNum)//Mask
	{
		dst[i] = 0;
	}
	//dst[i] = OTFem[i].x;
}


__global__ void Cal_clone_area(cufftDoubleComplex* dst, cufftDoubleComplex* src, int type, int Nz)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int blockId_2D0 = blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D + blockId_2D0;
	int jorderbis_y[5] = { 4, 3, 0, 1,2 };//jorderbis = [3 4 5 2 1];
	int jorderbis_x[5] = { 0,  2,  4,  1, 3 };//1,3,2,5,4=1,2,4,3,5;




	int xi = i % 5;
	//xi = jorderbis_x[xi];
	int yi = i / 5;
	yi = jorderbis_y[yi];
	yi = jorderbis_x[yi];
	int xj = xi + type * 5;
	int yj = yi + type * 5;



	int j = yj * Nz * 5 + xj;
	dst[j].x = src[i].x;
	dst[j].y = src[i].y;

}
__global__ void Cal_clone(cufftDoubleComplex* dst, cufftDoubleComplex* src)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int blockId_2D0 = blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D + blockId_2D0;

	dst[i].x = src[i].x;
	dst[i].y = src[i].y;

}
__global__ void Cal_mixing_matrix(cufftDoubleComplex* dst, cufftDoubleComplex* p1, double* Module, int numangles, double numsteps)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int xi = i % 5;
	int yi = i / 5;

	double patternphases = yi * 2 * pi_Loco / 5 - atan2(p1[numangles * 2].y, p1[numangles * 2].x);

	cufftDoubleComplex mixing_matrix;
	mixing_matrix.x = cos(-patternphases * (xi - 2)) / numsteps;
	mixing_matrix.y = sin(-patternphases * (xi - 2)) / numsteps;
	switch (xi)
	{
	case 0:
		dst[i].x = mixing_matrix.x * Module[3] / 2;
		dst[i].y = mixing_matrix.y * Module[3] / 2;
		break;
	case 1:
		dst[i].x = mixing_matrix.x * Module[1] / 2;
		dst[i].y = mixing_matrix.y * Module[1] / 2;
		break;
	case 2:
		dst[i].x = mixing_matrix.x;
		dst[i].y = mixing_matrix.y;
		break;
	case 3:
		dst[i].x = mixing_matrix.x * Module[1] / 2;
		dst[i].y = mixing_matrix.y * Module[1] / 2;
		break;
	case 4:
		dst[i].x = mixing_matrix.x * Module[3] / 2;
		dst[i].y = mixing_matrix.y * Module[3] / 2;
		break;
	}
}

__global__ void Cal_Zeros_C(cufftDoubleComplex* a)//处理核函数
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	a[i].x = 0.0;
	a[i].y = 0.0;
}
__global__ void Cal_module(double* dst, cufftDoubleComplex* src)
{


	double Temp_m1 = 0, Temp_m2 = 0;

	for (int qi = 0; qi < 3; qi++)
	{
		double Test_Temp_m1 = sqrt(src[0 + qi * 2].x * src[0 + qi * 2].x + src[0 + qi * 2].y * src[0 + qi * 2].y);
		if (Test_Temp_m1 > 1.0)
		{
			Test_Temp_m1 = 1.0;
		}
		Temp_m1 = Temp_m1 + Test_Temp_m1 / 3;
		Test_Temp_m1 = sqrt(src[1 + qi * 2].x * src[1 + qi * 2].x + src[1 + qi * 2].y * src[1 + qi * 2].y);
		if (Test_Temp_m1 > 1.0)
		{
			Test_Temp_m1 = 1.0;
		}
		Temp_m2 = Temp_m2 + Test_Temp_m1 / 3;
	}

	double dataparams_allmodule;
	dst[0] = 1;

	if (Temp_m1 < 0.2)
	{
		dst[1] = 0.2;
		dst[2] = 0.2;
	}
	else
	{
		dst[1] = Temp_m1;
		dst[2] = Temp_m1;
	}

	if (Temp_m2 < 0.3)
	{
		dst[3] = 0.3;
		dst[4] = 0.3;
	}
	else
	{
		dst[3] = Temp_m2;
		dst[4] = Temp_m2;
	}



}
__global__ void fourierShift(cufftDoubleComplex* dst, cufftDoubleComplex* vec, double kx, double ky, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;

	double x = (i0 % Raw_Size_Q) / double(Raw_Size_Q);
	double y = (i0 / Raw_Size_Q) / double(Raw_Size_Q);

	int xi = blockIdx.z % 10;
	int yi = blockIdx.z / 10;



	cufftDoubleComplex Temp;
	Temp.x = cos(2 * pi_Loco * (ky * y + kx * x));
	Temp.y = sin(2 * pi_Loco * (ky * y + kx * x));
	cufftDoubleComplex Temp2;
	Temp2.x = vec[i0].x;
	Temp2.y = vec[i0].y;
	dst[i].x = Temp.x * Temp2.x - Temp.y * Temp2.y;
	dst[i].y = Temp.x * Temp2.y + Temp.y * Temp2.x;

}

__global__ void Cal_b1s_dst(cufftDoubleComplex* dst, cufftDoubleComplex* b0, cufftDoubleComplex* b1, cufftDoubleComplex* Temp, double* scal, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;


	cufftDoubleComplex comshift;

	comshift.x = Temp[i0].x * b1[i0].x - Temp[i0].y * b1[i0].y;
	comshift.y = Temp[i0].x * b1[i0].y + Temp[i0].y * b1[i0].x;
	dst[i0].x = (comshift.x * b0[i0].x + comshift.y * b0[i0].y) / scal[0];
	dst[i0].y = (comshift.y * b0[i0].x - comshift.x * b0[i0].y) / scal[0];
}
__global__ void Cal_b1s_temp2(cufftDoubleComplex* dst, double tkx, double tky, double ts, int Blk_z, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + Blk_z * Raw_Size_Q * Raw_Size_Q;

	double x = double(i0 % Raw_Size_Q) / double(Raw_Size_Q);
	double y = double(i0 / Raw_Size_Q) / double(Raw_Size_Q);

	double kx = tkx + ((double(Blk_z % 10) - 4.5) / 4.5) * ts;
	double ky = tky + ((double(Blk_z / 10) - 4.5) / 4.5) * ts;


	dst[i0].x = cos(2.0 * pi_Loco * (ky * y + kx * x));
	dst[i0].y = sin(2.0 * pi_Loco * (ky * y + kx * x));
	//comshift[i0].x = Temp[i0].x * b1[i0].x - Temp[i0].y * b1[i0].y;
	//comshift[i0].y = Temp[i0].x * b1[i0].y + Temp[i0].y * b1[i0].x;
	//dst[i0].x =1 /*(comshift[i0].x * b0[i0].x + comshift[i0].y * b0[i0].y) / scal[0]*/;
	//dst[i0].y =2 /*(comshift[i0].y * b0[i0].x - comshift[i0].x * b0[i0].y) / scal[0]*/;
}
__global__ void Cal_Ones_C(cufftDoubleComplex* a)//处理核函数
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	a[i].x = 1.0;
	a[i].y = 0;
}
__global__ void Cal_Ones(double* a)//处理核函数
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	a[i] = 1.0;
}
__global__ void Abs_C_type2(double* dst, cufftDoubleComplex* src, int Num, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + Num * Raw_Size_Q * Raw_Size_Q;

	dst[i0] = src[i].x * src[i].x + src[i].y * src[i].y;

}
__global__ void Abs_C_type3(double* dst, cufftDoubleComplex* src)
{

	int i = blockIdx.z;

	dst[i] = sqrt(src[i].x * src[i].x + src[i].y * src[i].y);

}
__global__ void fftShfit_R_xy(double* dst, double* src, int x_0, int y_0, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int blockId_2D0 = blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D + blockId_2D0;
	int x, y;
	x = i % Raw_Size_Q + x_0;
	y = i / Raw_Size_Q + y_0;
	if (x < 0)
	{
		x = x + Raw_Size_Q;
	}
	if (x > Raw_Size_Q - 1)
	{
		x = x - Raw_Size_Q;
	}
	if (y < 0)
	{
		y = y + Raw_Size_Q;
	}
	if (y > Raw_Size_Q - 1)
	{
		y = y - Raw_Size_Q;
	}
	int j = x + (blockDim.x * blockDim.y) * y;
	dst[i] = src[j];


}
__global__ void CalMask_2_OR_C_type2(cufftDoubleComplex* dst, cufftDoubleComplex* src, double* ratio, double dist, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;



	if (ratio[i] < dist || ratio[i] > (1 - dist))//匹配mask=ceil(cpos)>otf.sampleLateral;
	{
		dst[i].x = 0;
		dst[i].y = 0;
	}
	else
	{
		dst[i].x = src[i].x;
		dst[i].y = src[i].y;
	}
}
__global__ void Divide(cufftDoubleComplex* dst, cufftDoubleComplex* src1, double* src2, int Raw_Size_Q) {
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;
	if (src2[i] == 0)
	{
		dst[i].x = src1[i].x;
		dst[i].y = src1[i].y;
	}
	else
	{
		dst[i].x = src1[i].x / src2[i];
		dst[i].y = src1[i].y / src2[i];
	}

}
__global__ void CalMask_2_OR_C(cufftDoubleComplex* dst, cufftDoubleComplex* src, double* weight0, double* wt0, double MaxNum, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;



	if (weight0[i] < MaxNum || wt0[i] < MaxNum)//匹配mask=ceil(cpos)>otf.sampleLateral;
	{
		dst[i].x = 0;
		dst[i].y = 0;
	}
	else
	{
		dst[i].x = src[i].x;
		dst[i].y = src[i].y;
	}
}
__global__ void CudaMultyDoubleWithOnedouble(double* dst, double* src1, double src2, int Raw_Size_Q) {
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	dst[i] = src1[i] * src2;
}
__global__ void Test(cufftDoubleComplex* dst, cufftDoubleComplex* src1, int Raw_Size_Q) {
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	dst[i].x = src1[i].x;
	dst[i].y = 0;
}
__global__ void Cal_Temp(double* dst, cufftDoubleComplex* src, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;
	double Abs = sqrt(src[i].x * src[i].x + src[i].y * src[i].y);
	dst[i] = log(1 + Abs);


}
__global__ void Normalization_type2(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Idx, int Num, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + Num * Raw_Size_Q * Raw_Size_Q;
	int MaxIdx = Idx + Num * Raw_Size_Q * Raw_Size_Q;

	double Normal = (src[MaxIdx].x * src[MaxIdx].x + src[MaxIdx].y * src[MaxIdx].y);
	double Temp = src[i].x;
	dst[i0].x = (src[i].x * src[MaxIdx].x + src[i].y * src[MaxIdx].y) / Normal;

	dst[i0].y = (src[i].y * src[MaxIdx].x - Temp * src[MaxIdx].y) / Normal;

}
__global__ void CudaMultyComplexWithConjugatecomplex(cufftDoubleComplex* dst, cufftDoubleComplex* src1, cufftDoubleComplex* src2, int Raw_Size_Q)//src1*conj(src2)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;
	cufftDoubleComplex src1_tem, src2_tem;
	src1_tem.x = src1[i].x;
	src1_tem.y = src1[i].y;
	src2_tem.x = src2[i].x;
	src2_tem.y = src2[i].y;
	double temp;
	temp = src1_tem.x * src2_tem.x + src1_tem.y * src2_tem.y;
	dst[i].y = src1_tem.y * src2_tem.x - src1_tem.x * src2_tem.y;
	dst[i].x = temp;
}
__global__ void Normalization(cufftDoubleComplex* dst_norm, cufftDoubleComplex* dst, cufftDoubleComplex* src, int Idx, int Num, int MaxNum, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + Num * Raw_Size_Q * Raw_Size_Q;
	int MaxIdx = Idx + MaxNum * Raw_Size_Q * Raw_Size_Q;
	double Normal = sqrt(src[MaxIdx].x * src[MaxIdx].x + src[MaxIdx].y * src[MaxIdx].y);
	dst[i0].x = src[i].x;
	dst[i0].y = src[i].y;

	dst_norm[i0].x = src[i].x / Normal;
	dst_norm[i0].y = src[i].y / Normal;

}
__global__ void Cal_pinvW(cufftDoubleComplex* dst)
{
	dst[0].x = 0.2;
	dst[0].y = 0;
	dst[1].x = 0.4;
	dst[1].y = 0;
	dst[2].x = 0.4;
	dst[2].y = 0;
	dst[3].x = 0.4;
	dst[3].y = 0;
	dst[4].x = 0.4;
	dst[4].y = 0;

	dst[5].x = 0.2;
	dst[5].y = 0;
	dst[6].x = 0.123606797749979;
	dst[6].y = 0.380422606518061;
	dst[7].x = 0.123606797749979;
	dst[7].y = -0.380422606518061;
	dst[8].x = -0.323606797749979;
	dst[8].y = 0.235114100916989;
	dst[9].x = -0.323606797749979;
	dst[9].y = -0.235114100916989;


	dst[10].x = 0.2;
	dst[10].y = 0;
	dst[11].x = -0.323606797749979;
	dst[11].y = 0.235114100916989;
	dst[12].x = -0.323606797749979;
	dst[12].y = -0.235114100916989;
	dst[13].x = 0.123606797749979;
	dst[13].y = -0.380422606518062;
	dst[14].x = 0.123606797749979;
	dst[14].y = 0.380422606518062;
	dst[15].x = 0.2;
	dst[15].y = 0;
	dst[16].x = -0.323606797749979;
	dst[16].y = -0.235114100916989;
	dst[17].x = -0.323606797749979;
	dst[17].y = 0.235114100916989;
	dst[18].x = 0.123606797749979;
	dst[18].y = 0.380422606518061;
	dst[19].x = 0.123606797749979;
	dst[19].y = -0.380422606518061;

	dst[20].x = 0.2;
	dst[20].y = 0;
	dst[21].x = 0.123606797749979;
	dst[21].y = -0.380422606518062;
	dst[22].x = 0.123606797749979;
	dst[22].y = 0.380422606518062;
	dst[23].x = -0.323606797749979;
	dst[23].y = -0.235114100916989;
	dst[24].x = -0.323606797749979;
	dst[24].y = 0.235114100916989;
}
__global__ void separateBandshifi(cufftDoubleComplex* dst, cufftDoubleComplex* src, cufftDoubleComplex* pinvW, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;

	int k1 = (blockIdx.z / 5) * 5;
	int k2 = (blockIdx.z / 5) * 5 + 1;
	int k3 = (blockIdx.z / 5) * 5 + 2;
	int k4 = (blockIdx.z / 5) * 5 + 3;
	int k5 = (blockIdx.z / 5) * 5 + 4;

	int i1 = i0 + k1 * Raw_Size_Q * Raw_Size_Q;
	int i2 = i0 + k2 * Raw_Size_Q * Raw_Size_Q;
	int i3 = i0 + k3 * Raw_Size_Q * Raw_Size_Q;
	int i4 = i0 + k4 * Raw_Size_Q * Raw_Size_Q;
	int i5 = i0 + k5 * Raw_Size_Q * Raw_Size_Q;
	int k = blockIdx.z % 5;

	dst[i].x = src[i1].x * pinvW[0 + k].x - src[i1].y * pinvW[0 + k].y +
		src[i2].x * pinvW[5 + k].x - src[i2].y * pinvW[5 + k].y +
		src[i3].x * pinvW[10 + k].x - src[i3].y * pinvW[10 + k].y +
		src[i4].x * pinvW[15 + k].x - src[i4].y * pinvW[15 + k].y +
		src[i5].x * pinvW[20 + k].x - src[i5].y * pinvW[20 + k].y;
	dst[i].y = src[i1].x * pinvW[0 + k].y + src[i1].y * pinvW[0 + k].x +
		src[i2].x * pinvW[5 + k].y + src[i2].y * pinvW[5 + k].x +
		src[i3].x * pinvW[10 + k].y + src[i3].y * pinvW[10 + k].x +
		src[i4].x * pinvW[15 + k].y + src[i4].y * pinvW[15 + k].x +
		src[i5].x * pinvW[20 + k].y + src[i5].y * pinvW[20 + k].x;

}
__global__ void getotfAtt(double* dst, double* rad, double cyclesPerMicron, double attfwhm, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + blockIdx.z * Raw_Size_Q * Raw_Size_Q;


	double cycl = rad[i] * cyclesPerMicron;

	dst[i] = (1 - exp(-pow(cycl, 4.0) / (2 * pow(attfwhm, 4.0))));

}
__global__ void fftShfit_C(cufftDoubleComplex* dst, cufftDoubleComplex* src)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
	int blockId_2D0 = blockIdx.z * gridDim.y * gridDim.x * blockDim.x * blockDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D + blockId_2D0;
	int x, y;

	if (blockId_2D < blockDim.x * blockDim.y / 2)
	{
		if (threadId_2D > blockDim.x * blockDim.y / 2 - 1) x = threadId_2D - blockDim.x * blockDim.y / 2;
		else x = threadId_2D + blockDim.x * blockDim.y / 2;
		y = blockId_2D + gridDim.x * gridDim.y / 2;
		int j = x + (blockDim.x * blockDim.y) * y + blockId_2D0;
		double a = 0;
		a = src[j].x;
		dst[j].x = src[i].x;
		dst[i].x = a;
		a = src[j].y;
		dst[j].y = src[i].y;
		dst[i].y = a;
	}
}
__global__ void CudaCalHypotRad(double* dst, double kx, double ky, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	double x, y;
	x = double(i % Raw_Size_Q) - (double(Raw_Size_Q / 2) + kx);
	y = double(i / Raw_Size_Q) - (double(Raw_Size_Q / 2) + ky);
	dst[i] = hypot(x, y);
}
__global__ void CalMask(double* dst, double* src, double* rad, double MaxNum, double MinNum, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;



	if (rad[i] > MaxNum)//匹配mask=ceil(cpos)>otf.sampleLateral;
	{
		dst[i] = 0;
	}
	else if (rad[i] < MinNum)
	{
		dst[i] = 0;
	}
	else
	{
		dst[i] = src[i];
	}
}
__global__ void writeOtfVector(double* dst, double* vals1, double* valsAtt, double* rad, int Raw_Size_Q, int att)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + blockDim.x * blockDim.y * blockId_2D;


	//double cpos = rad[i];//匹配cpos=pos+1;无需+1
	int lpos = floor(rad[i]);
	int hpos = ceil(rad[i]);
	double f = rad[i] - double(lpos);
	double retl, reth;

	if (att == 1)
	{
		retl = valsAtt[lpos] * (1 - f);
		reth = valsAtt[hpos] * f;
	}
	else
	{
		retl = vals1[lpos] * (1 - f);
		reth = vals1[hpos] * f;

	}
	dst[i] = retl + reth;

}
__global__ void fromEstimate(double* vals, double* valsOnlyAtt, double* valsAtt, int sampleLateral,
	double estimateAValue, double cyclesPerMicron, double attStrength, double attFWHM) {

	int i = blockIdx.x;
	double v = double(abs(i)) / double(sampleLateral);

	double va;
	if (v < 0 || v>1)
	{
		va = 0;
	}
	else
	{
		va = (1 / pi_Loco) * (2 * acos(v) - sin(2 * acos(v)));
	}
	vals[i] = va * pow(estimateAValue, v);

	double dist = abs(i) * cyclesPerMicron;

	valsOnlyAtt[i] = (1 - attStrength * (exp(-pow(dist, 2.0) / (pow(0.5 * attFWHM, 2.0)))));

	valsAtt[i] = vals[i] * valsOnlyAtt[i];


}
__global__ void weightCNR(double* dst, double* src, int Nz, double* MCNR, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	double sum = 0;
	for (int testi = 0; testi < Nz; testi++)
	{
		int i = i0 + (testi * 15) * Raw_Size_Q * Raw_Size_Q;
		sum = sum + src[i] * MCNR[testi];
	}
	dst[i0] = sum;
}
__global__ void averageMCNR_foreground_top(double* MCNR, int Nz)
{
	int i0 = blockIdx.z;

	switch (i0)
	{
	case 0:
		MCNR[i0] = 0;
		break;
	case 1:
		MCNR[i0] = 0;
		break;
	case 2:
		MCNR[i0] = 0;
		break;
	case 3:
		MCNR[i0] = 0;
		break;
	case 4:
		MCNR[i0] = 0.346989727181955;
		break;
	case 5:
		MCNR[i0] = 0;
		break;
	case 6:
		MCNR[i0] = 0.327915359483334;
		break;
	case 7:
		MCNR[i0] = 0;
		break;
	case 8:
		MCNR[i0] = 0;
		break;
	case 9:
		MCNR[i0] = 0.325094913334712;
		break;
	case 10:
		MCNR[i0] = 0;
		break;
	case 11:
		MCNR[i0] = 0;
		break;
	case 12:
		MCNR[i0] = 0;
		break;
	case 13:
		MCNR[i0] = 0;
		break;
	case 14:
		MCNR[i0] = 0;
		break;
	case 15:
		MCNR[i0] = 0;
		break;
	case 16:
		MCNR[i0] = 0;
		break;
	case 17:
		MCNR[i0] = 0;
		break;
	case 18:
		MCNR[i0] = 0;
		break;
	}
}
__global__ void do_OTFmasking3D_G(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Testi, double SupportSizex, double SupportSizey, double SupportSizez, double Dqx, double Dqy, double Dqz,
	double shiftqx, double shiftqy, double shiftqz, double NAl, double q0, cufftDoubleComplex OTFnorm, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int x = i0 % Raw_Size_Q;
	int y = i0 / Raw_Size_Q;

	double qx = -SupportSizex + Dqx / 2.0 + Dqx * x - shiftqx;
	double qy = -SupportSizey + Dqy / 2.0 + Dqy * y - shiftqy;
	double spatfreqsxy;

	spatfreqsxy = sqrt(qx * qx + qy * qy);
	double qcutoff_x;


	double fenmu;

	fenmu = (OTFnorm.x * OTFnorm.x + OTFnorm.y * OTFnorm.y);
	double qz = -SupportSizez + Dqz / 2.0 + Dqz * Testi - shiftqz;
	int k = i0 + Testi * Raw_Size_Q * Raw_Size_Q;




	if (spatfreqsxy < (2.0 * NAl))
	{
		double sum0, sum1;


		double Test1 = q0 * q0 - (spatfreqsxy - NAl) * (spatfreqsxy - NAl);
		if (Test1 > 0)
		{
			sum0 = sqrt(Test1);
		}
		else
		{
			sum0 = 0;
		}
		double Test2 = q0 * q0 - NAl * NAl;
		if (Test2 > 0)
		{
			sum1 = sqrt(Test2);
		}
		else
		{
			sum1 = 0;
		}
		qcutoff_x = (sum0 - sum1);
	}
	else
	{
		qcutoff_x = 0;

	}
	if (abs(qz) < (qcutoff_x + Dqz / 2.0))
	{
		double tem;
		tem = (src[k].x * OTFnorm.x + src[k].y * OTFnorm.y) / fenmu;
		dst[k].y = (src[k].y * OTFnorm.x - src[k].x * OTFnorm.y) / fenmu;
		dst[k].x = tem;
	}
	else
	{
		dst[k].x = 0;
		dst[k].y = 0;
	}

	if (dst[k].x > 0.4)
	{
		dst[k].x = 0.4;
		dst[k].y = 0;
	}

	//dst[k].x = src[k].x;
	//dst[k].y = src[k].y;







}
__global__ void Normalization_type4(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Nz, double maxNum, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	for (int Nznum = 0; Nznum < Nz; Nznum++)
	{
		int i = i0 + Nznum * Raw_Size_Q * Raw_Size_Q;
		dst[i].x = src[i].x / maxNum;
		dst[i].y = src[i].y / maxNum;
	}



}
__global__ void get_otf3d_cztfunc_dataout(cufftDoubleComplex* dst, cufftDoubleComplex* src, cufftDoubleComplex* B, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int x = i0 % Raw_Size_Q;
	int y = i0 / Raw_Size_Q;

	int j = y * Raw_Size_Q * (Nz * 2 - 1) + x * (Nz * 2 - 1);



	for (int Testi = 0; Testi < Nz - 1; Testi++)
	{

		//dst[j + Testi].x = D[Testi].x ;
		//dst[j + Testi].y = D[Testi].y ;
		src[j + Testi].x = src[j + Testi].x / (Nz * 2 - 1);//逆傅里叶变换需要
		src[j + Testi].y = src[j + Testi].y / (Nz * 2 - 1);
		int k = i0 + Testi * Raw_Size_Q * Raw_Size_Q;

		double Temp = src[j + Testi].x * B[Testi].x - src[j + Testi].y * B[Testi].y;
		dst[k].y = src[j + Testi].x * B[Testi].y + src[j + Testi].y * B[Testi].x;
		dst[k].x = Temp;
	}

	int k = i0 + (Nz - 1) * Raw_Size_Q * Raw_Size_Q;
	double Temp = src[j + 0].x * B[0].x - src[j + 0].y * B[0].y;
	dst[k].y = src[j + 0].x * B[0].y + src[j + 0].y * B[0].x;
	dst[k].x = Temp;




}
__global__ void get_otf3d_cztfunc_temp(cufftDoubleComplex* dst, cufftDoubleComplex* src, cufftDoubleComplex* D, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int x = i0 % Raw_Size_Q;
	int y = i0 / Raw_Size_Q;

	int j = y * Raw_Size_Q * (Nz * 2 - 1) + x * (Nz * 2 - 1);




	for (int Testi = 0; Testi < (Nz * 2 - 1); Testi++)
	{

		//dst[j + Testi].x = D[Testi].x ;
		//dst[j + Testi].y = D[Testi].y ;



		double Temp = src[j + Testi].x * D[Testi].x - src[j + Testi].y * D[Testi].y;
		dst[j + Testi].y = src[j + Testi].x * D[Testi].y + src[j + Testi].y * D[Testi].x;
		dst[j + Testi].x = Temp;

	}





}
__global__ void get_otf3d_cztfunc_cztin(cufftDoubleComplex* dst, cufftDoubleComplex* src, cufftDoubleComplex* A,
	int Nz, double zmin, double DzImage, double delqz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int x = i0 % Raw_Size_Q;
	int y = i0 / Raw_Size_Q;

	int j = y * Raw_Size_Q * (Nz * 2 - 1) + x * (Nz * 2 - 1);//申请Nz * 2 - 1个空间，实际本处用到Testi为Nz个


	cufftDoubleComplex datain;//>Nz

	for (int Testi = 0; Testi < Nz; Testi++)
	{
		double ZImage = zmin + DzImage / 2 + DzImage * Testi;
		cufftDoubleComplex tem;
		tem.x = cos(-2.0 * pi_Loco * (delqz * ZImage));
		tem.y = sin(-2.0 * pi_Loco * (delqz * ZImage));

		int i = i0 + Testi * Raw_Size_Q * Raw_Size_Q;

		datain.x = tem.x * src[i].x + tem.y * src[i].y;//squeeze(OTFstack(ii,jj,:))',虚部求相反数
		datain.y = -tem.x * src[i].y + tem.y * src[i].x;
		dst[j + Testi].x = datain.x * A[Testi].x - datain.y * A[Testi].y;
		dst[j + Testi].y = datain.x * A[Testi].y + datain.y * A[Testi].x;
		if (Testi != Nz - 1)
		{
			dst[j + Testi + Nz].x = 0;
			dst[j + Testi + Nz].y = 0;
		}



	}





}
__global__ void Normalization_type3(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Idx, int Num, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int i = i0 + Num * Raw_Size_Q * Raw_Size_Q;
	int MaxIdx = Idx + Num * Raw_Size_Q * Raw_Size_Q;

	double Normal = sqrt(src[MaxIdx].x * src[MaxIdx].x + src[MaxIdx].y * src[MaxIdx].y);
	double Temp = src[i].x;
	dst[i].x = src[i].x / Normal;

	dst[i].y = src[i].y / Normal;

}
__global__ void Abs_C(double* dst, cufftDoubleComplex* src, int Num, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int i = i0 + Num * Raw_Size_Q * Raw_Size_Q;

	dst[i0] = sqrt(src[i].x * src[i].x + src[i].y * src[i].y);

}
__global__ void Cal_PSFslice(cufftDoubleComplex* dst, double* src, double ImageSizex, double DxImage, double delqx, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int x = i % Raw_Size_Q;
	int y = i / Raw_Size_Q;

	double XImage = -ImageSizex + DxImage / 2.0 + double(x) * DxImage;
	double YImage = -ImageSizex + DxImage / 2.0 + double(y) * DxImage;


	dst[i].x = cos(-2.0 * pi_Loco * (delqx * XImage + delqx * YImage)) * src[i];
	dst[i].y = sin(-2.0 * pi_Loco * (delqx * XImage + delqx * YImage)) * src[i];


}
__global__ void Complex2Double(double* dst, cufftDoubleComplex* src, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	dst[i] = src[i].x;
}
__global__ void ifftR2RRec(cufftDoubleComplex* dst, cufftDoubleComplex* src, double Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	dst[i].x = src[i].x / (Raw_Size_Q * Raw_Size_Q);
	dst[i].y = src[i].y / (Raw_Size_Q * Raw_Size_Q);
}
__global__ void CudaMultyComplexWithDouble(cufftDoubleComplex* dst, cufftDoubleComplex* src1, double* src2, int Raw_Size_Q) {
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	dst[i].x = src1[i].x * src2[i];
	dst[i].y = src1[i].y * src2[i];
}
__global__ void Double2Complex(cufftDoubleComplex* dst, double* src)
{

	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;

	dst[i].x = src[i];
	dst[i].y = 0;
}
__global__ void CudaFftShfitComplexSizeafter(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Raw_Size_L0)
{
	int Raw_Size_L = 2 * Raw_Size_L0;
	int m = blockIdx.z % 4;
	int n = (blockIdx.z - blockIdx.z % 4) / 4;
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + m * gridDim.x * gridDim.y;
	int i0 = threadId_2D + blockDim.x * blockDim.y * blockId_2D;
	int i = i0 + Raw_Size_L * Raw_Size_L * n;


	int x, y;
	int threadId_2D1 = i0 % Raw_Size_L;
	int blockId_2D1 = (i0 - threadId_2D1) / Raw_Size_L;

	if (blockId_2D1 < Raw_Size_L0)
	{
		if (threadId_2D1 < Raw_Size_L0) x = threadId_2D1 + Raw_Size_L0;
		else x = threadId_2D1 - Raw_Size_L0;
		y = blockId_2D1 + Raw_Size_L0;
		int j = x + Raw_Size_L * y + Raw_Size_L * Raw_Size_L * n;
		double a = 0;
		a = src[j].x;
		src[j].x = src[i].x;
		src[i].x = a;
		a = src[j].y;
		src[j].y = src[i].y;
		src[i].y = a;
	}
}
__global__ void Cal_do_pixel_blurring_QxQy(double* dst, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int  x, y;
	x = i0 % Raw_Size_Q;
	y = i0 / Raw_Size_Q;

	double qxnorm = (double(x - double(Raw_Size_Q / 2))) / double(Raw_Size_Q);
	double qynorm = (double(y - double(Raw_Size_Q / 2))) / double(Raw_Size_Q);
	double sincqxnorm, sincqynorm;

	if (qxnorm == 0)
	{
		sincqxnorm = 1;
	}
	else
	{
		sincqxnorm = sin(pi_Loco * qxnorm) / (pi_Loco * qxnorm);
	}
	if (qynorm == 0)
	{
		sincqynorm = 1;
	}
	else
	{
		sincqynorm = sin(pi_Loco * qynorm) / (pi_Loco * qynorm);
	}

	dst[i0] = sincqxnorm * sincqynorm;
}
__global__ void Cal_Zeros_R(double* a)//处理核函数
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	a[i] = 0.0;

}
__global__ void Cal_Psf_multy(double* dst, cufftDoubleComplex* src, int Nz, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;


	double FieldMatrix = (src[i0].x * src[i0].x + src[i0].y * src[i0].y) / 3;


	int i = i0 + Nz * Raw_Size_Q * Raw_Size_Q;

	dst[i] = dst[i] + FieldMatrix;


}
__global__ void Cal_cztin(cufftDoubleComplex* dst, cufftDoubleComplex* datain, cufftDoubleComplex* Amt, int datain_size, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int  x, y;
	x = i0 % Raw_Size_Q;
	y = i0 / Raw_Size_Q;
	int x_New;
	x_New = x + Raw_Size_Q;
	int i, j;
	i = y * datain_size + x;
	j = y * datain_size + x_New;
	cufftDoubleComplex  cztin;
	cztin.x = datain[i0].x * Amt[x].x - datain[i0].y * Amt[x].y;
	cztin.y = datain[i0].x * Amt[x].y + datain[i0].y * Amt[x].x;
	dst[i].x = cztin.x;
	dst[i].y = cztin.y;
	if (x_New != datain_size)
	{
		dst[j].x = 0;
		dst[j].y = 0;
	}
}
__global__ void Cal_IntermediateImage(cufftDoubleComplex* dst, cufftDoubleComplex* cztout, cufftDoubleComplex* Bmt, int cztout_size, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int  x, y;
	x = i % Raw_Size_Q;
	y = i / Raw_Size_Q;



	int j = y * cztout_size + x;
	cztout[j].x = cztout[j].x / cztout_size;
	cztout[j].y = cztout[j].y / cztout_size;
	double Tem;

	int k = x * Raw_Size_Q + y;


	Tem = (cztout[j].x * Bmt[x].x - cztout[j].y * Bmt[x].y);
	dst[k].y = (cztout[j].x * Bmt[x].y + cztout[j].y * Bmt[x].x);
	dst[k].x = Tem;




}

__global__ void Cal_cztfunc_temp(cufftDoubleComplex* dst, cufftDoubleComplex* cztin, cufftDoubleComplex* Dmt, int cztin_size, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i0 = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int  x, y;
	x = i0 % Raw_Size_Q;
	y = i0 / Raw_Size_Q;
	int x_New;
	x_New = x + Raw_Size_Q;
	int i, j;
	i = y * cztin_size + x;
	j = y * cztin_size + x_New;
	double Tem;

	Tem = cztin[i].x * Dmt[x].x - cztin[i].y * Dmt[x].y;
	dst[i].y = cztin[i].x * Dmt[x].y + cztin[i].y * Dmt[x].x;
	dst[i].x = Tem;
	if (x_New != cztin_size)
	{

		Tem = cztin[j].x * Dmt[x_New].x - cztin[j].y * Dmt[x_New].y;
		dst[j].y = cztin[j].x * Dmt[x_New].y + cztin[j].y * Dmt[x_New].x;
		dst[j].x = Tem;
	}



}

__global__ void Cal_PupilFunction(cufftDoubleComplex* dst, cufftDoubleComplex* wavevector, double zemitrun, int itel, int jtel, int Raw_Size_Q)
{
	int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
	int blockId_2D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	int i = threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
	int  x, y;
	x = i % Raw_Size_Q;
	y = i / Raw_Size_Q;


	cufftDoubleComplex PositionPhaseMask;

	int i3 = i + 3 * Raw_Size_Q * Raw_Size_Q;
	double Wpos = 0;

	PositionPhaseMask.x = exp(-zemitrun * wavevector[i3].y) * cos(Wpos + zemitrun * wavevector[i3].x);
	PositionPhaseMask.y = exp(-zemitrun * wavevector[i3].y) * sin(Wpos + zemitrun * wavevector[i3].x);
	int j = itel * 3 + jtel + 4;
	int i4 = i + j * Raw_Size_Q * Raw_Size_Q;



	dst[i].x = PositionPhaseMask.x * wavevector[i4].x - PositionPhaseMask.y * wavevector[i4].y;;
	dst[i].y = PositionPhaseMask.x * wavevector[i4].y + PositionPhaseMask.y * wavevector[i4].x;

}
cv::Mat CalFuntion::double2Mat16(cv::Mat tes, int h, int w)//0-1double
{


	cv::Mat image(h, w, CV_16UC1);
	for (int i = 0; i < h; i++)
	{

		for (int j = 0; j < w; j++)
		{
			double test = tes.at< double>(i, j) * 255 * 255;
			image.at< ushort>(i, j) = (ushort)(test);
		}
	}
	return image;

}
void CalFuntion::get_normalization(double& normint_free, cv::Mat* PupilMatrix_x, cv::Mat* PupilMatrix_y, double pixelsize, double  NA, double lambda)
{


	double IntensityMatrix[3][3];
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			IntensityMatrix[i][j] = 0;
		}
	}
	cv::Mat Test[6];

	for (int itel = 0; itel < 3; itel++)
	{
		for (int jtel = 0; jtel < 3; jtel++)
		{
			for (int ztel = 0; ztel < 2; ztel++)
			{
				int pupmat1_Num = ztel * 3 + itel + 4;
				int pupmat2_Num = ztel * 3 + jtel + 4;


				;
				cv::Scalar ss = sum(PupilMatrix_x[pupmat1_Num].mul(PupilMatrix_x[pupmat2_Num]) + PupilMatrix_y[pupmat1_Num].mul(PupilMatrix_y[pupmat2_Num]));
				IntensityMatrix[itel][jtel] = IntensityMatrix[itel][jtel] + ss[0];


			}
		}
	}
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::printf("%f  ", IntensityMatrix[i][j]);
		}
		std::printf("\n");
	}
	double DxyPupil = 2.0 / PupilMatrix_y[0].rows;
	double normfac = DxyPupil * DxyPupil / ((pixelsize * NA / lambda) * (pixelsize * NA / lambda));
	normint_free = normfac * (IntensityMatrix[0][0] + IntensityMatrix[1][1] + IntensityMatrix[2][2]) / 3;



}


void CalFuntion::get_pupil_matrix_C(cv::Mat* result_x, cv::Mat* result_y, double NA, double lambda, double refimm, double refmed, double refcov, double fwd, double depth, double pixelsize, int Raw_Size_Q)
{
	double zvals_1, zvals_2, zvals_3;

	get_rimismatchpars(zvals_1, zvals_2, zvals_3, refimm, refcov, refmed, fwd, depth, NA);//refimmnom=refcov



	for (int ti = 0; ti < Raw_Size_Q; ti++)
	{
		for (int tj = 0; tj < Raw_Size_Q; tj++)
		{
			double XPupil, YPupil;
			XPupil = (1.0 / double(Raw_Size_Q)) + (2.0 / double(Raw_Size_Q)) * double(ti) - 1.0;
			YPupil = (1.0 / double(Raw_Size_Q)) + (2.0 / double(Raw_Size_Q)) * double(tj) - 1.0;


			double refimmnom = refcov;
			double CosThetaMed_0 = 1 - (XPupil * XPupil + YPupil * YPupil) * NA * NA / (refmed * refmed);
			double CosThetaCov_0 = 1 - (XPupil * XPupil + YPupil * YPupil) * NA * NA / (refcov * refcov);
			double CosThetaImm_0 = 1 - (XPupil * XPupil + YPupil * YPupil) * NA * NA / (refimm * refimm);



			cufftDoubleComplex CosThetaMed;
			cufftDoubleComplex CosThetaCov;
			cufftDoubleComplex CosThetaImm;


			if (CosThetaMed_0 < 0) { CosThetaMed.x = 0; CosThetaMed.y = sqrt(-CosThetaMed_0); }
			else { CosThetaMed.x = sqrt(CosThetaMed_0); CosThetaMed.y = 0; }

			if (CosThetaCov_0 < 0) { CosThetaCov.x = 0; CosThetaCov.y = sqrt(-CosThetaCov_0); }
			else { CosThetaCov.x = sqrt(CosThetaCov_0); CosThetaCov.y = 0; }

			if (CosThetaImm_0 < 0) { CosThetaImm.x = 0; CosThetaImm.y = sqrt(-CosThetaImm_0); }
			else { CosThetaImm.x = sqrt(CosThetaImm_0); CosThetaImm.y = 0; }




			double CosPhi = cos(atan2(YPupil, XPupil));
			double SinPhi = sin(atan2(YPupil, XPupil));

			double SinTheta = sqrt(1 - CosThetaMed_0);

			cufftDoubleComplex  FresnelPmedcov, FresnelPmedcov_0;
			FresnelPmedcov_0.x = refmed * CosThetaCov.x + refcov * CosThetaMed.x;
			FresnelPmedcov_0.y = refmed * CosThetaCov.y + refcov * CosThetaMed.y;
			FresnelPmedcov.x = 2.0 * FresnelPmedcov_0.x / (FresnelPmedcov_0.x * FresnelPmedcov_0.x + FresnelPmedcov_0.y * FresnelPmedcov_0.y);
			FresnelPmedcov.y = -2.0 * FresnelPmedcov_0.y / (FresnelPmedcov_0.x * FresnelPmedcov_0.x + FresnelPmedcov_0.y * FresnelPmedcov_0.y);

			cufftDoubleComplex  FresnelSmedcov, FresnelSmedcov_0;
			FresnelSmedcov_0.x = refmed * CosThetaMed.x + refcov * CosThetaCov.x;
			FresnelSmedcov_0.y = refmed * CosThetaMed.y + refcov * CosThetaCov.y;
			FresnelSmedcov.x = 2.0 * FresnelSmedcov_0.x / (FresnelSmedcov_0.x * FresnelSmedcov_0.x + FresnelSmedcov_0.y * FresnelSmedcov_0.y);
			FresnelSmedcov.y = -2.0 * FresnelSmedcov_0.y / (FresnelSmedcov_0.x * FresnelSmedcov_0.x + FresnelSmedcov_0.y * FresnelSmedcov_0.y);

			cufftDoubleComplex  FresnelScovimm, FresnelScovimm_1, FresnelScovimm_0;
			FresnelScovimm_0.x = refcov * CosThetaCov.x + refimm * CosThetaImm.x;
			FresnelScovimm_0.y = refcov * CosThetaCov.y + refimm * CosThetaImm.y;
			FresnelScovimm_1.x = 2.0 * refcov * FresnelScovimm_0.x / (FresnelScovimm_0.x * FresnelScovimm_0.x + FresnelScovimm_0.y * FresnelScovimm_0.y);
			FresnelScovimm_1.y = -2.0 * refcov * FresnelScovimm_0.y / (FresnelScovimm_0.x * FresnelScovimm_0.x + FresnelScovimm_0.y * FresnelScovimm_0.y);
			FresnelScovimm.x = CosThetaCov.x * FresnelScovimm_1.x - CosThetaCov.y * FresnelScovimm_1.y;
			FresnelScovimm.y = CosThetaCov.x * FresnelScovimm_1.y + CosThetaCov.y * FresnelScovimm_1.x;


			cufftDoubleComplex  FresnelPcovimm, FresnelPcovimm_1, FresnelPcovimm_0;
			FresnelPcovimm_0.x = refcov * CosThetaImm.x + refimm * CosThetaCov.x;
			FresnelPcovimm_0.y = refcov * CosThetaImm.y + refimm * CosThetaCov.y;
			FresnelPcovimm_1.x = 2.0 * refcov * FresnelPcovimm_0.x / (FresnelPcovimm_0.x * FresnelPcovimm_0.x + FresnelPcovimm_0.y * FresnelPcovimm_0.y);
			FresnelPcovimm_1.y = -2.0 * refcov * FresnelPcovimm_0.y / (FresnelPcovimm_0.x * FresnelPcovimm_0.x + FresnelPcovimm_0.y * FresnelPcovimm_0.y);
			FresnelPcovimm.x = CosThetaCov.x * FresnelPcovimm_1.x - CosThetaCov.y * FresnelPcovimm_1.y;
			FresnelPcovimm.y = CosThetaCov.x * FresnelPcovimm_1.y + CosThetaCov.y * FresnelPcovimm_1.x;





			cufftDoubleComplex  FresnelP, FresnelS;
			FresnelP.x = FresnelPmedcov.x * FresnelPcovimm.x - FresnelPmedcov.y * FresnelPcovimm.y;
			FresnelP.y = FresnelPmedcov.x * FresnelPcovimm.y + FresnelPmedcov.y * FresnelPcovimm.x;
			FresnelS.x = FresnelSmedcov.x * FresnelScovimm.x - FresnelSmedcov.y * FresnelScovimm.y;
			FresnelS.y = FresnelSmedcov.x * FresnelScovimm.y + FresnelSmedcov.y * FresnelScovimm.x;



			cufftDoubleComplex pvec[3];
			pvec[0].x = (FresnelP.x * CosThetaMed.x - FresnelP.y * CosThetaMed.y) * CosPhi;
			pvec[0].y = (FresnelP.x * CosThetaMed.y + FresnelP.y * CosThetaMed.x) * CosPhi;

			pvec[1].x = (FresnelP.x * CosThetaMed.x - FresnelP.y * CosThetaMed.y) * SinPhi;
			pvec[1].y = (FresnelP.x * CosThetaMed.y + FresnelP.y * CosThetaMed.x) * SinPhi;

			pvec[2].x = -FresnelP.x * SinTheta;
			pvec[2].y = -FresnelP.y * SinTheta;

			cufftDoubleComplex svec[3];
			svec[0].x = -FresnelS.x * SinPhi;
			svec[0].y = -FresnelS.y * SinPhi;
			svec[1].x = FresnelS.x * CosPhi;
			svec[1].y = FresnelS.y * CosPhi;
			svec[2].x = 0;
			svec[2].y = 0;

			cufftDoubleComplex PolarizationVector1[3];
			cufftDoubleComplex PolarizationVector2[3];
			for (int Pol = 0; Pol < 3; Pol++)
			{
				PolarizationVector1[Pol].x = CosPhi * pvec[Pol].x - SinPhi * svec[Pol].x;
				PolarizationVector1[Pol].y = CosPhi * pvec[Pol].y - SinPhi * svec[Pol].y;
				PolarizationVector2[Pol].x = SinPhi * pvec[Pol].x + CosPhi * svec[Pol].x;
				PolarizationVector2[Pol].y = SinPhi * pvec[Pol].y + CosPhi * svec[Pol].y;
			}

			double Test = XPupil * XPupil + YPupil * YPupil;
			double ApertureMask;
			if (Test < 1.0)
			{
				ApertureMask = 1.0;
			}
			else
			{
				ApertureMask = 0.0;
			}

			double Amplitude = ApertureMask * sqrt(CosThetaImm.x);
			cufftDoubleComplex Waberration;
			Waberration.x = zvals_1 * refimm * CosThetaImm.x - zvals_2 * refimmnom * CosThetaCov.x - zvals_3 * refmed * CosThetaMed.x;
			Waberration.y = zvals_1 * refimm * CosThetaImm.y - zvals_2 * refimmnom * CosThetaCov.y - zvals_3 * refmed * CosThetaMed.y;

			cufftDoubleComplex PhaseFactor;
			PhaseFactor.x = exp(-Waberration.y * 2.0 * pi_Loco / lambda) * cos(Waberration.x * 2.0 * pi_Loco / lambda);
			PhaseFactor.y = exp(-Waberration.y * 2.0 * pi_Loco / lambda) * sin(Waberration.x * 2.0 * pi_Loco / lambda);



			result_x[0].at<double>(ti, tj) = (2 * pi_Loco * NA / lambda) * XPupil;
			result_y[0].at<double>(ti, tj) = 0;
			result_x[1].at<double>(ti, tj) = (2 * pi_Loco * NA / lambda) * YPupil;
			result_y[1].at<double>(ti, tj) = 0;

			result_x[2].at<double>(ti, tj) = (2 * pi_Loco * refimm / lambda) * CosThetaImm.x;
			result_y[2].at<double>(ti, tj) = (2 * pi_Loco * refimm / lambda) * CosThetaImm.y;
			result_x[3].at<double>(ti, tj) = (2 * pi_Loco * refmed / lambda) * CosThetaMed.x;
			result_y[3].at<double>(ti, tj) = (2 * pi_Loco * refmed / lambda) * CosThetaMed.y;

			result_x[4].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector1[0].x - PhaseFactor.y * PolarizationVector1[0].y);
			result_y[4].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector1[0].y + PhaseFactor.y * PolarizationVector1[0].x);
			result_x[5].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector1[1].x - PhaseFactor.y * PolarizationVector1[1].y);
			result_y[5].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector1[1].y + PhaseFactor.y * PolarizationVector1[1].x);
			result_x[6].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector1[2].x - PhaseFactor.y * PolarizationVector1[2].y);
			result_y[6].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector1[2].y + PhaseFactor.y * PolarizationVector1[2].x);

			result_x[7].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector2[0].x - PhaseFactor.y * PolarizationVector2[0].y);
			result_y[7].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector2[0].y + PhaseFactor.y * PolarizationVector2[0].x);
			result_x[8].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector2[1].x - PhaseFactor.y * PolarizationVector2[1].y);
			result_y[8].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector2[1].y + PhaseFactor.y * PolarizationVector2[1].x);
			result_x[9].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector2[2].x - PhaseFactor.y * PolarizationVector2[2].y);
			result_y[9].at<double>(ti, tj) = Amplitude * (PhaseFactor.x * PolarizationVector2[2].y + PhaseFactor.y * PolarizationVector2[2].x);
		}
	}
	double normint_free;
	get_normalization(normint_free, result_x, result_y, pixelsize, NA, lambda);


	for (int i = 4; i < 10; i++)
	{
		result_x[i] = result_x[i] / sqrt(normint_free);
		result_y[i] = result_y[i] / sqrt(normint_free);
	}















}
void CalFuntion::prechirpz(cufftDoubleComplex* Dfft_G, cufftDoubleComplex* A_G, cufftDoubleComplex* B_G, double xsize, double qsize, double N, double M, sCudaPara& para)
{
	double L = N + M - 1;

	cufftDoubleComplex* A_C = new cufftDoubleComplex[M];
	cufftDoubleComplex* B_C = new cufftDoubleComplex[N];
	cv::Mat D(L, 1, CV_64FC2);
	cv::Mat D_C(L, 1, CV_64FC2);





	double sigma = 2.0 * pi_Loco * xsize * qsize / N / M;
	cufftDoubleComplex Afac, Bfac, sqW, W, Gfac;
	Afac.x = cos(2.0 * sigma * (1 - M));
	Afac.y = sin(2.0 * sigma * (1 - M));
	Bfac.x = cos(2.0 * sigma * (1 - N));
	Bfac.y = sin(2.0 * sigma * (1 - N));

	sqW.x = cos(2.0 * sigma);
	sqW.y = sin(2.0 * sigma);

	W.x = sqW.x * sqW.x - sqW.y * sqW.y;
	W.y = sqW.x * sqW.y + sqW.y * sqW.x;

	Gfac.x = (2.0 * xsize / N) * cos(sigma * (1 - N) * (1 - M));
	Gfac.y = (2.0 * xsize / N) * sin(sigma * (1 - N) * (1 - M));

	cufftDoubleComplex Utmp;
	Utmp.x = sqW.x * Afac.x - sqW.y * Afac.y;
	Utmp.y = sqW.x * Afac.y + sqW.y * Afac.x;

	A_C[0].x = 1; A_C[0].y = 0;
	for (int i = 1; i < M; i++)
	{
		A_C[i].x = Utmp.x * A_C[i - 1].x - Utmp.y * A_C[i - 1].y;
		A_C[i].y = Utmp.x * A_C[i - 1].y + Utmp.y * A_C[i - 1].x;
		double tep;
		tep = Utmp.x * W.x - Utmp.y * W.y;
		Utmp.y = Utmp.x * W.y + Utmp.y * W.x;
		Utmp.x = tep;

	}

	Utmp.x = sqW.x * Bfac.x - sqW.y * Bfac.y;
	Utmp.y = sqW.x * Bfac.y + sqW.y * Bfac.x;
	B_C[0].x = Gfac.x; B_C[0].y = Gfac.y;
	for (int i = 1; i < N; i++)
	{

		B_C[i].x = Utmp.x * B_C[i - 1].x - Utmp.y * B_C[i - 1].y;
		B_C[i].y = Utmp.x * B_C[i - 1].y + Utmp.y * B_C[i - 1].x;
		double tep;
		tep = Utmp.x * W.x - Utmp.y * W.y;
		Utmp.y = Utmp.x * W.y + Utmp.y * W.x;
		Utmp.x = tep;
	}


	cufftDoubleComplex Vtmp;
	Utmp.x = sqW.x; Utmp.y = sqW.y;
	Vtmp.x = 1.0; Vtmp.y = 0.0;

	for (int i = 0; i < M; i++)
	{
		D.at<cv::Vec2d>(i, 0)[0] = Vtmp.x;// D(i) = conj(Vtmp(i));
		D.at<cv::Vec2d>(i, 0)[1] = -Vtmp.y;



		double tep;
		tep = Utmp.x * Vtmp.x - Utmp.y * Vtmp.y;
		Vtmp.y = Utmp.x * Vtmp.y + Utmp.y * Vtmp.x;
		Vtmp.x = tep;
		tep = Utmp.x * W.x - Utmp.y * W.y;
		Utmp.y = Utmp.x * W.y + Utmp.y * W.x;
		Utmp.x = tep;
		D.at<cv::Vec2d>(int(L) - 1 - i, 0)[0] = Vtmp.x;// D(L+1-i) = conj(Vtmp(i+1));
		D.at<cv::Vec2d>(int(L) - 1 - i, 0)[1] = -Vtmp.y;

	}

	dft(D, D_C, cv::DFT_COMPLEX_OUTPUT);
	vector<cv::Mat> src1_channels;

	split(D_C, src1_channels);
	cv::Mat Dx_C_0 = src1_channels.at(0);
	cv::Mat Dx_C_1 = src1_channels.at(1);

	for (int i = 0; i < 1; i++)
	{
		double* h_DataRead = Dx_C_0.ptr<double>(0);//指向数据的指针
		double* h_DataRea_imag = Dx_C_1.ptr<double>(0);//指向数据的指针

		cufftDoubleComplex* h_DataComplex;//cuda复数结构体，含有x、y两个元素，x为实数，y为虚数
		const int dataH = L;
		const int dataW = 1;
		h_DataComplex = (cufftDoubleComplex*)malloc(dataH * dataW * sizeof(cufftDoubleComplex));//分配内存
		for (int i = 0; i < dataH * dataW; i++)
		{
			h_DataComplex[i].x = h_DataRead[i];
			h_DataComplex[i].y = h_DataRea_imag[i];
		}//赋值，相当于创建了complex
		cudaMemcpy(&Dfft_G[i * dataH * dataW], h_DataComplex, dataH * dataW * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	}


	for (int i = 0; i < 1; i++)
	{
		cufftDoubleComplex* h_DataComplex;//cuda复数结构体，含有x、y两个元素，x为实数，y为虚数
		const int dataH = N;
		const int dataW = 1;
		h_DataComplex = (cufftDoubleComplex*)malloc(dataH * dataW * sizeof(cufftDoubleComplex));//分配内存
		for (int i = 0; i < dataH * dataW; i++)
		{
			h_DataComplex[i].x = A_C[i].x;
			h_DataComplex[i].y = A_C[i].y;
		}//赋值，相当于创建了complex
		cudaMemcpy(&A_G[i * dataH * dataW], h_DataComplex, dataH * dataW * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < 1; i++)
	{
		cufftDoubleComplex* h_DataComplex;//cuda复数结构体，含有x、y两个元素，x为实数，y为虚数
		const int dataH = M;
		const int dataW = 1;
		h_DataComplex = (cufftDoubleComplex*)malloc(dataH * dataW * sizeof(cufftDoubleComplex));//分配内存
		for (int i = 0; i < dataH * dataW; i++)
		{
			h_DataComplex[i].x = B_C[i].x;
			h_DataComplex[i].y = B_C[i].y;
		}//赋值，相当于创建了complex
		cudaMemcpy(&B_G[i * dataH * dataW], h_DataComplex, dataH * dataW * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	}



}
void CalFuntion::get_rimismatchpars(double& zvals_1, double& zvals_2, double& zvals_3, double refimm, double  refimmnom, double refmed, double fwd, double depth, double NA)
{
	double refins[3];
	refins[0] = refimm; refins[1] = refimmnom; refins[2] = refmed;


	zvals_1 = 0; zvals_2 = fwd; zvals_3 = depth;
	if (NA > refmed)
	{
		NA = refmed;
	}
	double paraxiallimit = 0.2;
	double K = 3;//K = length(refins);
	double fsqav[3], fav[3];
	double Amat[3][3];
	double zvalsratio[3];
	double Wrmsratio[3][3];
	for (int itest = 0; itest < 3; itest++)
	{
		fsqav[itest] = 0;
		fav[itest] = 0;
		zvalsratio[itest] = 0;



		for (int jtest = 0; jtest < 3; jtest++)
		{
			Amat[itest][jtest] = 0;
			Wrmsratio[itest][jtest] = 0;
			Amat[itest][jtest] = 0;
		}
	}
	if (NA > paraxiallimit)
	{
		for (int jj = 0; jj < K; jj++)
		{
			fsqav[jj] = refins[jj] * refins[jj] - (1.0 / 2.0) * NA * NA;



			fav[jj] = (2.0 / 3.0 / (NA * NA)) * ((refins[jj] * refins[jj] * refins[jj] - pow((refins[jj] * refins[jj] - NA * NA), (3.0 / 2.0))));
			Amat[jj][jj] = fsqav[jj] - fav[jj] * fav[jj];
			for (int kk = 0; kk < K - 1; kk++)
			{
				Amat[jj][kk] = (1.0 / 4.0 / (NA * NA)) * (refins[jj] * refins[kk] * (refins[jj] * refins[jj] + refins[kk] * refins[kk]) -
					(refins[jj] * refins[jj] + refins[kk] * refins[kk] - 2.0 * NA * NA) * sqrt(refins[jj] * refins[jj] - NA * NA) * sqrt(refins[kk] * refins[kk] - NA * NA) +
					pow((refins[jj] * refins[jj] - refins[kk] * refins[kk]), 2.0) * log((sqrt(refins[jj] * refins[jj] - NA * NA) + sqrt(refins[kk] * refins[kk] - NA * NA)) / (refins[jj] + refins[kk])));
				Amat[jj][kk] = Amat[jj][kk] - fav[jj] * fav[kk];
				Amat[kk][jj] = Amat[jj][kk];

			}

		}
		for (int jv = 1; jv < 3; jv++)
		{
			zvalsratio[jv] = Amat[0][jv] / Amat[0][0];
			for (int kv = 1; kv < 3; kv++)
			{
				Wrmsratio[jv][kv] = Amat[jv][kv] - Amat[0][jv] * Amat[0][kv] / Amat[0][0];
			}
		}



	}
	else
	{
		for (int jv = 1; jv < K; jv++)
		{
			zvalsratio[jv] = refins[0] / refins[jv] + NA * NA * (refins[0] * refins[0] - refins[jv] * refins[jv]) / (4 * refins[0] * refins[jv] * refins[jv] * refins[jv]);
			for (int kv = 1; kv < K; kv++)
			{
				Wrmsratio[jv][kv] = pow(NA, 8.0) * (refins[1] * refins[1] - refins[jv] * refins[jv]) * (refins[1] * refins[1] - refins[kv] * refins[kv]) / (11520.0 * pow(refins[0], 4.0) * pow(refins[jv], 3.0) * pow(refins[kv], 3.0));
			}


		}
	}
	zvals_1 = zvalsratio[1] * zvals_2 + zvalsratio[2] * zvals_3;


}
void CalFuntion::get_field_matrix_C(cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By,
	double PupilSize, double Nx, double Ny, double* rawpixelsize, int Raw_Size_Q, sCudaPara& para)
{
	para.dataparamsCpu.ImageSizex = Nx * rawpixelsize[0] / 2.0 / 2.0;
	para.dataparamsCpu.ImageSizey = Ny * rawpixelsize[1] / 2.0 / 2.0;
	prechirpz(Dx, Ax, Bx, PupilSize, para.dataparamsCpu.ImageSizex, Nx, Nx, para);
	prechirpz(Dy, Ay, By, PupilSize, para.dataparamsCpu.ImageSizey, Nx, Ny, para);
}
void CalFuntion::get_throughfocusotf_C(cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By,
	double PupilSize, double Nx, double Ny, double* rawpixelsize, int Raw_Size_Q, sCudaPara& para)
{
	prechirpz(Dx, Ax, Bx, para.dataparamsCpu.ImageSizex, PupilSize, Nx, Nx, para);
	prechirpz(Dy, Ay, By, para.dataparamsCpu.ImageSizey, PupilSize, Nx, Ny, para);
}
void CalFuntion::get_pupil_matrix(cufftDoubleComplex* result, sCudaPara& para, int Raw_Size_Q)
{
	cv::Mat result_x[10];
	cv::Mat result_y[10];
	for (int i = 0; i < 10; i++)
	{
		result_x[i].create(Raw_Size_Q, Raw_Size_Q, CV_64FC1);
		result_y[i].create(Raw_Size_Q, Raw_Size_Q, CV_64FC1);
	}
	get_pupil_matrix_C(result_x, result_y, para.dataparamsCpu.NA, para.dataparamsCpu.lambda,
		para.dataparamsCpu.refimm, para.dataparamsCpu.refmed, para.dataparamsCpu.refcov, para.dataparamsCpu.fwd,
		para.dataparamsCpu.depth, para.dataparamsCpu.rawpixelsize[0] / 2.0, Raw_Size_Q);







	for (int i = 0; i < 10; i++)
	{
		double* h_DataRead = result_x[i].ptr<double>(0);//指向数据的指针
		double* h_DataRea_imag = result_y[i].ptr<double>(0);//指向数据的指针

		cufftDoubleComplex* h_DataComplex;//cuda复数结构体，含有x、y两个元素，x为实数，y为虚数
		const int dataH = ImgSizeAfter;
		const int dataW = ImgSizeAfter;
		h_DataComplex = (cufftDoubleComplex*)malloc(dataH * dataW * sizeof(cufftDoubleComplex));//分配内存
		for (int i = 0; i < dataH * dataW; i++)
		{
			h_DataComplex[i].x = h_DataRead[i];
			h_DataComplex[i].y = h_DataRea_imag[i];
		}//赋值，相当于创建了complex
		cudaMemcpy(&result[i * dataH * dataW], h_DataComplex, dataH * dataW * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	}





}
void CalFuntion::ShowDouble(double* paradouble, int dataH, int dataW, int id_start, int id_end, int jd_start, int jd_end, int Z_start, int Z_end)
{
	cv::Mat Retest;
	for (int ti = Z_start; ti < Z_end; ti++)
	{

		double* h_Result;
		h_Result = (double*)malloc(dataH * dataW * sizeof(double));
		//拷贝至内存
		cudaMemcpy(h_Result, &paradouble[ti * dataH * dataW], dataH * dataW * sizeof(double), cudaMemcpyDeviceToHost);
		//赋值给cv::Mat并打印
		cv::Mat_<double> resultReal = cv::Mat_<double>(dataH, dataW);
		cv::Mat_<double> resultImag = cv::Mat_<double>(dataH, dataW);
		for (int i = 0; i < dataH; i++) {
			double* rowPtrReal = resultReal.ptr<double>(i);
			for (int j = 0; j < dataW; j++) {
				rowPtrReal[j] = h_Result[i * dataW + j];
			}
		}
		resultReal.convertTo(Retest, CV_64FC1);
		free(h_Result);
		for (int i = id_start; i < id_end; i++)
		{
			for (int j = jd_start; j < jd_end; j++)
			{
				cout << Retest.at<double>(i, j) << "   ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}
void CalFuntion::ShowComplex(cufftDoubleComplex* paradouble, int dataH, int dataW, int id_start, int id_end, int jd_start, int jd_end, int zi, int zj)
{
	cv::Mat s_class_fDpf_real;
	cv::Mat s_class_fDpf_imag;
	for (int ti = zi; ti < zj; ti++)
	{

		cufftDoubleComplex* h_Result;
		h_Result = (cufftDoubleComplex*)malloc(dataH * dataW * sizeof(cufftDoubleComplex));
		//拷贝至内存
		cudaMemcpy(h_Result, &paradouble[ti * dataH * dataW], dataH * dataW * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
		//赋值给cv::Mat并打印
		cv::Mat_<double> resultReal = cv::Mat_<double>(dataH, dataW);
		cv::Mat_<double> resultImag = cv::Mat_<double>(dataH, dataW);
		for (int i = 0; i < dataH; i++) {
			double* rowPtrReal = resultReal.ptr<double>(i);
			double* rowPtrImag = resultImag.ptr<double>(i);
			for (int j = 0; j < dataW; j++) {
				rowPtrReal[j] = h_Result[i * dataW + j].x;
				rowPtrImag[j] = h_Result[i * dataW + j].y;

			}

		}
		resultReal.convertTo(s_class_fDpf_real, CV_64FC1);
		resultImag.convertTo(s_class_fDpf_imag, CV_64FC1);

		for (int i = id_start; i < id_end; i++)
		{
			for (int j = jd_start; j < jd_end; j++)
			{

				cout << s_class_fDpf_real.at<double>(i, j) << "+" << s_class_fDpf_imag.at<double>(i, j) << "   ";

			}
			cout << "\n";
		}
		cout << "\n";
		/*s_class_fDpf_real.convertTo(s_class_fDpf_real, CV_8UC1);

		imwrite("D:\\Test.tif", s_class_fDpf_real);*/

	}
}

void CalFuntion::get_modelOTF_G(cufftDoubleComplex* dst, sCudaPara& para, cufftDoubleComplex* wavevector, int Nz, int Raw_Size_Q)
{

}
void CalFuntion::transpose_cztfunc(cufftDoubleComplex* result, cufftDoubleComplex* PupilFunction, cufftDoubleComplex* A, cufftDoubleComplex* B, cufftDoubleComplex* D, int Raw_Size_Q, sCudaPara& para)
{
	cufftDoubleComplex* cztin;
	cudaMalloc((void**)&cztin, (1 * ImgSizeAfter * ImgSizeAfter * 2 - 1) * sizeof(cufftDoubleComplex));
	cufftHandle fft_loco;
	cufftPlan1d(&fft_loco, ImgSizeAfter * 2 - 1, CUFFT_Z2Z, ImgSizeAfter);
	Cal_cztin << <blocks7, threads >> > (cztin, PupilFunction, A, ImgSizeAfter * 2 - 1, ImgSizeAfter);
	cufftExecZ2Z(fft_loco, cztin, cztin, CUFFT_FORWARD);//fft(cztin,[],2);
	Cal_cztfunc_temp << <blocks7, threads >> > (cztin, cztin, D, ImgSizeAfter * 2 - 1, ImgSizeAfter);
	cufftExecZ2Z(fft_loco, cztin, cztin, CUFFT_INVERSE);//fft(cztin,[],2);
	Cal_IntermediateImage << <blocks7, threads >> > (result, cztin, B, ImgSizeAfter * 2 - 1, ImgSizeAfter);
	cudaFree(cztin);
	cufftDestroy(fft_loco);
}
void CalFuntion::get_field_matrix_G(double* dst, sCudaPara& para, cufftDoubleComplex* wavevector, cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By, int Nz, int Raw_Size_Q)
{
	cufftDoubleComplex* CTempOne_loco;
	cudaMalloc((void**)&CTempOne_loco, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
	double zmin = -Nz * para.dataparamsCpu.rawpixelsize[2] / 2.0;
	double zmax = Nz * para.dataparamsCpu.rawpixelsize[2] / 2.0;
	std::printf("Nz:%d  zmin:%f zmax:%f", Nz, zmin, zmax);
	double ImageSizez = (zmax - zmin) / 2.0;
	double DzImage = 2.0 * ImageSizez / Nz;
	double* ZImage = new double[int(Nz)];
	for (int Testj = 0; Testj < para.dataparamsCpu.Nz; Testj++)
	{
		Cal_Zeros_R << <blocks7, threads >> > (dst + (Testj)*ImgSizeAfter * ImgSizeAfter);

	}


	for (int i = 0; i < Nz; i++)
	{
		ZImage[i] = zmin + DzImage / 2 + DzImage * i;
		for (int itel = 0; itel < 2; itel++)
		{
			for (int jtel = 0; jtel < 3; jtel++)
			{
				Cal_PupilFunction << <blocks7, threads >> > (CTempOne_loco, wavevector, ZImage[i], itel, jtel, Raw_Size_Q);//wavevector->CTempFour

				transpose_cztfunc(CTempOne_loco, CTempOne_loco, Ay, By, Dy, Raw_Size_Q, para);
				transpose_cztfunc(CTempOne_loco, CTempOne_loco, Ax, Bx, Dx, Raw_Size_Q, para);

				Cal_Psf_multy << <blocks7, threads >> > (dst, CTempOne_loco, i, Raw_Size_Q);

				//20231229到这，测试正确FieldMatrix，后面是get_psf，对FieldMatrix的6个维度求和，因此FieldMatrix可以降低维度，只保留Nz维度即可。后续按照这个思路，在这里开始
			}
		}
	}

	cudaFree(CTempOne_loco);
}
void CalFuntion::get_throughfocusotf(cufftDoubleComplex* dst, double* PFS, sCudaPara& para)
{
	cufftDoubleComplex* Dx;
	cufftDoubleComplex* Ax;
	cufftDoubleComplex* Bx;
	cufftDoubleComplex* Dy;
	cufftDoubleComplex* Ay;
	cufftDoubleComplex* By;


	cudaMalloc((void**)&Ax, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Ay, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Bx, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&By, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Dx, (ImgSizeAfter * 2 - 1) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Dy, (ImgSizeAfter * 2 - 1) * sizeof(cufftDoubleComplex));

	get_throughfocusotf_C(Dx, Ax, Bx, Dy, Ay, By,
		(1.0 / 2.0 / (para.dataparamsCpu.rawpixelsize[0] / 2.0)), para.dataparamsCpu.Nx, para.dataparamsCpu.Ny, para.dataparamsCpu.rawpixelsize, ImgSizeAfter, para);

	get_throughfocusotf_G(dst, (para.dataparamsCpu.rawpixelsize[0] / 2.0), PFS, Dx, Ax, Bx, Dy, Ay, By, para.dataparamsCpu.Nz, para, ImgSizeAfter);

	cudaFree(Dx);
	cudaFree(Ax);
	cudaFree(Bx);
	cudaFree(Dy);
	cudaFree(Ay);
	cudaFree(By);


}
void CalFuntion::get_field_matrix(double * dst, cufftDoubleComplex* wavevector, sCudaPara& para)
{
	cufftDoubleComplex* Dx;
	cufftDoubleComplex* Ax;
	cufftDoubleComplex* Bx;
	cufftDoubleComplex* Dy;
	cufftDoubleComplex* Ay;
	cufftDoubleComplex* By;


	cudaMalloc((void**)&Ax, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Ay, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Bx, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&By, (ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Dx, (ImgSizeAfter * 2 - 1) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&Dy, (ImgSizeAfter * 2 - 1) * sizeof(cufftDoubleComplex));


	get_field_matrix_C(Dx, Ax, Bx, Dy, Ay, By,
		para.dataparamsCpu.NA / para.dataparamsCpu.lambda, para.dataparamsCpu.Nx, para.dataparamsCpu.Ny, para.dataparamsCpu.rawpixelsize, ImgSizeAfter, para);


	get_field_matrix_G(dst, para, wavevector, Dx, Ax, Bx, Dy, Ay, By, para.dataparamsCpu.Nz, ImgSizeAfter);//输出：para.dataparamsGpu.TempRealOne对应matlab PSF，内部：para.dataparamsGpu.CTempOne，para.dataparamsGpu.CTempTwo
	cudaFree(Dx);
	cudaFree(Ax);
	cudaFree(Bx);
	cudaFree(Dy);
	cudaFree(Ay);
	cudaFree(By);



}
void CalFuntion::get_psf(double* dst, double* PSFin, int Nz, sCudaPara& para, int Raw_Size_Q)
{
	double* pixelblurkernel;
	cudaMalloc((void**)&pixelblurkernel, (1 * ImgSizeAfter * ImgSizeAfter) * sizeof(double));
	Cal_do_pixel_blurring_QxQy << <blocks7, threads >> > (pixelblurkernel, ImgSizeAfter);


	cufftHandle fft_loco;
	//do_pixel_blurring
	int n[2];
	n[0] = ImgSizeAfter;
	n[1] = ImgSizeAfter;
	int istride = 1;
	int ostride = 1;
	int batch = 1;
	cufftPlanMany(&fft_loco, 2, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);

	cufftDoubleComplex* CTempOne_loco;
	cudaMalloc((void**)&CTempOne_loco, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));

	for (int i = 0; i < Nz; i++)
	{
		Double2Complex << <blocks7, threads >> > (CTempOne_loco, PSFin + i * Raw_Size_Q * Raw_Size_Q);
		cufftExecZ2Z(fft_loco, CTempOne_loco, CTempOne_loco, -1);
		CudaFftShfitComplexSizeafter << <blocks7, threads >> > (CTempOne_loco, CTempOne_loco, Raw_Size_Q / 2);
		CudaMultyComplexWithDouble << <blocks7, threads >> > (CTempOne_loco, CTempOne_loco, pixelblurkernel, Raw_Size_Q);
		CudaFftShfitComplexSizeafter << <blocks7, threads >> > (CTempOne_loco, CTempOne_loco, Raw_Size_Q / 2);
		cufftExecZ2Z(fft_loco, CTempOne_loco, CTempOne_loco, 1);
		ifftR2RRec << <blocks7, threads >> > (CTempOne_loco, CTempOne_loco, ImgSizeAfter);
		Complex2Double << <blocks7, threads >> > (dst + i * Raw_Size_Q * Raw_Size_Q, CTempOne_loco, Raw_Size_Q);

	}
	cudaFree(pixelblurkernel);
	cudaFree(CTempOne_loco);
	cufftDestroy(fft_loco);

}
void CalFuntion::get_throughfocusotf_G(cufftDoubleComplex* dst, double DxImage, double* PSFslice, cufftDoubleComplex* Dx, cufftDoubleComplex* Ax, cufftDoubleComplex* Bx, cufftDoubleComplex* Dy, cufftDoubleComplex* Ay, cufftDoubleComplex* By, int Nz, sCudaPara& para, int Raw_Size_Q)
{
	cufftDoubleComplex* CTempOne_loco;
	cudaMalloc((void**)&CTempOne_loco, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
	double* TempRealOne_loco;
	cudaMalloc((void**)&TempRealOne_loco, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(double));

	cublasHandle_t handle;
	cublasCreate(&handle);
	double delqx = (2.0 * (1.0 / 2.0 / (para.dataparamsCpu.rawpixelsize[0] / 2.0)) / double(Raw_Size_Q)) * (double(floor(Raw_Size_Q / 2.0)) + 1 - double(Raw_Size_Q + 1) / 2.0);
	for (int i = 0; i < Nz; i++)
	{
		Cal_PSFslice << <blocks7, threads >> > (CTempOne_loco, PSFslice + i * Raw_Size_Q * Raw_Size_Q, para.dataparamsCpu.ImageSizex, DxImage, delqx, Raw_Size_Q);
		transpose_cztfunc(CTempOne_loco, CTempOne_loco, Ay, By, Dy, Raw_Size_Q, para);
		transpose_cztfunc(CTempOne_loco, CTempOne_loco, Ax, Bx, Dx, Raw_Size_Q, para);
		int idx;
		
		Abs_C << <blocks7, threads >> > (TempRealOne_loco, CTempOne_loco, 0, ImgSizeAfter);//必须是0，ImgSizeAfter而不是i,ImgSizeAfter
		printf("Mark Point 1616 :***************************************\n");
		hengxiang = 510;
		zongxiang = 510;
		ShowDouble(TempRealOne_loco, ImgSizeAfter, ImgSizeAfter, zongxiang, zongxiang + 5, hengxiang + 0, hengxiang + 5, 0, 19);
		cublasIdamax(handle, ImgSizeAfter * ImgSizeAfter, TempRealOne_loco, 1, &idx);
		Normalization_type3 << <blocks7, threads >> > (dst + i * Raw_Size_Q * Raw_Size_Q, CTempOne_loco, idx - 1, 0, ImgSizeAfter);
		printf("test:%d", idx);
	}
	cudaFree(TempRealOne_loco);
	cudaFree(CTempOne_loco);
	cublasDestroy(handle);
}
void CalFuntion::get_otf3d(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int Nz, int Raw_Size_Q)
{
	cufftDoubleComplex* get_otf3d_D;
	cufftDoubleComplex* get_otf3d_A;
	cufftDoubleComplex* get_otf3d_B;
	cudaMalloc((void**)&get_otf3d_D, (para.dataparamsCpu.Nz * 2 - 1) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&get_otf3d_A, (para.dataparamsCpu.Nz) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&get_otf3d_B, (para.dataparamsCpu.Nz) * sizeof(cufftDoubleComplex));
	double zmin = -Nz * para.dataparamsCpu.rawpixelsize[2] / 2.0;
	double zmax = Nz * para.dataparamsCpu.rawpixelsize[2] / 2.0;
	double ImageSizez = (zmax - zmin) / 2.0;
	double DzImage = 2.0 * ImageSizez / Nz;
	double SupportSizez = 1.0 / 2.0 / para.dataparamsCpu.rawpixelsize[2];
	double DzSupport = 2.0 * SupportSizez / double(Nz);
	double delqz = (floor(double(Nz) / 2.0) + 1.0 - (double(Nz) + 1.0) / 2.0) * DzSupport;
	prechirpz(get_otf3d_D, get_otf3d_A, get_otf3d_B, ImageSizez, SupportSizez, Nz, Nz, para);//可能可以优化到initial中

	cufftDoubleComplex* CTempOne_loco;
	cudaMalloc((void**)&CTempOne_loco, (Nz * 2 - 1) * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));

	get_otf3d_cztfunc_cztin << <blocks7, threads >> > (CTempOne_loco, src, get_otf3d_A, Nz, zmin, DzImage, delqz, ImgSizeAfter);

	cufftHandle fft_loco;
	int batch = ImgSizeAfter;
	int istride = 1;
	int ostride = 1;
	int n[2];
	n[0] = para.dataparamsCpu.Nz * 2 - 1;
	n[1] = 1;
	cufftPlanMany(&fft_loco, 1, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	for (int i = 0; i < ImgSizeAfter; i++)
	{
		cufftExecZ2Z(fft_loco, CTempOne_loco + i * (Nz * 2 - 1) * ImgSizeAfter, CTempOne_loco + i * (Nz * 2 - 1) * ImgSizeAfter, CUFFT_FORWARD);//fft(cztin,[],2);//0104到这，fft错误，明天改为ImgSizeAfter*ImgSizeAfter的for循环

	}

	get_otf3d_cztfunc_temp << <blocks7, threads >> > (CTempOne_loco, CTempOne_loco, get_otf3d_D, Nz, ImgSizeAfter);

	for (int i = 0; i < ImgSizeAfter; i++)
	{
		cufftExecZ2Z(fft_loco, CTempOne_loco + i * (Nz * 2 - 1) * ImgSizeAfter, CTempOne_loco + i * (Nz * 2 - 1) * ImgSizeAfter, CUFFT_INVERSE);//fft(cztin,[],2);//0104到这，fft错误，明天改为ImgSizeAfter*ImgSizeAfter的for循环

	}
	cufftDoubleComplex* dataout;
	cudaMalloc((void**)&dataout, (Nz * 2 - 1) * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
	get_otf3d_cztfunc_dataout << <blocks7, threads >> > (dataout, CTempOne_loco, get_otf3d_B, Nz, ImgSizeAfter);
	cudaFree(CTempOne_loco);
	double norm = 0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	double* TempRealOne_loco;
	cudaMalloc((void**)&TempRealOne_loco, ImgSizeAfter * ImgSizeAfter * sizeof(double));
	for (int i = 0; i < Nz; i++)
	{
		Abs_C << <blocks7, threads >> > (TempRealOne_loco, dataout, i, ImgSizeAfter);
		int idx;
		cublasIdamax(handle, ImgSizeAfter * ImgSizeAfter, TempRealOne_loco, 1, &idx);
		double max_value;
		cudaMemcpy(&max_value, &TempRealOne_loco[idx - 1], sizeof(double), cudaMemcpyDeviceToHost);
		norm = max(norm, max_value);
	}
	cudaFree(TempRealOne_loco);
	Normalization_type4 << <blocks7, threads >> > (dst, dataout, Nz, norm, ImgSizeAfter);

	cudaFree(dataout);
	cublasDestroy(handle);
}
void CalFuntion::do_OTFmasking3D(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int Nz, int Raw_Size_Q)
{

	cufftDoubleComplex OTFnorm;
	cudaMemcpy(&OTFnorm, &src[ImgSizeAfter * ImgSizeAfter * (Nz / 2) + ImgSizeAfter * ImgSizeBefore + ImgSizeBefore], sizeof(double), cudaMemcpyDeviceToHost);//去psf中间值
	double shiftsupport[3];
	shiftsupport[0] = (double(floor(Raw_Size_Q / 2.0)) + 1.0 - double(Raw_Size_Q + 1.0) / 2.0);
	shiftsupport[1] = (double(floor(Raw_Size_Q / 2.0)) + 1.0 - double(Raw_Size_Q + 1.0) / 2.0);
	shiftsupport[2] = (double(floor(double(Nz) / 2.0)) + 1.0 - (double(Nz) + 1.0) / 2.0);


	double SupportSizex = 1.0 / 2.0 / (para.dataparamsCpu.rawpixelsize[0] / 2.0);
	double SupportSizey = 1.0 / 2.0 / (para.dataparamsCpu.rawpixelsize[1] / 2.0);
	double SupportSizez = 1.0 / 2.0 / para.dataparamsCpu.rawpixelsize[2];


	double Dqx = 2.0 * SupportSizex / double(ImgSizeAfter);
	double Dqy = 2.0 * SupportSizex / double(ImgSizeAfter);
	double Dqz = 2.0 * SupportSizez / double(Nz);

	double shiftqx = shiftsupport[0] * Dqx;
	double shiftqy = shiftsupport[1] * Dqy;
	double shiftqz = shiftsupport[2] * Dqz;
	std::printf("%f\n", shiftqx * 1000000);
	std::printf("%f\n", shiftqy * 1000000);
	std::printf("%f\n", shiftqz * 1000000);

	for (int Testi = 0; Testi < Nz; Testi++)
	{
		do_OTFmasking3D_G << <blocks7, threads >> > (dst, src, Testi,
			SupportSizex, SupportSizey, SupportSizez, Dqx, Dqy, Dqz, shiftqx, shiftqy, shiftqz,
			para.dataparamsCpu.NA / para.dataparamsCpu.lambda, para.dataparamsCpu.refmed / para.dataparamsCpu.lambda, OTFnorm, Nz, Raw_Size_Q);//部分内容，可以写到initial中
	}
	
}

void CalFuntion::SimOtfProvider(double* dst, sCudaPara& para, int Raw_Size_Q)
{
	double* vals;
	double* valsAtt;
	double* valsOnlyAtt;
	double* HypotRadOne;
	double* HypotRadFour;
	cudaMalloc((void**)&vals, 1 * 1 * para.dataparamsCpu.sampleLateral * sizeof(double));
	cudaMalloc((void**)&valsOnlyAtt, 1 * 1 * para.dataparamsCpu.sampleLateral * sizeof(double));
	cudaMalloc((void**)&valsAtt, 1 * 1 * para.dataparamsCpu.sampleLateral * sizeof(double));
	cudaMalloc((void**)&HypotRadOne, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	cudaMalloc((void**)&HypotRadFour, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(double));
	CudaCalHypotRad << <blocks, threads >> > (HypotRadOne, 0, 0, ImgSizeBefore);
	CudaCalHypotRad << <blocks7, threads >> > (HypotRadFour, 0, 0, ImgSizeAfter);
	dim3 blocks_loco(para.dataparamsCpu.sampleLateral, 1, 1);
	dim3 threads_loco(1, 1, 1);
	fromEstimate << <blocks_loco, threads_loco >> > (vals, valsOnlyAtt, valsAtt, para.dataparamsCpu.sampleLateral,
		para.dataparamsCpu.estimateAValue, para.dataparamsCpu.cyclesPerMicron, para.dataparamsCpu.attStrength, para.dataparamsCpu.attFWHM);

	if (Raw_Size_Q == ImgSizeBefore)
	{
		writeOtfVector << <blocks, threads >> > (dst, vals, valsAtt, HypotRadOne, ImgSizeBefore, 0);
		CalMask << <blocks, threads >> > (dst, dst, HypotRadOne, para.dataparamsCpu.cutoff / para.dataparamsCpu.cyclesPerMicron, 0, ImgSizeBefore);
	}
	else if (Raw_Size_Q == ImgSizeAfter)
	{
		writeOtfVector << <blocks7, threads >> > (dst, vals, valsAtt, HypotRadFour, ImgSizeAfter, 0);
		CalMask << <blocks7, threads >> > (dst, dst, HypotRadFour, para.dataparamsCpu.cutoff / para.dataparamsCpu.cyclesPerMicron, 0, ImgSizeAfter);

	}
	cudaFree(vals);
	cudaFree(valsAtt);
	cudaFree(valsOnlyAtt);
	cudaFree(HypotRadOne);
	cudaFree(HypotRadFour);
}
mxArray* CalFuntion::TifMat2mwArray_Double(cv::Mat src)//double Mat图片转double mwArray
{


	int h = src.rows;
	int w = src.cols;
	int c = src.channels();
	mxArray* pMat = NULL;
	double* input = NULL;
	if (c == 1) // gray image
	{
		mwSize dims[2] = { h, w };
		pMat = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
		input = (double*)mxGetData(pMat);
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w; j++)
			{
				double test = (double)src.at< double>(i, j);
				input[j * h + i] = test;
			}
		}
	}


	return pMat;
}
mxArray* CalFuntion::MatReal_2mwArray_Double(cv::Mat src)//double Mat图片转double mwArray
{


	int h = src.rows;
	int w = src.cols;
	int c = src.channels();
	mxArray* pMat = NULL;
	double* input = NULL;
	mwSize dims[2] = { h, w };
	pMat = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
	input = (double*)mxGetData(pMat);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{

			double test = src.at< double>(i, j);
			input[j * h + i] = test;
		}
	}


	return pMat;
}
mxArray* CalFuntion::Double2mwArray(double Num)
{

	mxArray* pMat = mxCreateDoubleMatrix(1, 1, mxREAL);
	double mat_lamda_d[1];
	mat_lamda_d[0] = Num;
	memcpy(mxGetPr(pMat), mat_lamda_d, 1 * sizeof(double)); //将数组x复制到mxarray数组xx中。



	return pMat;
}
cv::Mat CalFuntion::mwArry2doubleMat(mxArray* QQ)
{
	int n = mxGetNumberOfDimensions(QQ);
	int M = mxGetM(QQ);
	int N = mxGetN(QQ);
	double* imgData = NULL;
	imgData = (double*)mxGetPr(QQ);




	if (n == 2)
	{
		cv::Mat image(M, N, CV_64FC1);
		int h = M;
		int w = N;
		size_t subs[2]; // 三通道图像就需要 subs [3], 后续程序作相应修改

		for (int i = 0; i < h; i++)
		{
			subs[0] = i;
			for (int j = 0; j < w; j++)
			{
				subs[1] = j;
				int index = mxCalcSingleSubscript(QQ, 2, subs);
				double test = imgData[index];

				image.at< double>(i, j) = imgData[index];
			}
		}
		return image;
	}
	else if (n == 3)
	{
		int h = M;
		int w = N / 3;
		cv::Mat image(h, w, CV_64FC3);
		size_t subs[3]; // 三通道图像就需要 subs [3], 后续程序作相应修改

		for (int i = 0; i < h; i++)
		{
			subs[0] = i;
			for (int j = 0; j < w; j++)
			{
				subs[1] = j;
				for (int k = 0; k < 3; k++)
				{
					subs[2] = k;
					int index = mxCalcSingleSubscript(QQ, 3, subs);
					double test = imgData[index];
					image.at< cv::Vec3d>(i, j)[k] = imgData[index];
				}

			}
		}
		return image;
	}


}
void CalFuntion::Mat_deconvlucy(cv::Mat* dst, string srcName, int dataH, int dataW)
{
	

	mxArray* Out1 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out2 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out3 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out4 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out5 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out6 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out7 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out8 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out9 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out10 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out11 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out12 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out13 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out14 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	mxArray* Out15 = mxCreateDoubleMatrix(dataH, dataW, mxREAL);
	
	mxArray* mat_raw_file = mxCreateString((char*)srcName.c_str());
	mlfMatlab_deconvlucy(15, &Out1, &Out2, &Out3, &Out4, &Out5, &Out6, &Out7, &Out8, &Out9, &Out10, &Out11, &Out12, &Out13, &Out14, &Out15,mat_raw_file);
	dst[0] = mwArry2doubleMat(Out1);
	dst[1] = mwArry2doubleMat(Out2);
	dst[2] = mwArry2doubleMat(Out3);
	dst[3] = mwArry2doubleMat(Out4);
	dst[4] = mwArry2doubleMat(Out5);
	dst[5] = mwArry2doubleMat(Out6);
	dst[6] = mwArry2doubleMat(Out7);
	dst[7] = mwArry2doubleMat(Out8);
	dst[8] = mwArry2doubleMat(Out9);
	dst[9] = mwArry2doubleMat(Out10);
	dst[10] = mwArry2doubleMat(Out11);
	dst[11] = mwArry2doubleMat(Out12);
	dst[12] = mwArry2doubleMat(Out13);
	dst[13] = mwArry2doubleMat(Out14);
	dst[14] = mwArry2doubleMat(Out15);
}

void CalFuntion::Mat_edgetaper(cv::Mat& src)
{
	mxArray* In1;

	mxArray* Out1 = mxCreateDoubleMatrix(src.cols, src.rows, mxREAL);
	In1 = TifMat2mwArray_Double(src);

	
	mlfMat_edgetaper(1, &Out1, In1);
	src = mwArry2doubleMat(Out1);
	
}
void CalFuntion::edgetaper(std::vector<cv::Mat> &Snoisy)
{
	libMatlab_edgetaperInitialize();
	int PicNum=Snoisy.size();
	for (int i = 0; i < PicNum; i++)
	{
		Mat_edgetaper(Snoisy[i]);
	}
}
void CalFuntion::deconvlucy(double* result, string fileName, sCudaPara& para, int dataH, int dataW)
{
	libMatlab_deconvlucyInitialize();


	cv::Mat DSnoisy[15];
	
	Mat_deconvlucy(DSnoisy, fileName,dataH, dataW);
	for (int i = 0; i < 15; i++)
	{
		cudaMemcpy(&result[i * dataH * dataW], DSnoisy[i].data, dataH * dataW * sizeof(double), cudaMemcpyHostToDevice);
	}


}
void CalFuntion::FFT2D_15(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int inverse)
{
	cufftHandle fft_loco;
	int istride = 1;
	int ostride = 1;
	int n[2];
	n[0] = ImgSizeBefore;
	n[1] = ImgSizeBefore;
	int batch = 15;
	cufftPlanMany(&fft_loco, 2, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	cufftDoubleComplex* Temp;
	cudaMalloc((void**)&Temp, 15 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	if (inverse == 1)
	{

	}
	else
	{
		cufftExecZ2Z(fft_loco, src, Temp, -1);
		fftShfit_C << <blocks2, threads >> > (dst, Temp);
	}
	cudaFree(Temp);
	cufftDestroy(fft_loco);
	/*ifftR2RRec << <blocks, threads >> > (para.dataparamsGpu.CTempTwo, para.dataparamsGpu.CTempTwo, 1024);
	Complex2Double << <blocks, threads >> > (dst, para.dataparamsGpu.CTempTwo, 1024);*/


}
void CalFuntion::Mat2cufftDoubleComplex(cufftDoubleComplex* output, const cv::Mat& input)
{
	int dataH = input.cols;
	int dataW = input.rows;


	vector<cv::Mat> src1_channels;
	split(input, src1_channels);
	cv::Mat Dx_C_0 = src1_channels.at(0);
	cv::Mat Dx_C_1 = src1_channels.at(1);

	for (int i = 0; i < 1; i++)
	{
		double* h_DataRead = Dx_C_0.ptr<double>(0);//指向数据的指针
		double* h_DataRea_imag = Dx_C_1.ptr<double>(0);//指向数据的指针

		cufftDoubleComplex* h_DataComplex;//cuda复数结构体，含有x、y两个元素，x为实数，y为虚数
		h_DataComplex = (cufftDoubleComplex*)malloc(dataH * dataW * sizeof(cufftDoubleComplex));//分配内存
		for (int i = 0; i < dataH * dataW; i++)
		{
			h_DataComplex[i].x = h_DataRead[i];
			h_DataComplex[i].y = h_DataRea_imag[i];
		}//赋值，相当于创建了complex
		cudaMemcpy(&output[i * dataH * dataW], h_DataComplex, dataH * dataW * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
	}
}
cv::Mat CalFuntion::cufftDoubleComplex2Mat(cufftDoubleComplex* src, int dataH, int dataW)
{
	cv::Mat s_class_fDpf_real;
	cv::Mat s_class_fDpf_imag;
	cufftDoubleComplex* h_Result;
	h_Result = (cufftDoubleComplex*)malloc(dataH * dataW * sizeof(cufftDoubleComplex));
	//拷贝至内存
	cudaMemcpy(h_Result, &src[0], dataH * dataW * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	//赋值给cv::Mat并打印
	cv::Mat_<double> resultReal = cv::Mat_<double>(dataH, dataW);
	cv::Mat_<double> resultImag = cv::Mat_<double>(dataH, dataW);
	for (int i = 0; i < dataH; i++) {
		double* rowPtrReal = resultReal.ptr<double>(i);
		double* rowPtrImag = resultImag.ptr<double>(i);
		for (int j = 0; j < dataW; j++) {
			rowPtrReal[j] = h_Result[i * dataW + j].x;
			rowPtrImag[j] = h_Result[i * dataW + j].y;

		}

	}
	resultReal.convertTo(s_class_fDpf_real, CV_64FC1);
	resultImag.convertTo(s_class_fDpf_imag, CV_64FC1);

	cv::Mat planes[2] = { s_class_fDpf_real, s_class_fDpf_imag };
	cv::Mat dst;
	merge(planes, 2, dst);

	return dst;

}
mxArray* CalFuntion::Mat2mwArray_Double(cv::Mat src)//double Mat图片转double mwArray
{


	int h = src.rows;
	int w = src.cols;
	int c = src.channels();
	mxArray* pMat = NULL;
	double* input = NULL;
	if (c == 1) // gray image
	{
		mwSize dims[2] = { h, w };
		pMat = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
		input = (double*)mxGetData(pMat);
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w; j++)
			{

				double test = src.at< double>(i, j);
				input[j * h + i] = test;
			}
		}
	}
	else if (c == 3) // 3-channel image
	{
		mwSize dims[3] = { h, w, c };
		pMat = mxCreateNumericArray(c, dims, mxDOUBLE_CLASS, mxREAL);
		input = (double*)mxGetData(pMat);
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w; j++)
			{
				for (int ch = 0; ch < c; ch++)
				{
					double test = src.at<cv::Vec3d>(i, j)[ch];
					input[j * h + i + ch * h * w] = test;
				}
			}
		}
	}

	return pMat;
}


void CalFuntion::FFT2D_1(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para, int inverse)
{
	cufftHandle fft_loco;
	int batch = 1;
	int istride = 1;
	int ostride = 1;
	int n[2];
	n[0] = ImgSizeBefore;
	n[1] = ImgSizeBefore;
	cufftPlanMany(&fft_loco, 2, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	cufftDoubleComplex* Temp;
	cudaMalloc((void**)&Temp, ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));

	cufftHandle plan;
	cufftPlan2d(&plan, ImgSizeBefore, ImgSizeBefore, CUFFT_Z2Z);
	cublasHandle_t handle;
	cublasCreate(&handle);
	if (inverse == 1)
	{
		fftShfit_C << <blocks, threads >> > (Temp, src);
		cufftExecZ2Z(plan, Temp, Temp, CUFFT_INVERSE);
		ifftR2RRec << <blocks, threads >> > (dst, Temp, ImgSizeBefore);
	}
	else
	{
		cufftExecZ2Z(fft_loco, src, Temp, -1);
		fftShfit_C << <blocks, threads >> > (dst, Temp);
	}
	cudaFree(Temp);
	cufftDestroy(fft_loco);
	cublasDestroy(handle);
	/*ifftR2RRec << <blocks, threads >> > (para.dataparamsGpu.CTempTwo, para.dataparamsGpu.CTempTwo, 1024);
	Complex2Double << <blocks, threads >> > (dst, para.dataparamsGpu.CTempTwo, 1024);*/


}
void CalFuntion::find_illumination_pattern_Cal_C0C3(cufftDoubleComplex* result_fft, cufftDoubleComplex* result_Norm, cufftDoubleComplex* result, cufftDoubleComplex* src, double* NotchFilter1, sCudaPara& para, int Num, int MaxNum)
{
	int idx;
	cublasHandle_t handle;
	cublasCreate(&handle);

	double* Temp;
	cudaMalloc((void**)&Temp, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	Abs_C << <blocks, threads >> > (Temp, src, MaxNum, ImgSizeBefore);
	cublasIdamax(handle, ImgSizeBefore * ImgSizeBefore, Temp, 1, &idx);
	cudaFree(Temp);
	Normalization << <blocks, threads >> > (result_Norm, result, src, idx - 1, Num, MaxNum, ImgSizeBefore);


	CudaMultyComplexWithDouble << <blocks, threads >> > (result_fft, result_Norm, NotchFilter1, ImgSizeBefore);
	FFT2D_1(result_fft, result_fft, para, 0);
	cublasDestroy(handle);
}
void CalFuntion::commonRegion(cufftDoubleComplex* newb0, cufftDoubleComplex* newb1, cufftDoubleComplex* band0, cufftDoubleComplex* band1, cufftDoubleComplex* c0_Mask, cufftDoubleComplex* c3_Mask, double* otf, sCudaPara& para, double kx, double ky, double dist, double weightLimit)
{









	double* HypotRadOne;//建立，释放
	cudaMalloc((void**)&HypotRadOne, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	CudaCalHypotRad << <blocks, threads >> > (HypotRadOne, 0, 0, ImgSizeBefore);


	dim3 blocks_loco(para.dataparamsCpu.sampleLateral, 1, 1);
	dim3 threads_loco(1, 1, 1);

	double* vals;//建立，释放
	double* valsAtt;//建立，释放
	double* valsOnlyAtt;//建立，释放
	cudaMalloc((void**)&vals, 1 * 1 * para.dataparamsCpu.sampleLateral * sizeof(double));
	cudaMalloc((void**)&valsOnlyAtt, 1 * 1 * para.dataparamsCpu.sampleLateral * sizeof(double));
	cudaMalloc((void**)&valsAtt, 1 * 1 * para.dataparamsCpu.sampleLateral * sizeof(double));
	fromEstimate << <blocks_loco, threads_loco >> > (vals, valsOnlyAtt, valsAtt, para.dataparamsCpu.sampleLateral,
		para.dataparamsCpu.estimateAValue, para.dataparamsCpu.cyclesPerMicron, para.dataparamsCpu.attStrength, para.dataparamsCpu.attFWHM);
	cudaFree(valsOnlyAtt);


	double max = sqrt(double(kx * kx) + double(ky * ky));
	double* HypotRadTwo;//建立，释放
	cudaMalloc((void**)&HypotRadTwo, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	CudaCalHypotRad << <blocks, threads >> > (HypotRadTwo, kx, ky, ImgSizeBefore);

	double* wt0;//建立，释放
	cudaMalloc((void**)&wt0, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	writeOtfVector << <blocks, threads >> > (wt0, vals, valsAtt, HypotRadTwo, ImgSizeBefore, 0);
	CalMask << <blocks, threads >> > (wt0, wt0, HypotRadTwo, para.dataparamsCpu.cutoff / para.dataparamsCpu.cyclesPerMicron, 0, ImgSizeBefore);
	cudaFree(HypotRadTwo);

	double* HypotRadThree;//建立，释放
	cudaMalloc((void**)&HypotRadThree, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	CudaCalHypotRad << <blocks, threads >> > (HypotRadThree, -kx, -ky, ImgSizeBefore);
	double* wt1;//建立，释放
	cudaMalloc((void**)&wt1, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	writeOtfVector << <blocks, threads >> > (wt1, vals, valsAtt, HypotRadThree, ImgSizeBefore, 0);
	cudaFree(vals);
	cudaFree(valsAtt);

	CalMask << <blocks, threads >> > (wt1, wt1, HypotRadThree, para.dataparamsCpu.cutoff / para.dataparamsCpu.cyclesPerMicron, 0, ImgSizeBefore);
	double* Ratio_L;//建立，释放
	cudaMalloc((void**)&Ratio_L, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	CudaMultyDoubleWithOnedouble << <blocks, threads >> > (Ratio_L, HypotRadOne, 1.0 / max, ImgSizeBefore);
	cudaFree(HypotRadOne);
	double* Ratio_L_xy;//建立，释放
	cudaMalloc((void**)&Ratio_L_xy, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	CudaMultyDoubleWithOnedouble << <blocks, threads >> > (Ratio_L_xy, HypotRadThree, 1.0 / max, ImgSizeBefore);
	cudaFree(HypotRadThree);
	CalMask_2_OR_C << <blocks, threads >> > (c0_Mask, band0, otf, wt0, weightLimit, ImgSizeBefore);
	cudaFree(wt0);
	CalMask_2_OR_C << <blocks, threads >> > (c3_Mask, band1, otf, wt1, weightLimit, ImgSizeBefore);
	cudaFree(wt1);

	Divide << <blocks, threads >> > (c0_Mask, c0_Mask, otf, ImgSizeBefore);
	Divide << <blocks, threads >> > (c3_Mask, c3_Mask, otf, ImgSizeBefore);



	CalMask_2_OR_C_type2 << <blocks, threads >> > (newb0, c0_Mask, Ratio_L, dist, ImgSizeBefore);
	fftShfit_R_xy << <blocks, threads >> > (Ratio_L_xy, Ratio_L, round(kx), round(ky), ImgSizeBefore);
	cudaFree(Ratio_L);
	CalMask_2_OR_C_type2 << <blocks, threads >> > (newb1, c3_Mask, Ratio_L_xy, dist, ImgSizeBefore);
	cudaFree(Ratio_L_xy);











}
void CalFuntion::fitPeak(double& newKx, double& newKy, cufftDoubleComplex* c0_Norma, cufftDoubleComplex* c3_Norma, cufftDoubleComplex* c0_Mask, cufftDoubleComplex* c3_Mask, double* otf, sCudaPara& para, double kx, double ky, double weightLimit, double search)
{
	double dist = 0.15;
	cublasHandle_t handle;//建立，释放
	cublasCreate(&handle);
	double alpha, beta;
	cufftDoubleComplex alpha_C, beta_C;
	alpha = 1; beta = 0;
	alpha_C.x = 1; beta_C.x = 0;
	alpha_C.y = 0; beta_C.y = 0;
	double* Temp;//建立，释放
	cudaMalloc((void**)&Temp, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	double* Matrix_Ones;//建立，释放
	cufftDoubleComplex* Matrix_Ones_C;//建立，释放
	cudaMalloc((void**)&Matrix_Ones, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	cudaMalloc((void**)&Matrix_Ones_C, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
	Cal_Ones << <blocks, threads >> > (Matrix_Ones);
	Cal_Ones_C << <blocks7, threads >> > (Matrix_Ones_C);
	double* SumResult;//建立，释放
	cudaMalloc((void**)&SumResult, 1 * sizeof(double));
	cufftDoubleComplex* Temp2;//建立，释放
	cudaMalloc((void**)&Temp2, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));


	double* corr_abs;//建立，释放
	cufftDoubleComplex* corr;//建立，释放
	cudaMalloc((void**)&corr, 100 * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&corr_abs, 100 * sizeof(double));

	cufftDoubleComplex* Temp3;//建立，释放
	cudaMalloc((void**)&Temp3, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));

	for (int i = 0; i < 3; i++)//必须是3而非Nz
	{
		commonRegion(c0_Mask, c3_Mask, c0_Norma, c3_Norma, c0_Mask, c3_Mask, otf, para, kx, ky, dist, weightLimit);
		FFT2D_1(c0_Mask, c0_Mask, para, 1);//ifft正确
		FFT2D_1(c3_Mask, c3_Mask, para, 1);//ifft正确

		Abs_C_type2 << <blocks, threads >> > (Temp, c0_Mask, 0, ImgSizeBefore);
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, 1, ImgSizeBefore * ImgSizeBefore, &alpha,
			Temp, ImgSizeBefore * ImgSizeBefore, Matrix_Ones, 1, &beta, SumResult, 1);

		for (int Testi = 0; Testi < 100; Testi++)
		{
			Cal_b1s_temp2 << <blocks, threads >> > (Temp3, kx, ky, search, Testi, ImgSizeBefore);
			Cal_b1s_dst << <blocks, threads >> > (Temp2, c0_Mask, c3_Mask, Temp3, SumResult, ImgSizeBefore);


			
			cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, 1, ImgSizeBefore * ImgSizeBefore, &alpha_C,
				Temp2, ImgSizeBefore * ImgSizeBefore, Matrix_Ones_C, 1, &beta_C, corr + Testi, 1);
		}
		//Cal_b1s包含fourierShift

		Abs_C_type3 << <blocks6, threads3 >> > (corr_abs, corr);

		int idx;
		cublasIdamax(handle, 100, corr_abs, 1, &idx);

		std::printf("corr_abs_max: %d\n", idx);

		double xi = double((idx - 1) % 10);
		double yi = double((idx - 1) / 10);
		kx = kx + ((xi - 4.5) / 4.5) * search;
		ky = ky + ((yi - 4.5) / 4.5) * search;
		std::printf("peak: %f,%f,%f %f %f \n", xi, yi, kx, ky, search);





		search = search / 3;
	}
	newKx = kx;
	newKy = ky;
	cudaFree(Temp3);
	cudaFree(Temp);
	cudaFree(Matrix_Ones);
	cudaFree(Matrix_Ones_C);
	cudaFree(SumResult);
	cudaFree(Temp2);
	cudaFree(corr_abs);
	cudaFree(corr);


	cublasDestroy(handle);
}
void CalFuntion::getPeak(cufftDoubleComplex* dst, cufftDoubleComplex* src1, cufftDoubleComplex* src2, cufftDoubleComplex* c0_Mask, cufftDoubleComplex* c3_Mask, double* otf, sCudaPara& para, double kx, double ky, double weightLimit)
{



	//20231222做到这
	double dist = 0.15;
	commonRegion(c0_Mask, c3_Mask, src1, src2, c0_Mask, c3_Mask, otf, para, kx, ky, dist, weightLimit);

	FFT2D_1(c0_Mask, c0_Mask, para, 1);//
	FFT2D_1(c3_Mask, c3_Mask, para, 1);//

	fourierShift << <blocks, threads >> > (c0_Mask, c0_Mask, kx, ky, ImgSizeBefore);
	fourierShift << <blocks, threads >> > (c3_Mask, c3_Mask, kx * 2, ky * 2, ImgSizeBefore);
	CudaMultyComplexWithConjugatecomplex << <blocks, threads >> > (c3_Mask, c3_Mask, c0_Mask, ImgSizeBefore);


	double* Temp;//建立，释放
	cudaMalloc((void**)&Temp, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(double));
	Abs_C_type2 << <blocks, threads >> > (Temp, c0_Mask, 0, ImgSizeBefore);
	cublasHandle_t handle;//建立，释放
	cublasCreate(&handle);
	double alpha, beta;
	cufftDoubleComplex alpha_C, beta_C;
	alpha = 1; beta = 0;
	alpha_C.x = 1; beta_C.x = 0;
	alpha_C.y = 0; beta_C.y = 0;
	double* SumResult;//建立，释放
	cudaMalloc((void**)&SumResult, 10 * sizeof(double));
	double* Matrix_Ones;//建立，释放
	cufftDoubleComplex* Matrix_Ones_C;//建立，释放
	cudaMalloc((void**)&Matrix_Ones, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	cudaMalloc((void**)&Matrix_Ones_C, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
	Cal_Ones << <blocks, threads >> > (Matrix_Ones);
	Cal_Ones_C << <blocks7, threads >> > (Matrix_Ones_C);
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, 1, ImgSizeBefore * ImgSizeBefore, &alpha,
		Temp, ImgSizeBefore * ImgSizeBefore, Matrix_Ones, 1, &beta, SumResult, 1); //1/scal
	cufftDoubleComplex* corr;//建立，释放
	cudaMalloc((void**)&corr, 100 * sizeof(cufftDoubleComplex));
	cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, 1, ImgSizeBefore * ImgSizeBefore, &alpha_C,
		c3_Mask, ImgSizeBefore * ImgSizeBefore, Matrix_Ones_C, 1, &beta_C, corr, 1);


	Divide << <blocks3, threads3 >> > (dst, corr, SumResult, ImgSizeBefore);
	cudaFree(corr);
	cudaFree(SumResult);
	cudaFree(Temp);
	cudaFree(Matrix_Ones);
	cudaFree(Matrix_Ones_C);
	cublasDestroy(handle);

}
void CalFuntion::find_illumination_pattern(double* peak_kx, double* peak_ky, double* Snoisy, double SIMparams_SIMpixelsize, cufftDoubleComplex* dst, double* Module, sCudaPara& para)
{
	para.dataparamsCpu.cyclesPerMicron = double(1. / (para.dataparamsCpu.numpixelsx * SIMparams_SIMpixelsize * 0.001));
	para.dataparamsCpu.cutoff = 1000 / (0.5 * para.dataparamsCpu.lambda / para.dataparamsCpu.NA);
	para.dataparamsCpu.sampleLateral = ceil(para.dataparamsCpu.cutoff / para.dataparamsCpu.cyclesPerMicron) + 1;
	para.dataparamsCpu.estimateAValue = 1;
	std::printf("sampleLateral: %d\n", para.dataparamsCpu.sampleLateral);

	double* otf;//建立，释放
	cudaMalloc((void**)&otf, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	SimOtfProvider(otf, para, ImgSizeBefore);



	//otf2psf(para.dataparamsGpu.psf, otf, para);
	//real2Complex << <blocks, threads >> > (para.dataparamsGpu.psf, para.dataparamsGpu.CTempOne);
	//FFTss(para.dataparamsGpu.CTempOne, para.dataparamsGpu.CTempTwo, 3, -1, ImgSizeBefore);
	//complex2Real << <blocks, threads >> > (para.dataparamsGpu.CTempTwo, para.dataparamsGpu.psf);
	double* psf;//建立，未开辟

	double* IIraw;//建立，释放
	cudaMalloc((void**)&IIraw, 15 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	deconvlucy(IIraw, MainPath, para, ImgSizeBefore, ImgSizeBefore);
	cufftDoubleComplex* IIrawFFT;//建立，释放
	cudaMalloc((void**)&IIrawFFT, 15 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	Double2Complex << <blocks2, threads >> > (IIrawFFT, IIraw);
	cudaFree(IIraw);
	FFT2D_15(IIrawFFT, IIrawFFT, para, 0);


	double* HypotRadOne;//建立，释放
	cudaMalloc((void**)&HypotRadOne, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	CudaCalHypotRad << <blocks, threads >> > (HypotRadOne, 0, 0, ImgSizeBefore);

	double* NotchFilter0;//建立，释放
	double* NotchFilter1;//建立，释放
	double* NotchFilter2;//建立，释放
	cudaMalloc((void**)&NotchFilter0, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	cudaMalloc((void**)&NotchFilter1, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	cudaMalloc((void**)&NotchFilter2, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	getotfAtt << <blocks, threads >> > (NotchFilter0, HypotRadOne, para.dataparamsCpu.cyclesPerMicron, 0.5 * para.dataparamsCpu.cutoff, ImgSizeBefore);
	CalMask << <blocks, threads >> > (NotchFilter1, NotchFilter0, HypotRadOne, para.dataparamsCpu.cutoff / para.dataparamsCpu.cyclesPerMicron + 1, 0, ImgSizeBefore);
	CalMask << <blocks, threads >> > (NotchFilter2, NotchFilter0, HypotRadOne, 1.1 * (para.dataparamsCpu.cutoff / para.dataparamsCpu.cyclesPerMicron + 1), 0, ImgSizeBefore);


	cudaFree(HypotRadOne);
	cufftDoubleComplex* pinvW;//建立，释放
	cudaMalloc((void**)&pinvW, 1 * 5 * 5 * sizeof(cufftDoubleComplex));
	Cal_pinvW << <blocks3, threads3 >> > (pinvW);
	cufftDoubleComplex* separateII;//建立，释放
	cudaMalloc((void**)&separateII, 15 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	separateBandshifi << <blocks2, threads >> > (separateII, IIrawFFT, pinvW, ImgSizeBefore);//15组完成

	cudaFree(IIrawFFT);
	cudaFree(pinvW);

	cufftDoubleComplex* c0;//建立，释放
	cufftDoubleComplex* c2;//建立，释放
	cufftDoubleComplex* c3;//建立，释放
	cufftDoubleComplex* c0_Norma;//建立，释放
	cufftDoubleComplex* c3_Norma;//建立，释放
	cufftDoubleComplex* c0_Mask;//建立，释放
	cufftDoubleComplex* c3_Mask;//建立，释放
	cudaMalloc((void**)&c0, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&c2, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&c3, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&c0_Mask, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&c3_Mask, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&c0_Norma, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&c3_Norma, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	double* Temp;//建立，释放
	cudaMalloc((void**)&Temp, ImgSizeBefore * ImgSizeBefore * sizeof(double));
	cublasHandle_t handle;//建立，释放
	cublasCreate(&handle);
	cufftDoubleComplex* vec;
	cudaMalloc((void**)&vec, 1 * ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));
	cufftDoubleComplex* CTemp;//建立，释放
	cudaMalloc((void**)&CTemp, ImgSizeBefore * ImgSizeBefore * sizeof(cufftDoubleComplex));

	for (int i = 0; i < 3; i++)//必须是3而非Nz
	{
		int Num0 = 0 + 5 * i;
		int Num1 = 3 + 5 * i;
		int Num2 = 1 + 5 * i;
		find_illumination_pattern_Cal_C0C3(c0_Mask, c0_Norma, c0, separateII, NotchFilter1, para, Num0, Num0);//c0_Norma->separateII(:,:,1)/(max(max(abs(separateII(:,:,1)))))
		find_illumination_pattern_Cal_C0C3(c3_Mask, c3_Norma, c2, separateII, NotchFilter1, para, Num2, Num2);//只要para.dataparamsGpu.c2，其余的被下一句覆盖
		find_illumination_pattern_Cal_C0C3(c3_Mask, c3_Norma, c3, separateII, NotchFilter1, para, Num1, Num1);
		CudaMultyComplexWithConjugatecomplex << <blocks, threads >> > (c3_Mask, c3_Mask, c0_Mask, ImgSizeBefore);
		int idx;
		Abs_C << <blocks, threads >> > (Temp, c3_Mask, 0, ImgSizeBefore);
		cublasIdamax(handle, ImgSizeBefore * ImgSizeBefore, Temp, 1, &idx);
		Normalization_type2 << <blocks, threads >> > (c3_Mask, c3_Mask, idx - 1, 0, ImgSizeBefore);
		//CudaMultyComplexWithDouble << <blocks, threads >> > (c3_Mask, c3_Mask, NotchFilter1, ImgSizeBefore);

		FFT2D_1(c3_Mask, c3_Mask, para, 1);//已错ifft错误
		fftShfit_C << <blocks, threads >> > (vec, c3_Mask);
		CudaMultyComplexWithDouble << <blocks, threads >> > (CTemp, vec, NotchFilter2, ImgSizeBefore);
		Cal_Temp << <blocks, threads >> > (Temp, vec, ImgSizeBefore);

		cublasIdamax(handle, ImgSizeBefore * ImgSizeBefore, Temp, 1, &idx);//索引正确
		int yPos = (idx - 1) / ImgSizeBefore;//215
		int xPos = (idx - 1) % ImgSizeBefore;//527
		int ky = yPos - (ImgSizeBefore / 2);
		int kx = xPos - (ImgSizeBefore / 2);

		std::printf("yPos:%d  xPos:%d  ky:%d  kx:%d\n", yPos, xPos, ky, kx);
		double overlap = 0.15;
		double step = 2.5;
		fitPeak(peak_kx[i], peak_ky[i], c0_Norma, c3_Norma, c0_Mask, c3_Mask, otf, para, double(-kx), double(-ky), overlap, step);


		getPeak(dst + i * 2, c0, c2, c0_Mask, c3_Mask, otf, para, peak_kx[i] / 2, peak_ky[i] / 2, overlap);
		find_illumination_pattern_Cal_C0C3(c3_Mask, c3_Norma, c3, separateII, NotchFilter1, para, Num1, Num0);
		getPeak(dst + i * 2 + 1, c0_Norma, c3_Norma, c0_Mask, c3_Mask, otf, para, peak_kx[i], peak_ky[i], overlap);
	}

	Cal_module << <blocks3, threads3 >> > (Module, dst);


	cudaFree(otf);
	cudaFree(NotchFilter0);
	cudaFree(NotchFilter1);
	cudaFree(NotchFilter2);
	cudaFree(separateII);
	cudaFree(c0);
	cudaFree(c2);
	cudaFree(c3);
	cudaFree(c0_Norma);
	cudaFree(c3_Norma);
	cudaFree(c0_Mask);
	cudaFree(c3_Mask);
	cudaFree(Temp);
	cudaFree(CTemp);




	cublasDestroy(handle);

}
void CalFuntion::invert_device(cufftDoubleComplex* cu_5_a, cufftDoubleComplex* cu_5_o, sCudaPara& para)
{
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	int* INFO;
	cufftDoubleComplex** A_5_d;
	cufftDoubleComplex** C_5_d;
	int* P_5;
	cufftDoubleComplex* Mat_5_In_g[25];
	cufftDoubleComplex* Mat_5_Out_g[25];







	cudaMalloc((void**)&INFO, 1 * sizeof(int));
	cudaMalloc((void**)&P_5, 5 * sizeof(int));
	cudaMalloc((void***)&A_5_d, sizeof(Mat_5_In_g));
	cudaMalloc((void***)&C_5_d, sizeof(Mat_5_Out_g));






	for (int i = 0; i < 25; i++)
	{
		Mat_5_In_g[i] = &cu_5_a[i];
		Mat_5_Out_g[i] = &cu_5_o[i];
	}

	cudaMemcpy(A_5_d, Mat_5_In_g, sizeof(Mat_5_In_g), cudaMemcpyHostToDevice);
	cudaMemcpy(C_5_d, Mat_5_Out_g, sizeof(Mat_5_Out_g), cudaMemcpyHostToDevice);






	int n = 5;
	cublasZgetrfBatched(handle, n, A_5_d, n, P_5, INFO, 1);
	cublasZgetriBatched(handle, n, A_5_d, n, P_5, C_5_d, n, INFO, 1);


	cudaFree(INFO);

	cudaFree(A_5_d);
	cudaFree(C_5_d);
	cudaFree(P_5);
	cudaFree(Mat_5_In_g);
	cudaFree(Mat_5_Out_g);
	cublasDestroy_v2(handle);
};
void CalFuntion::Inv(cufftDoubleComplex* dst, cufftDoubleComplex* src, sCudaPara& para)
{
	cufftDoubleComplex* cu_5_a, *cu_5_o;
	cudaMalloc((void**)&cu_5_a, 25 * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&cu_5_o, 25 * sizeof(cufftDoubleComplex));

	Cal_clone << <blocks8, threads3 >> > (cu_5_a, src);
	invert_device(cu_5_a, cu_5_o, para);
	Cal_clone << <blocks8, threads3 >> > (dst, cu_5_o);
	cudaFree(cu_5_a);
	cudaFree(cu_5_o);
}
void CalFuntion::shift_2D(cufftDoubleComplex* dst, cufftDoubleComplex* src, double kx, double ky, int Testi2, sCudaPara& para)
{
	cufftHandle fft_loco;//建立，释放
	int istride = 1;
	int ostride = 1;
	int n[2];
	n[0] = ImgSizeAfter;
	n[1] = ImgSizeAfter;
	int batch = 1;
	cufftPlanMany(&fft_loco, 2, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	cufftDoubleComplex * CTempThree;//建立，释放
	cudaMalloc((void**)&CTempThree, ( ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));

	cufftExecZ2Z(fft_loco, src, CTempThree, CUFFT_INVERSE);
	ifftR2RRec << <blocks7, threads >> > (CTempThree, CTempThree, ImgSizeAfter);
	CudaFftShfitComplexSizeafter << <blocks7, threads >> > (CTempThree, CTempThree, ImgSizeBefore);
	
	Cal_tempout << <blocks7, threads >> > (CTempThree, CTempThree, kx, ky, ImgSizeAfter);
	CudaFftShfitComplexSizeafter << <blocks7, threads >> > (CTempThree, CTempThree, ImgSizeBefore);

	cufftExecZ2Z(fft_loco, CTempThree, dst, CUFFT_FORWARD);

	cudaFree(CTempThree);
	cufftDestroy(fft_loco);
}
void CalFuntion::shift_3D(cufftDoubleComplex* dst, cufftDoubleComplex* src, int Nz, double kx, double ky, double kz, sCudaPara& para, int Raw_Size_Q)
{
	cufftHandle fft_loco;//建立，释放
	int istride = 1;
	int ostride = 1;
	int n[2];
	n[0] = ImgSizeAfter;
	n[1] = ImgSizeAfter;
	int batch = 1;
	cufftPlanMany(&fft_loco, 2, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	cufftHandle fft_loco2;//建立，释放
	batch = ImgSizeAfter;
	istride = 1;
	ostride = 1;
	n[2];
	n[0] = para.dataparamsCpu.Nz;
	n[1] = 1;
	cufftPlanMany(&fft_loco2, 1, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	cufftDoubleComplex * CTempThree;//建立，释放
	cudaMalloc((void**)&CTempThree, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftDoubleComplex * CTempFour;//建立，释放
	cudaMalloc((void**)&CTempFour, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	for (int i = 0; i < Nz; i++)
	{
		cufftExecZ2Z(fft_loco, src + i * Raw_Size_Q* Raw_Size_Q,
			CTempThree + i * Raw_Size_Q * Raw_Size_Q, CUFFT_INVERSE);
		ifftR2RRec << <blocks7, threads >> > (CTempThree + i * Raw_Size_Q * Raw_Size_Q,
			CTempThree + i * Raw_Size_Q * Raw_Size_Q, ImgSizeAfter);
	}



	fftn3D_0 << <blocks7, threads >> > (CTempFour, CTempThree, para.dataparamsCpu.Nz, ImgSizeAfter);

	for (int Testi_3 = 0; Testi_3 < ImgSizeAfter; Testi_3++)
	{
		cufftExecZ2Z(fft_loco2, CTempFour + Testi_3 * Nz * ImgSizeAfter, CTempFour + Testi_3 * Nz * ImgSizeAfter, CUFFT_INVERSE);
	}
	fftn3D_1 << <blocks7, threads >> > (CTempThree, CTempFour, para.dataparamsCpu.Nz, ImgSizeAfter);
	for (int i = 0; i < Nz; i++)
	{
		ifftR2RRec_2 << <blocks7, threads >> > (CTempFour + i * Raw_Size_Q * Raw_Size_Q,
			CTempThree + i * Raw_Size_Q * Raw_Size_Q, Nz);
	}



	dim3 blocks_loco(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, para.dataparamsCpu.Nz * 4);
	ifftShfit_3D_C << <blocks_loco, threads >> > (CTempThree, CTempFour, ImgSizeAfter);//输入与输出一定不同变量

	for (int i = 0; i < Nz; i++)
	{
		Cal_shiftedFFT << <blocks7, threads >> > (CTempThree + i * ImgSizeAfter * ImgSizeAfter, CTempThree + i * ImgSizeAfter * ImgSizeAfter, 0, 0, kz, i, Nz, ImgSizeAfter);//shiftedFFT
	}



	fftShfit_3D_C_2 << <blocks_loco, threads >> > (CTempFour, CTempThree, ImgSizeAfter);//输入与输出一定不同变量//




	for (int i = 0; i < Nz; i++)
	{
		cufftExecZ2Z(fft_loco, CTempFour + i * Raw_Size_Q * Raw_Size_Q, CTempFour + i * Raw_Size_Q * Raw_Size_Q, CUFFT_FORWARD);
	}


	fftn3D_0 << <blocks7, threads >> > (CTempThree, CTempFour, para.dataparamsCpu.Nz, ImgSizeAfter);

	for (int Testi_3 = 0; Testi_3 < ImgSizeAfter; Testi_3++)
	{
		cufftExecZ2Z(fft_loco2, CTempThree + Testi_3 * Nz * ImgSizeAfter, CTempThree + Testi_3 * Nz * ImgSizeAfter, CUFFT_FORWARD);
	}
	fftn3D_1 << <blocks7, threads >> > (dst, CTempThree, para.dataparamsCpu.Nz, ImgSizeAfter);//19层从第6层后正确，原因是一维数组傅里叶变换的精度
	cudaFree(CTempThree);
	cudaFree(CTempFour);
	
	cufftDestroy(fft_loco);
	cufftDestroy(fft_loco2);
}

void CalFuntion::process_data(cufftDoubleComplex* dst1, double* dst2, double* kx, double* ky, double* allimages_in,cufftDoubleComplex* OTFem, cufftDoubleComplex* find_illumination_pattern_p1, double* Module, sCudaPara& para)//有冗余优化空间，Testj循环三次，三次计算相同取不同维度，extend。
{

	double average_pitch = 0;//doshift parameters
	for (int i = 0; i < 3; i++)//必须是3,不是Nz
	{
		average_pitch = average_pitch + sqrt((kx[i] / 2) * (kx[i] / 2) + (ky[i] / 2) * (ky[i] / 2)) / 3;
	}
	double kxy = ImgSizeAfter * para.dataparamsCpu.rawpixelsize[0] / 2 / average_pitch;
	double q0ex = para.dataparamsCpu.refmed / para.dataparamsCpu.exwavelength;
	double kz = q0ex - sqrt(q0ex * q0ex - 1 / (kxy * kxy));
	double qvector_3 = kz * para.dataparamsCpu.Nz * para.dataparamsCpu.rawpixelsize[2];
	double qvector_1, qvector_2;

	cufftDoubleComplex* unmixing_matrix;//建立，释放
	cufftDoubleComplex* mixing_matrix;//建立，释放
	cufftDoubleComplex* unmixing_matrix_15;//建立，释放



	cudaMalloc((void**)&unmixing_matrix, 25 * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&mixing_matrix, 25 * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&unmixing_matrix_15, para.dataparamsCpu.Nz * 5 * para.dataparamsCpu.Nz * 5 * sizeof(cufftDoubleComplex));
	dim3 blocks_loco(para.dataparamsCpu.Nz * 5, para.dataparamsCpu.Nz * 5, 1);

	Cal_Zeros_C << <blocks_loco, threads3 >> > (unmixing_matrix_15);

	cublasHandle_t handle;
	cublasCreate(&handle);
	double* notch;//建立，释放

	cufftDoubleComplex* OTFtemp1;//建立，释放
	cudaMalloc((void**)&OTFtemp1, para.dataparamsCpu.Nz*ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
	double * do_upsample_image_in;//建立，释放
	cudaMalloc((void**)&do_upsample_image_in, (para.dataparamsCpu.Nz* ImgSizeBefore* ImgSizeBefore) * sizeof(double));

	double * TempRealOne;//建立，释放
	cudaMalloc((void**)&TempRealOne, (para.dataparamsCpu.Nz* ImgSizeBefore* ImgSizeBefore) * sizeof(double));
	cufftDoubleComplex * CTempSix;//建立，释放
	cudaMalloc((void**)&CTempSix, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftDoubleComplex * CTempFive;//建立，释放
	cudaMalloc((void**)&CTempFive, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftDoubleComplex * CTempOne;//建立，释放
	cudaMalloc((void**)&CTempOne, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftDoubleComplex * CTempTwo;//建立，释放
	cudaMalloc((void**)&CTempTwo, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftHandle fft_loco;//建立，释放
	cufftHandle fft_loco2;//建立，释放
	int n[2];
	int istride = 1;
	int ostride = 1;
	n[0] = ImgSizeBefore;
	n[1] = ImgSizeBefore;
	int batch = para.dataparamsCpu.Nz;
	cufftPlanMany(&fft_loco, 2, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	batch = ImgSizeBefore;
	n[0] = para.dataparamsCpu.Nz;
	n[1] = 1;
	cufftPlanMany(&fft_loco2, 1, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	cufftDoubleComplex alpha_C, beta_C;
	alpha_C.x = 1; beta_C.x = 0;
	alpha_C.y = 0; beta_C.y = 0;

	cufftDoubleComplex * allftorderims;//建立，释放
	cufftDoubleComplex* Matrix_Ones_C;//建立，释放
	cudaMalloc((void**)&Matrix_Ones_C, 1 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
	Cal_Ones_C << <blocks7, threads >> > (Matrix_Ones_C);
	double* corr_abs;//建立，释放
	cufftDoubleComplex* corr;//建立，释放
	cudaMalloc((void**)&corr, 100 * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&corr_abs, 100 * sizeof(double));



	for (int Testi2 = 0; Testi2 < 3; Testi2++)//必须是3,不是Nz
	{
		qvector_1 = -kx[Testi2];
		qvector_2 = -ky[Testi2];
		double qvector_xy = sqrt((kx[Testi2] / 2) * (kx[Testi2] / 2) + (ky[Testi2] / 2) * (ky[Testi2] / 2)) / 2;
		std::printf("qvector_xy:%f\n", qvector_xy);
		printf("qvector_1:%f, qvector_2:%f, qvector_3:%f\n", qvector_1, qvector_2, qvector_3);
		//get_orders calculate the unmixing_matrix begin
		Cal_mixing_matrix << <blocks8, threads3 >> > (mixing_matrix, find_illumination_pattern_p1, Module, Testi2, 5);
		Inv(unmixing_matrix, mixing_matrix, para);
		for (int Testk = 0; Testk < para.dataparamsCpu.Nz; Testk++)
		{
			Cal_clone_area << <blocks8, threads3 >> > (unmixing_matrix_15, unmixing_matrix, Testk, para.dataparamsCpu.Nz);
		}
		//get_orders calculate the unmixing_matrix end
		//cal doshift OTFtemp1 begin
		int* idx = new int(para.dataparamsCpu.Nz);
		double maxidx = -999999;
		double maxidx0;

		cudaMalloc((void**)&notch, para.dataparamsCpu.Nz*ImgSizeAfter * ImgSizeAfter * sizeof(double));
		for (int Testi_5 = 0; Testi_5 < para.dataparamsCpu.Nz; Testi_5++)//计算Temp_1024为notch
		{
			Cal_notch << <blocks7, threads >> > (notch + Testi_5 * ImgSizeAfter * ImgSizeAfter,
				OTFem + Testi_5 * ImgSizeAfter * ImgSizeAfter, Testi_5,
				para.dataparamsCpu.Nz,
				qvector_3,
				qvector_xy,
				para.dataparamsCpu.notchdips1,
				para.dataparamsCpu.notchwidthxy1,
				1.0,
				0.001,
				ImgSizeAfter);

			int idxresult;
			cublasIdamax(handle, ImgSizeAfter * ImgSizeAfter, notch + Testi_5 * ImgSizeAfter * ImgSizeAfter, 1, &idxresult);
			cudaMemcpy(&maxidx0, notch + Testi_5 * ImgSizeAfter * ImgSizeAfter + idxresult - 1, 1 * sizeof(double), cudaMemcpyDeviceToHost);
			if (maxidx < maxidx0)
			{
				maxidx = maxidx0;
			}
		}


		Normalization_type5 << <blocks7, threads >> > (notch, notch, int(para.dataparamsCpu.Nz), maxidx, ImgSizeAfter);//计算Temp_1024为 notch./max(max(max(notch))); 
		
		for (int Testi_5 = 0; Testi_5 < para.dataparamsCpu.Nz; Testi_5++)
		{
			CudaMultyComplexWithDouble << <blocks7, threads >> > (OTFtemp1 + Testi_5 * ImgSizeAfter * ImgSizeAfter,
				OTFem + Testi_5 * ImgSizeAfter * ImgSizeAfter,
				notch + Testi_5 * ImgSizeAfter * ImgSizeAfter,
				ImgSizeAfter);//得到OTFtemp1给到CTempTwo
		}
		cudaFree(notch);
		//cal doshift OTFtemp1 end
		
		//do_upsample begin
		dim3 blocks_loco(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, para.dataparamsCpu.Nz);

		cudaMalloc((void**)&allftorderims, 5 * para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
		for (int Testi = 0; Testi < 5; Testi++)
		{
			Cal_do_upsampleimage_in << <blocks_loco, threads >> > (do_upsample_image_in, allimages_in, Testi + Testi2 * 5, para.dataparamsCpu.Nz, ImgSizeBefore);
			ifftShfit_3D_R << <blocks_loco, threads >> > (TempRealOne, do_upsample_image_in);
			Double2Complex << <blocks_loco, threads >> > (CTempSix, TempRealOne);
			cufftExecZ2Z(fft_loco, CTempSix, CTempSix, CUFFT_FORWARD);
			fftn3D_0 << <blocks, threads >> > (CTempFive, CTempSix, para.dataparamsCpu.Nz, ImgSizeBefore);
			int Nz = para.dataparamsCpu.Nz;
			for (int Testi_3 = 0; Testi_3 < ImgSizeBefore; Testi_3++)
			{
				cufftExecZ2Z(fft_loco2, CTempFive + Testi_3 * Nz * ImgSizeBefore, CTempFive + Testi_3 * Nz * ImgSizeBefore, CUFFT_FORWARD);
			}
			fftn3D_1 << <blocks, threads >> > (CTempSix, CTempFive, para.dataparamsCpu.Nz, ImgSizeBefore);
			fftShfit_3D_C << <blocks_loco, threads >> > (CTempFive, CTempSix);
			for (int Testj = 0; Testj < para.dataparamsCpu.Nz; Testj++)
			{
				Cal_Zeros_C << <blocks7, threads >> > (allftorderims + (Testi + Testj * 5) * ImgSizeAfter * ImgSizeAfter);
				extend << <blocks, threads >> > (allftorderims + (Testi + Testj * 5) * ImgSizeAfter * ImgSizeAfter, 
					CTempFive + Testj * ImgSizeBefore * ImgSizeBefore, ImgSizeBefore);
			}

		}

		//do_upsample end  without extend

		//get_orders begin with unmixing_matrix
		cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ImgSizeAfter * ImgSizeAfter, para.dataparamsCpu.Nz * 5, para.dataparamsCpu.Nz * 5, 
			&alpha_C, allftorderims, ImgSizeAfter * ImgSizeAfter, unmixing_matrix_15, para.dataparamsCpu.Nz * 5, &beta_C, allftorderims, 
			ImgSizeAfter * ImgSizeAfter);

		

		//get_orders end
		//doshift begin with extend

		for (int Testi_14 = 0; Testi_14 < para.dataparamsCpu.Nz * 5; Testi_14++)
		{
			para.dataparamsCpu.allftorderims[Testi_14] = cufftDoubleComplex2Mat(allftorderims + (Testi_14)* ImgSizeAfter * ImgSizeAfter, 
				ImgSizeAfter, ImgSizeAfter);
		}
		//20240403对的
		
		cudaFree(allftorderims);
		for (int jorder = 0; jorder < 5; jorder++)
		{
			for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
			{
				//CudaCopyTypeOne << <blocks7, threads >> > (CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, allftorderims + (jorder + jNz * 5) * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
				Mat2cufftDoubleComplex(CTempOne + (jNz* ImgSizeAfter * ImgSizeAfter), para.dataparamsCpu.allftorderims[jorder + jNz * 5]);
			}
				if (jorder == 0)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					CudaMultyComplexWithOnedouble << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, 
						OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, 1, ImgSizeAfter);
				}

			}
			if (jorder == 1)
			{
				

				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{


					shift_2D(CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, 
						qvector_1 / 2.0, qvector_2 / 2.0, Testi2, para);//CTempOne ftshiftorderims
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, 
						qvector_1 / 2.0, qvector_2 / 2.0, Testi2, para);
					CudaMultyComplexWithOnedouble << <blocks7, threads >> > (CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, 
						CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
					//printf("fftShfit_3D_C_2 here %d\n", 0);

				}
				

				shift_3D(CTempFive, CTempTwo, para.dataparamsCpu.Nz, 0, 0, qvector_3, para, ImgSizeAfter);
				shift_3D(CTempSix, CTempTwo, para.dataparamsCpu.Nz, 0, 0, -qvector_3, para, ImgSizeAfter);
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					Add_type1 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, 
						CTempFive + jNz * ImgSizeAfter * ImgSizeAfter, CTempSix + jNz * ImgSizeAfter * ImgSizeAfter);
				}
			}
			if (jorder == 2)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					shift_2D(CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, 
						-qvector_1 / 2.0, -qvector_2 / 2.0, Testi2, para);//CTempOne ftshiftorderims
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, 
						-qvector_1 / 2.0, -qvector_2 / 2.0, Testi2, para);
					CudaMultyComplexWithOnedouble << <blocks7, threads >> > (CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, 
						CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
					//CTempTwo OTFshift0
				}

				shift_3D(CTempFive, CTempTwo, para.dataparamsCpu.Nz, 0, 0, qvector_3, para, ImgSizeAfter);
				shift_3D(CTempSix, CTempTwo, para.dataparamsCpu.Nz, 0, 0, -qvector_3, para, ImgSizeAfter);
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					Add_type1 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, CTempFive + jNz * ImgSizeAfter * ImgSizeAfter, 
						CTempSix + jNz * ImgSizeAfter * ImgSizeAfter);
				}



			}
			if (jorder == 3)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					shift_2D(CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, qvector_1, qvector_2, Testi2, para);//CTempOne ftshiftorderims
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, qvector_1, qvector_2, Testi2, para);
					//CudaMultyComplexWithOnedouble << <blocks7, threads >> > (para.dataparamsGpu.CTempTwo + (jNz) * ImgSizeAfter * ImgSizeAfter, para.dataparamsGpu.CTempTwo + (jNz) * ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
					//CTempTwo OTFshift0
				}
			}
			if (jorder == 4)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					shift_2D(CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, 
						-qvector_1, -qvector_2, Testi2, para);//CTempOne ftshiftorderims
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, 
						-qvector_1, -qvector_2, Testi2, para);
					//CudaMultyComplexWithOnedouble << <blocks7, threads >> > (para.dataparamsGpu.CTempTwo + (jNz) * ImgSizeAfter * ImgSizeAfter, para.dataparamsGpu.CTempTwo + (jNz) * ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
					//CTempTwo OTFshift0
				}
			}
			printf("doshift jorder:%d\n", jorder);

			//20240403到这ftshiftorderims(:,:,5,1);等于CTempOne,jorder == 4,jNz=0;
			//20240403到这OTFshift0(:,:,5,1);等于CTempTwo,jorder == 4,jNz=0;
			//ftshiftorderims1->allftorderims，allftorderims被替代
			for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
			{
				para.dataparamsCpu.OTFshift0[jorder + jNz * 5] = cufftDoubleComplex2Mat(CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter,
					ImgSizeAfter, ImgSizeAfter);
				//para.dataparamsCpu.OTFshift0应该改为para.dataparamsCpu.ftshiftorderims

				//para.dataparamsCpu.OTFshift0存储ftshiftorderims
				//para.dataparamsCpu.allftorderims存储ftshiftorderims1
				Mul_type6 << <blocks7, threads >> > (CTempFive + (jNz) * ImgSizeAfter * ImgSizeAfter,
					CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter,
					CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
				para.dataparamsCpu.allftorderims[jorder + jNz * 5] = cufftDoubleComplex2Mat(CTempFive + (jNz)* ImgSizeAfter * ImgSizeAfter, 
					ImgSizeAfter, ImgSizeAfter);
				/*CudaCopyTypeOne << <blocks7, threads >> > (para.dataparamsGpu.CTempSeven + (jorder + jNz * 5) * ImgSizeAfter * ImgSizeAfter,
					para.dataparamsGpu.CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
				Mul_type6 << <blocks7, threads >> > (para.dataparamsGpu.CTempFive + (jorder + jNz * 5) * ImgSizeAfter * ImgSizeAfter,
					para.dataparamsGpu.CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter,
					para.dataparamsGpu.CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);*/
			}


			//ftshiftorderims1是CTempFive
			//allftorderims是matlab的ftshiftorderims1，显示以5为周期，比如Z=19，需要显示jorder=1，则显示1，1*5+1，2*5+1.......
			//计算ftshiftorderims1 end
			//计算OTFshift1 begin
			for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
			{

				if (jorder == 0)
				{
					CudaMultyComplexWithOnedouble << <blocks7, threads >> > (CTempSix, CTempTwo + (jNz) * ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
					Abs_Add_type1 << <blocks7, threads >> > (dst2 + (jNz) * ImgSizeAfter * ImgSizeAfter, 
						dst2 + (jNz) * ImgSizeAfter * ImgSizeAfter, CTempSix );

				}
				else
				{
					Abs_Add_type1 << <blocks7, threads >> > (dst2 + (jNz) * ImgSizeAfter * ImgSizeAfter, 
						dst2 + (jNz) * ImgSizeAfter * ImgSizeAfter, 
						CTempTwo + (jNz) * ImgSizeAfter * ImgSizeAfter);
				}
				

				
			}
			//计算OTFshift1 end
			//CTempFive,CTempSix为上面中间变量，已用完

		}

		
		//dst2为matlab主函数的OTFshiftfinal1

		//计算originfftshift begin
		cufftDoubleComplex Test;
		dim3 blocks_loco2(1, 1, para.dataparamsCpu.Nz);
		for (int jorder = 1; jorder < 5; jorder++)
		{

			for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
			{
				Mat2cufftDoubleComplex(CTempFive , para.dataparamsCpu.allftorderims[0 + jNz * 5]);
				Mat2cufftDoubleComplex(CTempSix, para.dataparamsCpu.allftorderims[jorder + jNz * 5]);

				Mul_type7 << <blocks7, threads >> > (CTempOne, CTempFive, CTempSix, ImgSizeAfter);
				cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 1, 1, ImgSizeAfter * ImgSizeAfter, 
					&alpha_C, CTempOne, ImgSizeAfter * ImgSizeAfter, Matrix_Ones_C, 1, &beta_C, corr + jNz, 1);

				



			}



			Cal_sum_angle << <blocks3, threads3 >> > (corr_abs, corr, para.dataparamsCpu.Nz, 0);
			
			for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
			{
				Mat2cufftDoubleComplex(CTempFive, para.dataparamsCpu.OTFshift0[jorder + jNz * 5]);

				Mul_type8 << <blocks7, threads >> > (CTempSix+ jNz * ImgSizeAfter * ImgSizeAfter,
					CTempFive, corr_abs, ImgSizeAfter);
				para.dataparamsCpu.allftorderims[jorder + jNz * 5] = cufftDoubleComplex2Mat(CTempSix + jNz * ImgSizeAfter * ImgSizeAfter, 
					ImgSizeAfter, ImgSizeAfter);


			}
			
		}
		for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
		{
			int jorder = 0;
			Mat2cufftDoubleComplex(CTempFive, para.dataparamsCpu.OTFshift0[jorder + jNz * 5]);
			CudaCopyTypeOne << <blocks7, threads >> > (CTempSix + jNz * ImgSizeAfter * ImgSizeAfter,
				CTempFive, ImgSizeAfter);
			para.dataparamsCpu.allftorderims[jorder + jNz * 5] = cufftDoubleComplex2Mat(CTempSix + jNz * ImgSizeAfter * ImgSizeAfter, 
				ImgSizeAfter, ImgSizeAfter);

		}
		

		//计算originfftshift end  allftorderims是 originfft
		//计算notchfilter begin  notch
		maxidx = -99999.0;
		//printf("maxidx:%f\n", maxidx);
		cudaMalloc((void**)&notch, para.dataparamsCpu.Nz*ImgSizeAfter * ImgSizeAfter * sizeof(double));
		for (int Testi_5 = 0; Testi_5 < para.dataparamsCpu.Nz; Testi_5++)//计算Temp_1024为notch
		{
			Cal_notch << <blocks7, threads >> > (notch + Testi_5 * ImgSizeAfter * ImgSizeAfter,
				OTFem + Testi_5 * ImgSizeAfter * ImgSizeAfter, Testi_5,
				para.dataparamsCpu.Nz,
				qvector_3,
				qvector_xy,
				para.dataparamsCpu.notchdips2,
				para.dataparamsCpu.notchwidthxy2,
				para.dataparamsCpu.notchheight,
				0.01,
				ImgSizeAfter);

			int idxresult;
			cublasIdamax(handle, ImgSizeAfter * ImgSizeAfter, notch + Testi_5 * ImgSizeAfter * ImgSizeAfter, 1, &idxresult);
			cudaMemcpy(&maxidx0, notch + Testi_5 * ImgSizeAfter * ImgSizeAfter + idxresult - 1, 1 * sizeof(double), cudaMemcpyDeviceToHost);
			if (maxidx < maxidx0)
			{
				maxidx = maxidx0;
			}
			//printf("maxidx:%f\n", maxidx);
		}
		
		Normalization_type5 << <blocks7, threads >> > (notch, notch, int(para.dataparamsCpu.Nz), maxidx, ImgSizeAfter);//计算Temp_1024为 notch./max(max(max(notch))); 
		for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
		{
			Double2Complex << <blocks7, threads >> > (OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter,
				notch + jNz * ImgSizeAfter * ImgSizeAfter);
		}
		cudaFree(notch);

		//计算notchfilter begin
		for (int jorder = 0; jorder < 5; jorder++)
		{
			for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
			{
				Mat2cufftDoubleComplex(CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, para.dataparamsCpu.allftorderims[jorder + jNz * 5]);
			}
		
			if (jorder == 0)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					Mul_type6 << <blocks7, threads >> > (CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, 
						OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);

				}

			}
			if (jorder == 1)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, qvector_1 / 2.0, qvector_2 / 2.0, Testi2, para);
				}

				shift_3D(CTempFive, CTempTwo, para.dataparamsCpu.Nz, 0, 0, qvector_3, para, ImgSizeAfter);
				shift_3D(CTempSix, CTempTwo, para.dataparamsCpu.Nz, 0, 0, -qvector_3, para, ImgSizeAfter);

				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					Add_type1 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, CTempFive + jNz * ImgSizeAfter * ImgSizeAfter, CTempSix + jNz * ImgSizeAfter * ImgSizeAfter);
					CalMask_type2 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, 0);
					Mul_type6 << <blocks7, threads >> > (CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, 
						CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
					CudaMultyComplexWithOnedouble << <blocks7, threads >> > (CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, 
						CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
				}


			}
			if (jorder == 2)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, -qvector_1 / 2.0, -qvector_2 / 2.0, Testi2, para);
				}

				shift_3D(CTempFive, CTempTwo, para.dataparamsCpu.Nz, 0, 0, qvector_3, para, ImgSizeAfter);
				shift_3D(CTempSix, CTempTwo, para.dataparamsCpu.Nz, 0, 0, -qvector_3, para, ImgSizeAfter);
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					Add_type1 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, CTempFive + jNz * ImgSizeAfter * ImgSizeAfter, 
						CTempSix + jNz * ImgSizeAfter * ImgSizeAfter);
					CalMask_type2 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, 0);
					Mul_type6 << <blocks7, threads >> > (CTempOne + jNz * ImgSizeAfter * ImgSizeAfter, 
						CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
					CudaMultyComplexWithOnedouble << <blocks7, threads >> > (CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, 
						CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
				}

			}
			if (jorder == 3)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, qvector_1, qvector_2, Testi2, para);
					CalMask_type2 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, 0);
					Mul_type6 << <blocks7, threads >> > (CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
				}
			}
			if (jorder == 4)
			{
				for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
				{
					shift_2D(CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, OTFtemp1 + jNz * ImgSizeAfter * ImgSizeAfter, -qvector_1, -qvector_2, Testi2, para);
					CalMask_type2 << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter, 0);
					Mul_type6 << <blocks7, threads >> > (CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter, CTempTwo + (jNz)* ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
				}


			}
			
			for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
			{
				Add_type1 << <blocks7, threads >> > (dst1 + (jNz)* ImgSizeAfter * ImgSizeAfter, dst1 + (jNz)* ImgSizeAfter * ImgSizeAfter, CTempOne + (jNz)* ImgSizeAfter * ImgSizeAfter);
			}
			
			//19层的CTempOne中1-3有误差，4-15是正确的，16-19有误差。
		}
		
		//计算notchfilter end





	}



	cufftDestroy(fft_loco);
	cufftDestroy(fft_loco2);
	cudaFree(TempRealOne);
	cudaFree(CTempSix);
	cudaFree(CTempFive);
	cudaFree(CTempOne);
	cudaFree(CTempTwo);
	cudaFree(Matrix_Ones_C);
	cudaFree(corr_abs);
	cudaFree(corr);
	
	cudaFree(unmixing_matrix);
	cudaFree(mixing_matrix);
	cudaFree(unmixing_matrix_15);

	cudaFree(OTFtemp1);
	cudaFree(do_upsample_image_in);


	cublasDestroy(handle);

}
void CalFuntion::FuctionGetFilter(double* Filter1, double* Filter2, double* OTFshiftfinal, double lambdaregul1, double lambdaregul2, double* peak_kx, double* peak_ky, int Npola, int Nazim, sCudaPara& para)
{
	//20240409到这
	//全隐掉，是好的。初步判定cudaMalloc((void**)&cutoff, Npola* Nazim * sizeof(double));出问题

	double NAl = para.dataparamsCpu.NA / para.dataparamsCpu.emwavelength;
	double q0 = para.dataparamsCpu.refmed / para.dataparamsCpu.emwavelength;
	double q0ex = para.dataparamsCpu.refmed / para.dataparamsCpu.exwavelength;
	double NB0 = q0 * q0 - NAl * NAl;
	cufftDoubleComplex NBl;
	if (NB0 < 0)
	{
		NBl.x = 0;
		NBl.y = sqrt(-NB0);
	}
	else
	{
		NBl.x = sqrt(NB0);
		NBl.y = 0;
	}
	dim3 blocks_loco(para.dataparamsCpu.ImageCut, para.dataparamsCpu.Nz, 1);
	dim3 threads_loco(Npola, 1, 1);

	double* cutoff;
	double* TempRealOne;
	double* TempRealTwo;
	double* TempRealThree;

	cudaMalloc((void**)&cutoff, Npola* Nazim * sizeof(double));
	cudaMalloc((void**)&TempRealOne, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(double));
	cudaMalloc((void**)&TempRealTwo, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(double));
	cudaMalloc((void**)&TempRealThree, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(double));
//	cublasHandle_t handle;
//cublasCreate(&handle);
	//CudaCalFilterStepCalCutoff << <blocks_loco, threads_loco >> > (cutoff, NAl, NBl, q0* q0, q0ex, para.dataparamsCpu.rawpixelsize[0] / 2.0,
	//	peak_kx[0], peak_kx[1], peak_kx[2], peak_ky[0], peak_ky[1], peak_ky[2], Npola, Nazim, ImgSizeAfter);



	//for (int i = 0; i < para.dataparamsCpu.Nz; i++)
	//{
	//	//备注：CudaCalFilterStepTwo函数对应Recon3DSIM的get_filter1函数
	//	CudaCalFilterStepTwo << <blocks7, threads >> > (TempRealOne + i * ImgSizeAfter * ImgSizeAfter, cutoff,
	//		para.dataparamsCpu.rawpixelsize[0] / 2.0, para.dataparamsCpu.rawpixelsize[1] / 2.0, para.dataparamsCpu.rawpixelsize[2],
	//		i, para.dataparamsCpu.Nz, Npola, Nazim, ImgSizeAfter);//Temp_1024->Apo = 1-qrad./cutoffmap;
	//		CudaMultyDoubleWithOnedouble << <blocks7, threads >> > (TempRealTwo, TempRealOne + i * ImgSizeAfter * ImgSizeAfter, -1.0, ImgSizeAfter);
	//	int idx;
	//	cublasIdamax(handle, ImgSizeAfter * ImgSizeAfter, TempRealTwo, 1, &idx);
	//	Add_type3 << <blocks7, threads >> > (TempRealThree + i * ImgSizeAfter * ImgSizeAfter,
	//		TempRealOne + i * ImgSizeAfter * ImgSizeAfter,
	//		TempRealTwo, idx - 1);

	//	CudaCalFilterStepThree << <blocks7, threads >> > (Filter2 + i * ImgSizeAfter * ImgSizeAfter, TempRealThree + i * ImgSizeAfter * ImgSizeAfter, OTFshiftfinal + i * ImgSizeAfter * ImgSizeAfter, lambdaregul2, ImgSizeAfter);

	//}









	//for (int i = 0; i < para.dataparamsCpu.Nz; i++)
	//{
	//	CudaCalFilterStepOne << <blocks7, threads >> > (TempRealOne + i * ImgSizeAfter * ImgSizeAfter, TempRealOne + i * ImgSizeAfter * ImgSizeAfter);
	//}
	////Temp_1024 Apo = Apo.^triangleexponent;

	////后面是Cal_Filter1 complexparity
	////complexparity begin
	//int dimshift = 2 * (floor(ImgSizeAfter / 2) + 1) - 1 - ImgSizeAfter;
	//printf("dimshift:%d\n", dimshift);
	//CudaCalFlip << <blocks7, threads >> > (TempRealTwo, TempRealOne, 0, para.dataparamsCpu.Nz, ImgSizeAfter);
	//CudaCircshiftThreeDimension << <blocks7, threads >> > (TempRealThree, TempRealTwo, 0, para.dataparamsCpu.Nz, dimshift, ImgSizeAfter);//x
	//CudaCalFlip << <blocks7, threads >> > (TempRealTwo, TempRealThree, 1, para.dataparamsCpu.Nz, ImgSizeAfter);
	//CudaCircshiftThreeDimension << <blocks7, threads >> > (TempRealThree, TempRealTwo, 1, para.dataparamsCpu.Nz, dimshift, ImgSizeAfter);//y
	//dimshift = 2 * (floor(para.dataparamsCpu.Nz / 2) + 1) - 1 - para.dataparamsCpu.Nz;
	//printf("dimshift:%d\n", dimshift);
	//CudaCalFlip << <blocks7, threads >> > (TempRealTwo, TempRealThree, 2, para.dataparamsCpu.Nz, ImgSizeAfter);
	//CudaCircshiftThreeDimension << <blocks7, threads >> > (TempRealThree, TempRealTwo, 2, para.dataparamsCpu.Nz, dimshift, ImgSizeAfter);//z
	////complexparity end


	//for (int i = 0; i < para.dataparamsCpu.Nz; i++)
	//{
	//	Add_type2 << <blocks7, threads >> > (TempRealTwo + i * ImgSizeAfter * ImgSizeAfter, TempRealThree + i * ImgSizeAfter * ImgSizeAfter, TempRealOne + i * ImgSizeAfter * ImgSizeAfter);
	//	CudaMultyDoubleWithOnedouble << <blocks7, threads >> > (TempRealTwo + i * ImgSizeAfter * ImgSizeAfter, TempRealTwo + i * ImgSizeAfter * ImgSizeAfter, 0.5, ImgSizeAfter);
	//	//TempRealTwo->Apo;

	//	CudaCalFilterStepThree << <blocks7, threads >> > (Filter1 + i * ImgSizeAfter * ImgSizeAfter, TempRealTwo + i * ImgSizeAfter * ImgSizeAfter, OTFshiftfinal + i * ImgSizeAfter * ImgSizeAfter, lambdaregul1, ImgSizeAfter);
	//	//Filter TempRealTwo
	//}


	//cudaFree(cutoff);
	//cudaFree(TempRealOne);
	//cudaFree(TempRealTwo);
	//cudaFree(TempRealThree);
	//cublasDestroy(handle);	
}
void CalFuntion::filter_3D(double* Result, cufftDoubleComplex* sum_fft,   sCudaPara& para)
{
	double * TempRealOne;//建立，释放
	cudaMalloc((void**)&TempRealOne, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(double));


	cufftDoubleComplex * CTempOne;//建立，释放
	cudaMalloc((void**)&CTempOne, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftDoubleComplex * CTempTwo;//建立，释放
	cudaMalloc((void**)&CTempTwo, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftDoubleComplex * CTempThree;//建立，释放
	cudaMalloc((void**)&CTempThree, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cufftDoubleComplex * CTempFour;//建立，释放
	cudaMalloc((void**)&CTempFour, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));
	dim3 blocks_loco_2(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, para.dataparamsCpu.Nz * 4);
	ifftShfit_3D_C << <blocks_loco_2, threads >> > (CTempOne, sum_fft, ImgSizeAfter);//输入与输出一定不同变量

	cufftHandle fft_loco;//建立，释放
	int istride = 1;
	int ostride = 1;
	int n[2];
	n[0] = ImgSizeAfter;
	n[1] = ImgSizeAfter;
	int batch = 1;
	cufftPlanMany(&fft_loco, 2, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);
	cufftHandle fft_loco2;//建立，释放
	batch = ImgSizeAfter;
	istride = 1;
	ostride = 1;
	n[2];
	n[0] = para.dataparamsCpu.Nz;
	n[1] = 1;
	cufftPlanMany(&fft_loco2, 1, n, NULL, istride, 0, NULL, ostride, 0, CUFFT_Z2Z, batch);

	
	for (int i = 0; i < para.dataparamsCpu.Nz; i++)
	{
		cufftExecZ2Z(fft_loco, CTempOne + i * ImgSizeAfter* ImgSizeAfter,
			CTempTwo + i * ImgSizeAfter * ImgSizeAfter, CUFFT_INVERSE);
		ifftR2RRec << <blocks7, threads >> > (CTempTwo + i * ImgSizeAfter * ImgSizeAfter,
			CTempTwo + i * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
	}
	

	fftn3D_0 << <blocks7, threads >> > (CTempOne, CTempTwo, para.dataparamsCpu.Nz, ImgSizeAfter);


	int Nz = para.dataparamsCpu.Nz;
	for (int Testi_3 = 0; Testi_3 < ImgSizeAfter; Testi_3++)
	{
		cufftExecZ2Z(fft_loco2, CTempOne + Testi_3 * Nz * ImgSizeAfter, CTempTwo + Testi_3 * Nz * ImgSizeAfter, CUFFT_INVERSE);
	}
	
	fftn3D_1 << <blocks7, threads >> > (CTempOne, CTempTwo, para.dataparamsCpu.Nz, ImgSizeAfter);
	
	for (int i = 0; i < Nz; i++)
	{
		ifftR2RRec_2 << <blocks7, threads >> > (CTempOne + i * ImgSizeAfter * ImgSizeAfter,
			CTempOne + i * ImgSizeAfter * ImgSizeAfter, Nz);
	}
	


	fftShfit_3D_C_2 << <blocks_loco_2, threads >> > (CTempTwo, CTempOne, ImgSizeAfter);//输入与输出一定不同变量//
	
	for (int i = 0; i < Nz; i++)
	{
		Abs_C_type4 << <blocks7, threads >> > (TempRealOne + i * ImgSizeAfter * ImgSizeAfter, CTempTwo + i * ImgSizeAfter * ImgSizeAfter);
	}
	cublasHandle_t handle;
	cublasCreate(&handle);
	double norm = 0;
	for (int i = 0; i < Nz; i++)
	{
		int idx;
		cublasIdamax(handle, ImgSizeAfter * ImgSizeAfter, TempRealOne, 1, &idx);
		double max_value;
		cudaMemcpy(&max_value, &TempRealOne[idx - 1], sizeof(double), cudaMemcpyDeviceToHost);
		norm = max(norm, max_value);
	}
	for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
	{
		CudaMultyDoubleWithOnedouble << <blocks7, threads >> > (Result + jNz * ImgSizeAfter * ImgSizeAfter,
			TempRealOne + jNz * ImgSizeAfter * ImgSizeAfter, 65535.0 / norm, ImgSizeAfter);
	}










	

	


	/*cufftDoubleComplex* CTempOne;
	cufftDoubleComplex* CTempTwo;
	cudaMalloc((void**)&CTempOne, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&CTempTwo, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(cufftDoubleComplex));
	for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
	{
		CudaMultyComplexWithDouble << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter,
			sum_fft + jNz * ImgSizeAfter * ImgSizeAfter,
			Filter1 + jNz * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);

		CudaMultyComplexWithDouble << <blocks7, threads >> > (CTempOne + jNz * ImgSizeAfter * ImgSizeAfter,
			CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter,
			Filter2 + jNz * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
	}
	
	/*double* TempRealOne;
	cudaMalloc((void**)&TempRealOne, (para.dataparamsCpu.Nz* ImgSizeAfter* ImgSizeAfter) * sizeof(double));


	


	cufftDoubleComplex* CTempOne;
	cufftDoubleComplex* CTempTwo;




	for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
	{
		CudaMultyComplexWithDouble << <blocks7, threads >> > (CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter,
			sum_fft + jNz * ImgSizeAfter * ImgSizeAfter,
			TempRealOne + jNz * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);

		CudaMultyComplexWithDouble << <blocks7, threads >> > (CTempOne + jNz * ImgSizeAfter * ImgSizeAfter,
			CTempTwo + jNz * ImgSizeAfter * ImgSizeAfter,
			Filter2 + jNz * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);
	}*/

	

	//cudaFree(TempRealOne);
	//cudaFree(CTempOne);

	//cudaFree(CTempTwo);



	//cudaFree(Filter2);
	//cufftDestroy(fft_loco);
}

void CalFuntion::Recon3DSIM(cv::Mat* SimResult, sCudaPara& para, int dataH, int dataW)
{
	blocks = dim3(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, 1);
	threads = dim3(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, 1);
	blocks2 = dim3(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, 15);
	threads2 = dim3(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, 1);
	blocks3 = dim3(1, 1, 1);
	threads3 = dim3(1, 1, 1);

	blocks6 = dim3(1, 1, 100);
	blocks8 = dim3(5, 5, 1);
	blocks9 = dim3(15, 15, 1);
	blocks7 = dim3(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, 4);

	blocks10 = dim3(para.dataparamsCpu.ImageCutBlock, para.dataparamsCpu.ImageCutThread, 40);

	cufftDoubleComplex* OTFem;//建立，释放
	cudaMalloc((void**)&OTFem, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(cufftDoubleComplex));

	if (para.dataparamsCpu.Nz != 1)
	{
		cufftDoubleComplex* wavevector;//建立，释放
		cudaMalloc((void**)&wavevector, 10 * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
		double* PFS;//建立，释放
		cudaMalloc((void**)&PFS, para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter * sizeof(double));



		get_pupil_matrix(wavevector, para, ImgSizeAfter);////输出：para.dataparamsGpu.TempRealOne对应matlab wavevector，内部：无
		get_field_matrix(PFS, wavevector, para);
		get_psf(PFS, PFS, para.dataparamsCpu.Nz, para, ImgSizeAfter);
		cudaFree(wavevector);
		//get_modelOTF_G(para.dataparamsGpu.OTFem_C, para, wavevector, para.dataparamsCpu.Nz, ImgSizeAfter);
		cufftDoubleComplex* OTFinc2d_throughfocus;//建立，释放
		cudaMalloc((void**)&OTFinc2d_throughfocus, para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));
		get_throughfocusotf(OTFinc2d_throughfocus, PFS, para);





		cudaFree(PFS);

		get_otf3d(OTFinc2d_throughfocus, OTFinc2d_throughfocus, para, para.dataparamsCpu.Nz, ImgSizeAfter);



		do_OTFmasking3D(OTFem, OTFinc2d_throughfocus, para, para.dataparamsCpu.Nz, ImgSizeAfter);
		hengxiang = 256;
		zongxiang = 256;
		ShowComplex(OTFinc2d_throughfocus, 512, 512, zongxiang, zongxiang + 5, hengxiang + 0, hengxiang + 5, 0, 19);
		cudaFree(OTFinc2d_throughfocus);



	}
	else//dataparams_Nz=1暂时为谢，可参照20240311版本
	{
	}

	double* allimages_in;//建立，释放
	cudaMalloc((void**)&allimages_in, 15 * para.dataparamsCpu.Nz * ImgSizeBefore * ImgSizeBefore * sizeof(double));



	for (int i = 0; i < para.dataparamsCpu.Nz * 15; i++)
	{
		para.dataparamsCpu.Snoisy[i].convertTo(para.dataparamsCpu.Snoisy[i], CV_64FC1);
	}
	//edgetaper(para.dataparamsCpu.Snoisy);


	for (int i = 0; i < para.dataparamsCpu.Nz * 15; i++)
	{
		cudaMemcpy(&allimages_in[i * dataH * dataW], para.dataparamsCpu.Snoisy[i].data, dataH * dataW * sizeof(double), cudaMemcpyHostToDevice);
	}



	dim3 blocks_loco(1, 1, para.dataparamsCpu.Nz);
	double* MCNR;//建立，释放
	cudaMalloc((void**)&MCNR, (para.dataparamsCpu.Nz) * sizeof(double));
	averageMCNR_foreground_top << <blocks_loco, threads >> > (MCNR, para.dataparamsCpu.Nz);

	double* Snoisy;//建立，释放
	cudaMalloc((void**)&Snoisy, 15 * ImgSizeBefore * ImgSizeBefore * sizeof(double));
	weightCNR << <blocks2, threads >> > (Snoisy, allimages_in, para.dataparamsCpu.Nz, MCNR, ImgSizeBefore);//Snoisy_Temp->Temp_1024
	cudaFree(MCNR);

	double peak_kx_loco[3];//目前分析是3不是Nz
	double peak_ky_loco[3];

	cufftDoubleComplex* find_illumination_pattern_p1;//建立
	cudaMalloc((void**)&find_illumination_pattern_p1, 6 * 1 * 1 * sizeof(cufftDoubleComplex));


	double* Module;//建立，释放
	cudaMalloc((void**)&Module, 1 * 1 * 5 * sizeof(double));
	find_illumination_pattern(peak_kx_loco, peak_ky_loco, Snoisy, para.dataparamsCpu.rawpixelsize[0], find_illumination_pattern_p1, Module, para);
	cudaFree(Snoisy);


	{
		for (int Test10i = 0; Test10i < 3; Test10i++)//必须是3
		{
			peak_kx[Test10i] = peak_kx_loco[Test10i];
			peak_ky[Test10i] = peak_ky_loco[Test10i];
		}
	}
	std::printf("params.Dir(jangle).px: %f %f %f\n", peak_kx[0] / 2, peak_kx[1] / 2, peak_kx[2] / 2);
	std::printf("params.Dir(jangle).py: %f %f %f\n", peak_ky[0] / 2, peak_ky[1] / 2, peak_ky[2] / 2);
	cufftDoubleComplex* sum_fft;
	double* OTFshiftfinal;
	cudaMalloc((void**)&OTFshiftfinal, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(double));
	cudaMalloc((void**)&sum_fft, para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter * sizeof(cufftDoubleComplex));//替代allftorderims
	for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
	{
		Cal_Zeros_R << <blocks7, threads >> > (OTFshiftfinal + (jNz)*ImgSizeAfter * ImgSizeAfter);
		Cal_Zeros_C << <blocks7, threads >> > (sum_fft + (jNz)*ImgSizeAfter * ImgSizeAfter);
	}
	process_data(sum_fft, OTFshiftfinal, peak_kx, peak_ky, allimages_in,OTFem, find_illumination_pattern_p1, Module, para);
	
	//19层的sum_fft中1-3有误差，4-15是正确的，16-19有误差。
	cudaFree(OTFem);
	cudaFree(Module);

	//double* Filter1;
	//cudaMalloc((void**)&Filter1, para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter * sizeof(double));
	//double* Filter2;
	//cudaMalloc((void**)&Filter2, para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter * sizeof(double));
	//FuctionGetFilter(Filter1, Filter2, OTFshiftfinal,
	//	para.dataparamsCpu.lambdaregul1, para.dataparamsCpu.lambdaregul2, peak_kx, peak_ky, Npola, Nazim, para);
	//cudaFree(OTFshiftfinal);
	
	//for (int jNz = 0; jNz < para.dataparamsCpu.Nz; jNz++)
	//{
	//	CudaMultyComplexWithDouble << <blocks7, threads >> > (sum_fft + jNz * ImgSizeAfter * ImgSizeAfter,
	//		sum_fft + jNz * ImgSizeAfter * ImgSizeAfter,
	//		Filter1 + jNz * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);

	//	CudaMultyComplexWithDouble << <blocks7, threads >> > (sum_fft + jNz * ImgSizeAfter * ImgSizeAfter,
	//		sum_fft + jNz * ImgSizeAfter * ImgSizeAfter,
	//		Filter2 + jNz * ImgSizeAfter * ImgSizeAfter, ImgSizeAfter);

	//}


	double* Result;

	cudaMalloc((void**)&Result, (para.dataparamsCpu.Nz * ImgSizeAfter * ImgSizeAfter) * sizeof(double));



	filter_3D(Result,sum_fft,  para);

	cudaFree(sum_fft);



	std::vector<cv::Mat> images;
	for (int ti = 0; ti < para.dataparamsCpu.Nz; ti++)
	{
		cv::Mat Retest;
		int dataH = ImgSizeAfter;
		int dataW = ImgSizeAfter;
		double* h_Result;
		h_Result = (double*)malloc(dataH * dataW * sizeof(double));
		//拷贝至内存
		cudaMemcpy(h_Result, &Result[ti * dataH * dataW], dataH * dataW * sizeof(double), cudaMemcpyDeviceToHost);
		//赋值给cv::Mat并打印
		cv::Mat_<double> resultReal = cv::Mat_<double>(dataH, dataW);
		cv::Mat_<double> resultImag = cv::Mat_<double>(dataH, dataW);
		for (int i = 0; i < dataH; i++) {
			double* rowPtrReal = resultReal.ptr<double>(i);
			for (int j = 0; j < dataW; j++) {
				rowPtrReal[j] = h_Result[i * dataW + j];
			}
		}
		resultReal.convertTo(SimResult[ti], CV_64FC1);
		free(h_Result);
	}

	cudaFree(Result);

}
