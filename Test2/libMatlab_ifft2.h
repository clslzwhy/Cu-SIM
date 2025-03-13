/*
 * MATLAB Compiler: 8.2 (R2021a)
 * Date: Fri Mar 15 11:13:55 2024
 * Arguments:
 * "-B""macro_default""-W""lib:libMatlab_ifft2""-T""link:lib""Matlab_ifft2.m"
 */

#ifndef libMatlab_ifft2_h
#define libMatlab_ifft2_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libMatlab_ifft2_C_API 
#define LIB_libMatlab_ifft2_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_libMatlab_ifft2_C_API 
bool MW_CALL_CONV libMatlab_ifft2InitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libMatlab_ifft2_C_API 
bool MW_CALL_CONV libMatlab_ifft2Initialize(void);

extern LIB_libMatlab_ifft2_C_API 
void MW_CALL_CONV libMatlab_ifft2Terminate(void);

extern LIB_libMatlab_ifft2_C_API 
void MW_CALL_CONV libMatlab_ifft2PrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libMatlab_ifft2_C_API 
bool MW_CALL_CONV mlxMatlab_ifft2(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libMatlab_ifft2_C_API bool MW_CALL_CONV mlfMatlab_ifft2(int nargout, mxArray** O_real, mxArray** O_imag, mxArray* I_real, mxArray* I_imag);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif
