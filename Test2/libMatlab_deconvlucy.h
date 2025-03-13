/*
 * MATLAB Compiler: 8.2 (R2021a)
 * Date: Thu May  9 14:43:15 2024
 * Arguments:
 * "-B""macro_default""-W""lib:libMatlab_deconvlucy""-T""link:lib""Matlab_deconv
 * lucy.m"
 */

#ifndef libMatlab_deconvlucy_h
#define libMatlab_deconvlucy_h 1

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
#ifndef LIB_libMatlab_deconvlucy_C_API 
#define LIB_libMatlab_deconvlucy_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_libMatlab_deconvlucy_C_API 
bool MW_CALL_CONV libMatlab_deconvlucyInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libMatlab_deconvlucy_C_API 
bool MW_CALL_CONV libMatlab_deconvlucyInitialize(void);

extern LIB_libMatlab_deconvlucy_C_API 
void MW_CALL_CONV libMatlab_deconvlucyTerminate(void);

extern LIB_libMatlab_deconvlucy_C_API 
void MW_CALL_CONV libMatlab_deconvlucyPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libMatlab_deconvlucy_C_API 
bool MW_CALL_CONV mlxMatlab_deconvlucy(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                       *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libMatlab_deconvlucy_C_API bool MW_CALL_CONV mlfMatlab_deconvlucy(int nargout, mxArray** Out1, mxArray** Out2, mxArray** Out3, mxArray** Out4, mxArray** Out5, mxArray** Out6, mxArray** Out7, mxArray** Out8, mxArray** Out9, mxArray** Out10, mxArray** Out11, mxArray** Out12, mxArray** Out13, mxArray** Out14, mxArray** Out15, mxArray* datasets);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif
