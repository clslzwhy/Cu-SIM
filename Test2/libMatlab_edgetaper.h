/*
 * MATLAB Compiler: 8.2 (R2021a)
 * Date: Thu Apr 11 12:47:59 2024
 * Arguments:
 * "-B""macro_default""-W""lib:libMatlab_edgetaper""-T""link:lib""Mat_edgetaper.
 * m"
 */

#ifndef libMatlab_edgetaper_h
#define libMatlab_edgetaper_h 1

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
#ifndef LIB_libMatlab_edgetaper_C_API 
#define LIB_libMatlab_edgetaper_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_libMatlab_edgetaper_C_API 
bool MW_CALL_CONV libMatlab_edgetaperInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libMatlab_edgetaper_C_API 
bool MW_CALL_CONV libMatlab_edgetaperInitialize(void);

extern LIB_libMatlab_edgetaper_C_API 
void MW_CALL_CONV libMatlab_edgetaperTerminate(void);

extern LIB_libMatlab_edgetaper_C_API 
void MW_CALL_CONV libMatlab_edgetaperPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libMatlab_edgetaper_C_API 
bool MW_CALL_CONV mlxMat_edgetaper(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libMatlab_edgetaper_C_API bool MW_CALL_CONV mlfMat_edgetaper(int nargout, mxArray** dst, mxArray* src);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif
