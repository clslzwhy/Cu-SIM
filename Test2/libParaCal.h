/*
 * MATLAB Compiler: 8.2 (R2021a)
 * Date: Fri Nov 17 13:37:20 2023
 * Arguments: "-B""macro_default""-W""lib:libParaCal""-T""link:lib""ParaCal.m"
 */

#ifndef libParaCal_h
#define libParaCal_h 1

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
#ifndef LIB_libParaCal_C_API 
#define LIB_libParaCal_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_libParaCal_C_API 
bool MW_CALL_CONV libParaCalInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libParaCal_C_API 
bool MW_CALL_CONV libParaCalInitialize(void);

extern LIB_libParaCal_C_API 
void MW_CALL_CONV libParaCalTerminate(void);

extern LIB_libParaCal_C_API 
void MW_CALL_CONV libParaCalPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libParaCal_C_API 
bool MW_CALL_CONV mlxParaCal(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libParaCal_C_API bool MW_CALL_CONV mlfParaCal(int nargout, mxArray** Result, mxArray* raw_data1, mxArray* raw_data2, mxArray* raw_data3, mxArray* raw_data4, mxArray* raw_data5, mxArray* raw_data6, mxArray* raw_data7, mxArray* raw_data8, mxArray* raw_data9, mxArray* lambda, mxArray* NA, mxArray* pixelsize, mxArray* nrbands, mxArray* nrDirs, mxArray* nrPhases, mxArray* RL, mxArray* useABCthreshold);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif
