/*
 * MATLAB Compiler: 8.2 (R2021a)
 * Date: Thu Nov 16 09:11:15 2023
 * Arguments: "-B""macro_default""-W""lib:libPsfCal""-T""link:lib""libPsfCal.m"
 */

#ifndef libPsfCal_h
#define libPsfCal_h 1

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
#ifndef LIB_libPsfCal_C_API 
#define LIB_libPsfCal_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_libPsfCal_C_API 
bool MW_CALL_CONV libPsfCalInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libPsfCal_C_API 
bool MW_CALL_CONV libPsfCalInitialize(void);

extern LIB_libPsfCal_C_API 
void MW_CALL_CONV libPsfCalTerminate(void);

extern LIB_libPsfCal_C_API 
void MW_CALL_CONV libPsfCalPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libPsfCal_C_API 
bool MW_CALL_CONV mlxLibPsfCal(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libPsfCal_C_API bool MW_CALL_CONV mlfLibPsfCal(int nargout, mxArray** R1, mxArray** R2, mxArray** R3, mxArray** R4, mxArray** R5, mxArray** R6, mxArray** R7, mxArray** R8, mxArray** R9, mxArray* lambda, mxArray* sizeLoc);

#ifdef __cplusplus
}
#endif
/* C INTERFACE -- MLF WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#endif
