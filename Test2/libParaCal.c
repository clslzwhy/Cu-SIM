/*
 * MATLAB Compiler: 8.2 (R2021a)
 * Date: Fri Nov 17 13:37:20 2023
 * Arguments: "-B""macro_default""-W""lib:libParaCal""-T""link:lib""ParaCal.m"
 */

#define EXPORTING_libParaCal 1
#include "libParaCal.h"

static HMCRINSTANCE _mcr_inst = NULL; /* don't use nullptr; this may be either C or C++ */

#if defined( _MSC_VER) || defined(__LCC__) || defined(__MINGW64__)
#ifdef __LCC__
#undef EXTERN_C
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <windows.h>
#undef interface

static char path_to_dll[_MAX_PATH];

BOOL WINAPI DllMain(HINSTANCE hInstance, DWORD dwReason, void *pv)
{
    if (dwReason == DLL_PROCESS_ATTACH)
    {
        if (GetModuleFileName(hInstance, path_to_dll, _MAX_PATH) == 0)
            return FALSE;
    }
    else if (dwReason == DLL_PROCESS_DETACH)
    {
    }
    return TRUE;
}
#endif
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

static int mclDefaultPrintHandler(const char *s)
{
    return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern C block */
#endif

#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

static int mclDefaultErrorHandler(const char *s)
{
    int written = 0;
    size_t len = 0;
    len = strlen(s);
    written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
    if (len > 0 && s[ len-1 ] != '\n')
        written += mclWrite(2 /* stderr */, "\n", sizeof(char));
    return written;
}

#ifdef __cplusplus
} /* End extern C block */
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libParaCal_C_API
#define LIB_libParaCal_C_API /* No special import/export declaration */
#endif

LIB_libParaCal_C_API 
bool MW_CALL_CONV libParaCalInitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
    if (_mcr_inst)
        return true;
    if (!mclmcrInitialize())
        return false;
    if (!GetModuleFileName(GetModuleHandle("libParaCal"), path_to_dll, _MAX_PATH))
        return false;
    {
        mclCtfStream ctfStream = 
            mclGetEmbeddedCtfStream(path_to_dll);
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(&_mcr_inst,
                                                             error_handler, 
                                                             print_handler,
                                                             ctfStream);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
    return true;
}

LIB_libParaCal_C_API 
bool MW_CALL_CONV libParaCalInitialize(void)
{
    return libParaCalInitializeWithHandlers(mclDefaultErrorHandler, 
                                          mclDefaultPrintHandler);
}

LIB_libParaCal_C_API 
void MW_CALL_CONV libParaCalTerminate(void)
{
    if (_mcr_inst)
        mclTerminateInstance(&_mcr_inst);
}

LIB_libParaCal_C_API 
void MW_CALL_CONV libParaCalPrintStackTrace(void) 
{
    char** stackTrace;
    int stackDepth = mclGetStackTrace(&stackTrace);
    int i;
    for(i=0; i<stackDepth; i++)
    {
        mclWrite(2 /* stderr */, stackTrace[i], sizeof(char)*strlen(stackTrace[i]));
        mclWrite(2 /* stderr */, "\n", sizeof(char)*strlen("\n"));
    }
    mclFreeStackTrace(&stackTrace, stackDepth);
}


LIB_libParaCal_C_API 
bool MW_CALL_CONV mlxParaCal(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    return mclFeval(_mcr_inst, "ParaCal", nlhs, plhs, nrhs, prhs);
}

LIB_libParaCal_C_API 
bool MW_CALL_CONV mlfParaCal(int nargout, mxArray** Result, mxArray* raw_data1, mxArray* 
                             raw_data2, mxArray* raw_data3, mxArray* raw_data4, mxArray* 
                             raw_data5, mxArray* raw_data6, mxArray* raw_data7, mxArray* 
                             raw_data8, mxArray* raw_data9, mxArray* lambda, mxArray* NA, 
                             mxArray* pixelsize, mxArray* nrbands, mxArray* nrDirs, 
                             mxArray* nrPhases, mxArray* RL, mxArray* useABCthreshold)
{
    return mclMlfFeval(_mcr_inst, "ParaCal", nargout, 1, 17, Result, raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6, raw_data7, raw_data8, raw_data9, lambda, NA, pixelsize, nrbands, nrDirs, nrPhases, RL, useABCthreshold);
}

