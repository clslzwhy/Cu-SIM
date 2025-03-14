/*
 * MATLAB Compiler: 8.2 (R2021a)
 * Date: Thu May  9 14:43:15 2024
 * Arguments:
 * "-B""macro_default""-W""lib:libMatlab_deconvlucy""-T""link:lib""Matlab_deconv
 * lucy.m"
 */

#define EXPORTING_libMatlab_deconvlucy 1
#include "libMatlab_deconvlucy.h"

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
#ifndef LIB_libMatlab_deconvlucy_C_API
#define LIB_libMatlab_deconvlucy_C_API /* No special import/export declaration */
#endif

LIB_libMatlab_deconvlucy_C_API 
bool MW_CALL_CONV libMatlab_deconvlucyInitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
    if (_mcr_inst)
        return true;
    if (!mclmcrInitialize())
        return false;
    if (!GetModuleFileName(GetModuleHandle("libMatlab_deconvlucy"), path_to_dll, _MAX_PATH))
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

LIB_libMatlab_deconvlucy_C_API 
bool MW_CALL_CONV libMatlab_deconvlucyInitialize(void)
{
    return libMatlab_deconvlucyInitializeWithHandlers(mclDefaultErrorHandler, 
                                                    mclDefaultPrintHandler);
}

LIB_libMatlab_deconvlucy_C_API 
void MW_CALL_CONV libMatlab_deconvlucyTerminate(void)
{
    if (_mcr_inst)
        mclTerminateInstance(&_mcr_inst);
}

LIB_libMatlab_deconvlucy_C_API 
void MW_CALL_CONV libMatlab_deconvlucyPrintStackTrace(void) 
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


LIB_libMatlab_deconvlucy_C_API 
bool MW_CALL_CONV mlxMatlab_deconvlucy(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                       *prhs[])
{
    return mclFeval(_mcr_inst, "Matlab_deconvlucy", nlhs, plhs, nrhs, prhs);
}

LIB_libMatlab_deconvlucy_C_API 
bool MW_CALL_CONV mlfMatlab_deconvlucy(int nargout, mxArray** Out1, mxArray** Out2, 
                                       mxArray** Out3, mxArray** Out4, mxArray** Out5, 
                                       mxArray** Out6, mxArray** Out7, mxArray** Out8, 
                                       mxArray** Out9, mxArray** Out10, mxArray** Out11, 
                                       mxArray** Out12, mxArray** Out13, mxArray** Out14, 
                                       mxArray** Out15, mxArray* datasets)
{
    return mclMlfFeval(_mcr_inst, "Matlab_deconvlucy", nargout, 15, 1, Out1, Out2, Out3, Out4, Out5, Out6, Out7, Out8, Out9, Out10, Out11, Out12, Out13, Out14, Out15, datasets);
}

