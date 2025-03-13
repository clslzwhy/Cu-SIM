#pragma once
#include "afxdialogex.h"


// CTest2Dlg 对话框

class CTest2Dlg : public CDialogEx
{
	DECLARE_DYNAMIC(CTest2Dlg)

public:
	CTest2Dlg(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CTest2Dlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TEST2_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
};
