// CTest2Dlg.cpp: 实现文件
//

#include "pch.h"
#include "Test2.h"
#include "afxdialogex.h"
#include "CTest2Dlg.h"


// CTest2Dlg 对话框

IMPLEMENT_DYNAMIC(CTest2Dlg, CDialogEx)

CTest2Dlg::CTest2Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_TEST2_DIALOG, pParent)
{

}

CTest2Dlg::~CTest2Dlg()
{
}

void CTest2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CTest2Dlg, CDialogEx)
END_MESSAGE_MAP()


// CTest2Dlg 消息处理程序
