#ifndef __MINST__ 
#define __MINST__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
/*
	MINST是一个手写图像数据库
*/

typedef struct MinstImg {
	int c;	//	图像宽
	int r;	//	图像高
	float** ImgData; //图像数据二维动态数组
}MinstImg;

typedef struct MinstImgArr {
	int ImgNum; //存储图像的数目
	MinstImg* ImgPtr; //存储图像数组的指针
}*ImgArr;

typedef struct MnistLabel {
	int l;	//输出标记的长
	float* LabelData; //输出标记数据
}MinstLabel;

typedef struct MinstLabelArr {
	int LabelNum; // 标记的数目
	MinstLabel* LabelPtr; //指向标记的指针
}*LabelArr;

LabelArr read_Label(int); //输入图像标记

ImgArr read_Img(int); //读入图像

#endif


