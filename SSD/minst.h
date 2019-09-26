#ifndef __MINST__ 
#define __MINST__
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


typedef struct MinstImg{
	int c;	
	int r;
	float** ImgData; //array of one image
}MinstImg;

typedef struct MinstImgArr{
	int ImgNum; 
	MinstImg* ImgPtr;
}*ImgArr;

typedef struct MnistLabel{
	int l;	//length of label
	float* LabelData; //vector of label
} MinstLabel;

typedef struct MinstLabelArr{
	int LabelNum; 
	MinstLabel* LabelPtr; 
}*LabelArr;

LabelArr read_Label(int); 

ImgArr read_Img(int);

#endif


