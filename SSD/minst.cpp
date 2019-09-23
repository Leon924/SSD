#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"


ImgArr read_Img(int number_of_images)//读入图像
{

	int n_rows = 28;
	int n_cols = 28;

	//获取第一幅图像，保存到vec中
	int i, r, c;
	//图像数组的初始化
	ImgArr imgarr = (ImgArr)malloc(sizeof(MinstImgArr));
	imgarr->ImgNum = number_of_images;
	imgarr->ImgPtr = (MinstImg*)malloc(number_of_images * sizeof(MinstImg));

	for (i = 0; i < number_of_images; i++) {
		imgarr->ImgPtr[i].r = n_rows;
		imgarr->ImgPtr[i].c = n_cols;
		imgarr->ImgPtr[i].ImgData = (float**)malloc(n_rows * sizeof(float*));
		for (r = 0; r < n_rows; r++) {
			imgarr->ImgPtr[i].ImgData[r] = (float*)malloc(n_cols * sizeof(float));
			for (c = 0; c < n_cols; c++) {
				imgarr->ImgPtr[i].ImgData[r][c] = (float)(rand() % 2);
			}
		}
	}
	return imgarr;
}


LabelArr read_Label(int number_of_labels) {

	int label_long = 10;
	int i, l;

	//图像标记数组的初始化
	LabelArr labelarr = (LabelArr)malloc(sizeof(MinstLabelArr));
	labelarr->LabelNum = number_of_labels;
	labelarr->LabelPtr = (MinstLabel*)malloc(number_of_labels * sizeof(MinstLabel));


	for (i = 0; i < number_of_labels; i++) {
		labelarr->LabelPtr[i].l = 10;
		labelarr->LabelPtr[i].LabelData = (float*)calloc(label_long, sizeof(float));
		labelarr->LabelPtr[i].LabelData[i] = 1.0;
	}

	return labelarr;
}










