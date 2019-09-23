#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"

float** rotate180(float** mat, nSize matSize) { //矩阵翻转 180度
	int i, c, r;
	int outSizeW = matSize.c;
	int outSizeH = matSize.r;
	//翻转之后的矩阵outputData，开辟内存空间
	float** outputData = (float**)malloc(outSizeH * sizeof(float*));
	for (i = 0; i < outSizeH; i++) {
		outputData[i] = (float*)malloc(outSizeW * sizeof(float));
	}
	for (r = 0; r < outSizeH; r++) {
		for (c = 0; c < outSizeW; c++)
			outputData[r][c] = mat[outSizeH - r - 1][outSizeW - c - 1];//有异常
	}

	return outputData;
}

//关于卷积核相关操作的输出选项
//这里共有三种选择：full， same， valid，分别表示
//full指完全操作后，结果的大小为inSize+（mapSize-1）
//same指同输入大小
//valid指完全操作后的大小，一般为inSize-(mapSize-1)大小，其不需要将输入添0扩大

float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type) {
	//互相关表示图像矩阵和卷积核的按位点乘
	//先将输入数据打补丁得到扩展的exInputData
	//这里的卷积要分成两种，偶数和奇数

	int i, j, c, r;

	//卷积核的一半大小
	int halfmapsizew;
	int halfmapsizeh;

	//如果mapSize为偶数，直接整除2
	if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0) {
		halfmapsizew = (mapSize.c) / 2;
		halfmapsizeh = (mapSize.r) / 2;
	}
	else {
		//若为奇数，减1再整除2
		halfmapsizew = (mapSize.c - 1) / 2;
		halfmapsizeh = (mapSize.r - 1) / 2;
	}

	//这里先默认进行full模式的操作，full模式的意思是，从filter和image刚相交开始做卷积
	//full模式的输出大小为inSize + (mapSize -1)
	int outSizeW = inSize.c + (mapSize.c - 1);
	int outSizeH = inSize.r + (mapSize.r - 1);
	float** outputData = (float**)malloc(outSizeH * sizeof(float*));
	for (i = 0; i < outSizeH; i++) {
		outputData[i] = (float*)calloc(outSizeW, sizeof(float));
	}

	float** exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);

	//这里相当于把full模式的输出算出来了
	for (j = 0; j < outSizeH; j++)
		for (i = 0; i < outSizeW; i++)
			for (r = 0; r < mapSize.r; r++)
				for (c = 0; c < mapSize.c; c++) {
					outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];
				}
	//释放有补丁的输入数据
	for (i = 0; i < inSize.r + 2 * (mapSize.r - 1); i++) {
		free(exInputData[i]);
	}
	free(exInputData);

	nSize outSize = { outSizeW, outSizeH };
	switch (type) {
	case full:   //从filter和image刚相交开始做卷积
		return outputData;
	case same: {	//当filter的中心(K)与image的边角重合时，开始做卷积运算，res表示residual
		float** sameres = matEdgeShrink(outputData, outSize, halfmapsizew, halfmapsizeh);
		for (i = 0; i < outSize.r; i++) {
			free(outputData[i]);
		}
		free(outputData);
		return sameres;
	}
	case valid: {	//当filter全部在image里面的时候，进行卷积运算
		float** validres;
		if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)
			validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
		else
			validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2, halfmapsizeh * 2);
		for (i = 0; i < outSize.r; i++)
			free(outputData[i]);
		free(outputData);
		return validres;
	}
	default:
		return outputData;
	}
}

float** conv(float** map, nSize mapSize, float** inputData, nSize inSize, int type) {
	//卷积操作可以用旋转180度的卷积核来求
	float** flipmap = rotate180(map, mapSize);
	float** res = correlation(flipmap, mapSize, inputData, inSize, type);
	int i;
	for (i = 0; i < mapSize.r; i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

//float** upSample(float** mat, nSize matSize, int upc, int upr) {
//	int i, j, m, n;
//	int c = matSize.c;
//	int r = matSize.r;
//	float** res = (float**)malloc((r * upr) * sizeof(float*));
//	for (i = 0; i < (r * upr); i++)
//		res[i] = (float*)malloc((c * upc) * sizeof(float));
//	for (j = 0; j < (r * upr); j = j + upr) {
//		for (i = 0; i < (c * upc); i = i + upc) // 宽的扩充
//			for (m = 0; m < upc; m++)
//				res[j][i + m] = mat[j / upr][i / upc];
//		for (n = 1; n < upr; n++) //高的扩充
//			for (i = 0; i < c * upc; i++)
//				res[j + n][i] = res[j][i];
//	}
//	return res;
//}

//给二微矩阵的边缘扩大，增加addw大小的0值边
float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr) {
	int i, j;
	int c = matSize.c;
	int r = matSize.r;
	float** res = (float**)malloc((r + 2 * addr) * sizeof(float*));
	for (i = 0; i < (r + 2 * addr); i++)
		res[i] = (float*)malloc((c + 2 * addc) * sizeof(float));
	for (j = 0; j < r + 2 * addr; j++)
		for (i = 0; i < c + 2 * addc; i++) {
			if (j < addr || i < addc || j >= (r + addr) || i >= (c + addc))
				res[j][i] = (float)0.0;
			else
				res[j][i] = mat[j - addr][i - addc];
		}
	return res;
}

//给二维矩阵边缘缩小，擦出shrinkc大小的边
float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr) {
	//向量的缩小，宽缩小addw，高缩小addh
	int i, j;
	int c = matSize.c;
	int r = matSize.r;
	float** res = (float**)malloc((r - 2 * shrinkr) * sizeof(float*));
	for (i = 0; i < (r - 2 * shrinkr); i++)
		res[i] = (float*)malloc((c - 2 * shrinkc) * sizeof(float));

	for (j = 0; j < r; j++) {
		for (i = 0; i < c; i++) {
			if (j >= shrinkr && i >= shrinkc && j < (r - shrinkr) && i < (c - shrinkc)) {
				res[j - shrinkr][i - shrinkc] = mat[j][i];
			}
		}
	}
	return res;
}


void addMat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)//矩阵相加
{
	int i, j;
	if (matSize1.c != matSize2.c || matSize1.r != matSize2.r)
		printf("ERROR: Size is not same!");

	for (i = 0; i < matSize1.r; i++) {
		for (j = 0; j < matSize1.c; j++) {
			res[i][j] = mat1[i][j] * mat2[i][j];
		}
	}
}
