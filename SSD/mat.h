#ifndef __MAT__
#define __MAT__

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <time.h>

#define full 0
#define same 1
#define valid 2

typedef struct mat2DSize {
	int c;//列
	int r;//行
}nSize;

float** rotate180(float** mat, nSize matSize);//矩阵翻转180度

float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);//互相关

float** conv(float** map, nSize mapSize, float** inputData, nSize inSize, int type);//矩阵卷积操作

//给二维矩阵的边缘扩大，增加addw大小的0值边
float** matEdgeExpand(float** map, nSize matSize, int addc, int addr);

//给二维矩阵边缘缩小，擦除shrinkc大小的边
float** matEdgeShrink(float** map, nSize matSize, int shrinkc, int shrinkr);


void addMat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);//矩阵相加

#endif


