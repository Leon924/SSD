#ifndef __MAT__
#define __MAT__

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define full 0
#define same 1
#define valid 2

typedef struct mat2DSize {
	int c;
	int r;
}nSize;

float** rotate180(float** mat, nSize matSize);//rotate 180 degree

float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);

float** conv(float** map, nSize mapSize, float** inputData, nSize inSize, int type);

//expand the edge of matrice，add length of "addw" cols and rows, feed them with 0;
float** matEdgeExpand(float** map, nSize matSize, int addc, int addr);

///expand the edge of matrice，substract length of "addw" cols and rows
float** matEdgeShrink(float** map, nSize matSize, int shrinkc, int shrinkr);

void addMat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);s

#endif


