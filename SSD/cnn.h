#ifndef __CNN__
#define __CNN__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mat.h"
#include "minst.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

#define True 1
#define False 0

//convLayer
typedef struct convolutional_layer {
	int inputWidth; 
	int inputHeight; 
	int mapSize; 

	int inChannels; 
	int outChannels; 

	float**** mapData;
	float* biasData; 

	int isFullConnect;
	int connectmode; 

	float*** v;//the result of w*x +b
	float*** y;//y = a(v)

}CovLayer;

//sub-sampling Layer
typedef struct pooling_layer {
	int inputWidth; 
	int inputHeight; 
	int mapSize; 

	int inchannels; 
	int outchannels; 

	int pooltype; //the way of sampling
	float* biasData; 

	float*** y;
}PoolLayer;

//OutputLayer  full-connect
typedef struct nn_layer {
	int inputNum; 
	int outputNum; 

	float** wData; 
	float* biasData; 

	float* v;  
	float* y;

	int isFullConnect; 
}OutLayer;

//net
typedef struct cnn_network {
	int layerNum;
	CovLayer* C1;
	PoolLayer* P2;
	CovLayer* C3;
	PoolLayer* P4;
	OutLayer* O5;
}CNN;

void cnnSetup(CNN* cnn, nSize inputsize, int outputSize);

float cnnTest(CNN* cnn, ImgArr inputData, LabelArr outputdata, int testNum);

void importCnn(CNN* cnn, const char* filename);

CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
//void CovLayerConnect(CovLayer* covL, bool* connecctmode);

PoolLayer* initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType);
//void PoolLayerConnect(PoolLayer* poolL, bool* connectmode);

OutLayer* initOutLayer(int inputNum, int outputNum);

void cnnFf(CNN* cnn, float** inputData);

float activation_Sigma(float input, float bias);

float vecMulti(float* vec1, float* vec2, int vecL);

void avePooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize);

void nnFf(float* output, float* input, float** wdata, float* bias, nSize nnSize);

void cnnClear(CNN* cnn);


#endif





