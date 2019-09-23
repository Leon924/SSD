#ifndef __CNN__
#define __CNN__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"
#include "minst.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

//卷积层 convL
typedef struct convolutional_layer {
	int inputWidth; //输入图像的宽
	int inputHeight; //输入图像的高
	int mapSize; //卷积核的大小

	int inChannels; //输入图像的数目
	int outChannels; //输出图像的数目

	//卷积核的权重，是一个四维的数组
	//其大小为outChannels * inChannels * mapSize * mapSize
	float**** mapData;//存放卷积核的数据

	float* biasData; //偏置，偏置的大小为outChannels
	bool isFullConnect;//是否为全连接
	bool* connectmode; //连接模式（默认为全连接）

	//下面三个变量的输出维度和输出的维度相同
	float*** v;//进入激活函数的值
	float*** y;//激活后，神经元的输出

}CovLayer;


//采样层 pooling
typedef struct pooling_layer {
	int inputWidth; //输入图像的宽
	int inputHeight; //输入图像的长
	int mapSize; // 卷积核的大小

	int inchannels; //输入的通道数目
	int outchannels; //输出的通道数目

	int pooltype; //池化的方法
	float* biasData; //偏置

	//无需激活，没有V
	float*** y;//采样后神经元的输出，没有激活函数
}PoolLayer;

//输出层 全连接的神经网络
typedef struct nn_layer {
	int inputNum; //输入数据的数目
	int outputNum; //输出数据的数目

	float** wData; //权重数据，inputNum * outputNum大小
	float* biasData; //偏置，大小为outputNum

	float* v;  //进入激活函数的值
	float* y; //激活函数的输出

	bool isFullConnect; //是否为全连接
}OutLayer;

//卷积神经网络的定义
typedef struct cnn_network {
	int layerNum;
	CovLayer* C1;
	PoolLayer* P2;
	CovLayer* C3;
	PoolLayer* P4;
	OutLayer* O5;
}CNN;


void cnnSetup(CNN* cnn, nSize inputsize, int outputSize);

//测试cnn函数
float cnnTest(CNN* cnn, ImgArr inputData, LabelArr outputdata, int testNum);

//导入cnn的数据
void importCnn(CNN* cnn, const char* filename);

//初始化卷积层
CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
//void CovLayerConnect(CovLayer* covL, bool* connecctmode);

//初始化采样层
PoolLayer* initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType);
//void PoolLayerConnect(PoolLayer* poolL, bool* connectmode);

//初始化输出层
OutLayer* initOutLayer(int inputNum, int outputNum);

void cnnFf(CNN* cnn, float** inputData);

//激活函数 input是数据，inputNum说明数据数目，bias表明偏置、
float activation_Sigma(float input, float bias);


float vecMulti(float* vec1, float* vec2, int vecL);

//Pooling function  input 输入数据 inputNum 输入数据数目  mapSize 求平均的模块区域
void avePooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize);

//单层全连接神经网络的前向传播
//nSize是网络的大小
void nnFf(float* output, float* input, float** wdata, float* bias, nSize nnSize);

void cnnClear(CNN* cnn);
#endif





