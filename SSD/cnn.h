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

//����� convL
typedef struct convolutional_layer {
	int inputWidth; //����ͼ��Ŀ�
	int inputHeight; //����ͼ��ĸ�
	int mapSize; //����˵Ĵ�С

	int inChannels; //����ͼ�����Ŀ
	int outChannels; //���ͼ�����Ŀ

	//����˵�Ȩ�أ���һ����ά������
	//���СΪoutChannels * inChannels * mapSize * mapSize
	float**** mapData;//��ž���˵�����

	float* biasData; //ƫ�ã�ƫ�õĴ�СΪoutChannels
	bool isFullConnect;//�Ƿ�Ϊȫ����
	bool* connectmode; //����ģʽ��Ĭ��Ϊȫ���ӣ�

	//�����������������ά�Ⱥ������ά����ͬ
	float*** v;//���뼤�����ֵ
	float*** y;//�������Ԫ�����

}CovLayer;


//������ pooling
typedef struct pooling_layer {
	int inputWidth; //����ͼ��Ŀ�
	int inputHeight; //����ͼ��ĳ�
	int mapSize; // ����˵Ĵ�С

	int inchannels; //�����ͨ����Ŀ
	int outchannels; //�����ͨ����Ŀ

	int pooltype; //�ػ��ķ���
	float* biasData; //ƫ��

	//���輤�û��V
	float*** y;//��������Ԫ�������û�м����
}PoolLayer;

//����� ȫ���ӵ�������
typedef struct nn_layer {
	int inputNum; //�������ݵ���Ŀ
	int outputNum; //������ݵ���Ŀ

	float** wData; //Ȩ�����ݣ�inputNum * outputNum��С
	float* biasData; //ƫ�ã���СΪoutputNum

	float* v;  //���뼤�����ֵ
	float* y; //����������

	bool isFullConnect; //�Ƿ�Ϊȫ����
}OutLayer;

//���������Ķ���
typedef struct cnn_network {
	int layerNum;
	CovLayer* C1;
	PoolLayer* P2;
	CovLayer* C3;
	PoolLayer* P4;
	OutLayer* O5;
}CNN;


void cnnSetup(CNN* cnn, nSize inputsize, int outputSize);

//����cnn����
float cnnTest(CNN* cnn, ImgArr inputData, LabelArr outputdata, int testNum);

//����cnn������
void importCnn(CNN* cnn, const char* filename);

//��ʼ�������
CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
//void CovLayerConnect(CovLayer* covL, bool* connecctmode);

//��ʼ��������
PoolLayer* initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType);
//void PoolLayerConnect(PoolLayer* poolL, bool* connectmode);

//��ʼ�������
OutLayer* initOutLayer(int inputNum, int outputNum);

void cnnFf(CNN* cnn, float** inputData);

//����� input�����ݣ�inputNum˵��������Ŀ��bias����ƫ�á�
float activation_Sigma(float input, float bias);


float vecMulti(float* vec1, float* vec2, int vecL);

//Pooling function  input �������� inputNum ����������Ŀ  mapSize ��ƽ����ģ������
void avePooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize);

//����ȫ�����������ǰ�򴫲�
//nSize������Ĵ�С
void nnFf(float* output, float* input, float** wdata, float* bias, nSize nnSize);

void cnnClear(CNN* cnn);
#endif





