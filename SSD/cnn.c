#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cnn.h"
#include "mat.h"


void cnnSetup(CNN* cnn, nSize inputSize, int outputSize) {
	cnn->layerNum = 5;

	nSize inSize;
	int mapSize = 5;
	inSize.c = inputSize.c;
	inSize.r = inputSize.r;
	cnn->C1 = initCovLayer(inSize.c, inSize.r, 5, 1, 6);
	inSize.c = inSize.c - mapSize + 1;
	inSize.r = inSize.r - mapSize + 1;

	cnn->P2 = initPoolLayer(inSize.c, inSize.r, 2, 6, 6, AvePool);//stride equals 2
	inSize.c = inSize.c / 2;
	inSize.r = inSize.r / 2;

	cnn->C3 = initCovLayer(inSize.c, inSize.r, 5, 6, 12);
	inSize.c = inSize.c - mapSize + 1;
	inSize.r = inSize.r - mapSize + 1;

	cnn->P4 = initPoolLayer(inSize.c, inSize.r, 2, 12, 12, AvePool);//stride equals 2
	inSize.c = inSize.c / 2;
	inSize.r = inSize.r / 2;

	cnn->O5 = initOutLayer(inSize.c * inSize.r * 12, outputSize);

	//cnn->e = (float*)calloc(cnn->O5->outputNum, sizeof(float));
}

//import the data ofg cnn
void importCnn(CNN* cnn, const char* filename) {

	int i, j, c, r;
	for (i = 0; i < cnn->C1->outChannels; i++)
		for (j = 0; j < cnn->C1->inChannels; j++)
			for (r = 0; r < cnn->C1->mapSize; r++)
				for (c = 0; c < cnn->C1->mapSize; c++) {
					cnn->C1->mapData[i][j][r][c] = (c % 2);
				}
	for (i = 0; i < cnn->C1->outChannels; i++)
		cnn->C1->biasData[i] = i % 2;

	//C3 convLayer
	for (i = 0; i < cnn->C3->outChannels; i++)
		for (j = 0; j < cnn->C3->inChannels; j++)
			for (r = 0; r < cnn->C3->mapSize; r++)
				for (c = 0; c < cnn->C3->mapSize; c++)
					cnn->C3->mapData[i][j][r][c] = (c % 2);

	for (i = 0; i < cnn->C3->outChannels; i++)
		cnn->C3->biasData[i] = i % 2;

	//O5 outputLayer
	for (i = 0; i < cnn->O5->outputNum; i++)
		for (j = 0; j < cnn->O5->inputNum; j++)
			cnn->O5->wData[i][j] = j % 2;

	for (i = 0; i < cnn->O5->outputNum; i++)
		cnn->O5->biasData[i] = i % 2;

}

CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{
	CovLayer* covL = (CovLayer*)malloc(sizeof(CovLayer));

	covL->inputHeight = inputHeight;
	covL->inputWidth = inputWidth;
	covL->mapSize = mapSize;

	covL->inChannels = inChannels;
	covL->outChannels = outChannels;

	covL->isFullConnect = True;

	//initialize mapData
	int i, j, c, r;
	covL->mapData = (float****)malloc(outChannels * sizeof(float***));
	for (i = 0; i < outChannels; i++) {
		covL->mapData[i] = (float***)malloc(inChannels * sizeof(float**));
		for (j = 0; j < inChannels; j++) {
			covL->mapData[i][j] = (float**)malloc(mapSize * sizeof(float*));
			for (r = 0; r < mapSize; r++) {
				covL->mapData[i][j][r] = (float*)malloc(mapSize * sizeof(float));
				for (c = 0; c < mapSize; c++) {
					covL->mapData[i][j][r][c] = (c % 2);
				}
			}
		}
	}

	//Every outchannels has a same bias variable
	covL->biasData = (float*)calloc(outChannels, sizeof(float));

	int outW = inputWidth - mapSize + 1;
	int outH = inputHeight - mapSize + 1;

	covL->v = (float***)malloc(outChannels * sizeof(float**));
	covL->y = (float***)malloc(outChannels * sizeof(float**));

	for (j = 0; j < outChannels; j++) {
		covL->v[j] = (float**)malloc(outH * sizeof(float*));
		covL->y[j] = (float**)malloc(outH * sizeof(float*));
		for (r = 0; r < outH; r++) {
			covL->v[j][r] = (float*)calloc(outW, sizeof(float));
			covL->y[j][r] = (float*)calloc(outW, sizeof(float));
		}
	}

	return covL;
}

PoolLayer* initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType) {
	PoolLayer* poolL = (PoolLayer*)malloc(sizeof(PoolLayer));

	poolL->inputHeight = inputHeight;
	poolL->inputWidth = inputWidth;

	poolL->mapSize = mapSize;
	poolL->inchannels = inChannels;
	poolL->outchannels = outChannels;

	poolL->pooltype = poolType;

	poolL->biasData = (float*)calloc(outChannels, sizeof(float));

	int outW = inputWidth / mapSize;
	int outH = inputHeight / mapSize;

	int j, r;

	poolL->y = (float***)malloc(outChannels * sizeof(float**));
	for (j = 0; j < outChannels; j++) {
		poolL->y[j] = (float**)malloc(outH * sizeof(float*));
		for (r = 0; r < outH; r++) {
			poolL->y[j][r] = (float*)calloc(outW, sizeof(float));
		}
	}

	return poolL;
}

OutLayer* initOutLayer(int inputNum, int outputNum) {
	OutLayer* outL = (OutLayer*)malloc(sizeof(OutLayer));

	outL->inputNum = inputNum;
	outL->outputNum = outputNum;

	outL->biasData = (float*)calloc(outputNum, sizeof(float));

	outL->v = (float*)calloc(outputNum, sizeof(float));
	outL->y = (float*)calloc(outputNum, sizeof(float));

	//initialize weight
	outL->wData = (float**)malloc(outputNum * sizeof(float*));
	int i, j;

	for (i = 0; i < outputNum; i++) {
		outL->wData[i] = (float*)malloc(inputNum * sizeof(float));
		for (j = 0; j < inputNum; j++) {
			outL->wData[i][j] = rand() % 2;
		}
	}
	outL->isFullConnect = True;
	return outL;
}

//return the subscripts of the biggest element of this vector
int vecmaxIndex(float* vec, int veclength) {
	int i;
	float maxnum = -1.0;
	int maxIndex = 0;
	for (i = 0; i < veclength; i++) {
		if (maxnum < vec[i]) {
			maxnum = vec[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

float cnnTest(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum) {
	int n = 0;
	int incorrectNum = 0;
	for (n = 0; n < testNum; n++) {
		cnnFf(cnn, inputData->ImgPtr[n].ImgData);
		if (vecmaxIndex(cnn->O5->y, cnn->O5->outputNum) != vecmaxIndex(outputData->LabelPtr[n].LabelData, cnn->O5->outputNum))
			incorrectNum++;
		cnnClear(cnn);
	}
	return (float)incorrectNum / (float)testNum;
}

void cnnFf(CNN* cnn, float** inputData) {
	int outSizeW = cnn->P2->inputWidth;
	int outSizeH = cnn->P2->inputHeight;
	//The feedforward propogation in the first layer
	int i, j, r, c;
	//Size of kernel, input featuremap, and output featuremap
	nSize mapSize = { cnn->C1->mapSize, cnn->C1->mapSize };
	nSize inSize = { cnn->C1->inputWidth, cnn->C1->inputHeight };
	nSize outSize = { cnn->P2->inputWidth, cnn->P2->inputHeight };
	//w * x + b
	for (i = 0; i < (cnn->C1->outChannels); i++) {
		for (j = 0; j < (cnn->C1->inChannels); j++) {
			float** mapout = conv(cnn->C1->mapData[i][j], mapSize, inputData, inSize, valid);
			addMat(cnn->C1->v[i], cnn->C1->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
				free(mapout[r]);
			free(mapout);
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++)
				cnn->C1->y[i][r][c] = activation_Sigma(cnn->C1->v[i][r][c], cnn->C1->biasData[i]);
		}
	}

	//S2 Layer, sub-sampling Layer
	outSize.c = cnn->C3->inputWidth;
	outSize.r = cnn->C3->inputHeight;
	inSize.c = cnn->P2->inputWidth;
	inSize.r = cnn->P2->inputHeight;
	for (i = 0; i < cnn->P2->outchannels; i++) {
		if (cnn->P2->pooltype == AvePool)
			avePooling(cnn->P2->y[i], outSize, cnn->C1->y[i], inSize, cnn->P2->mapSize);
	}

	//C3 Layer, conv Layer
	outSize.c = cnn->P4->inputWidth;
	outSize.r = cnn->P4->inputHeight;
	inSize.c = cnn->C3->inputWidth;
	inSize.r = cnn->C3->inputHeight;
	mapSize.c = cnn->C3->mapSize;
	mapSize.r = cnn->C3->mapSize;

	for (i = 0; i < cnn->C3->outChannels; i++) {
		for (j = 0; j < cnn->C3->inChannels; j++) {
			float** mapout = conv(cnn->C3->mapData[i][j], mapSize, cnn->P2->y[j], inSize, valid);
			addMat(cnn->C3->v[i], cnn->C3->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
				free(mapout[r]);
			free(mapout);
		}
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->biasData[i]);
	}

	//S4 Layer, subsampling Layer
	inSize.c = cnn->P4->inputWidth;
	inSize.r = cnn->P4->inputHeight;
	outSize.c = inSize.c / cnn->P4->mapSize;
	outSize.r = inSize.r / cnn->P4->mapSize;
	for (i = 0; i < cnn->P4->outchannels; i++) {
		if (cnn->P4->pooltype == AvePool)
			avePooling(cnn->P4->y[i], outSize, cnn->C3->y[i], inSize, cnn->P4->mapSize);
	}

	//output Layer
	//expand the ouput of S4, muti-dimensional vector to one dimensional vector
	float* O5inData = (float*)malloc((cnn->O5->inputNum) * sizeof(float));
	for (i = 0; i < cnn->P4->outchannels; i++)
		for (r = 0; r < outSize.r; r++)
			for (c = 0; c < outSize.c; c++)
				O5inData[i * outSize.r * outSize.c + r * outSize.c + c] = cnn->P4->y[i][r][c];

	nSize nnSize = { cnn->O5->inputNum, cnn->O5->outputNum };
	nnFf(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->biasData, nnSize);//Vector multiplication
	for (i = 0; i < cnn->O5->outputNum; i++)
		cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->biasData[i]);
	free(O5inData);
}


//activation function
float activation_Sigma(float input, float bias) {
	float temp = input + bias;
	return (float)1.0 / ((float)(1.0 + exp(-temp)));
}

// average pooling
void avePooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize)
{
	int outputW = inputSize.c / mapSize;
	int outputH = inputSize.r / mapSize;
	if (outputSize.c != outputW || outputSize.r != outputH)
		printf("ERROR; output size is wrong!\n");

	int i, j, m, n;
	for (i = 0; i < outputH; i++)
		for (j = 0; j < outputW; j++) {
			float sum = 0.0;
			for (m = i * mapSize; m < i * mapSize + mapSize; m++)
				for (n = j * mapSize; n < j * mapSize + mapSize; n++)
					sum = sum + input[m][n];

			output[i][j] = sum / (float)(mapSize * mapSize);
		}
}

//The feedforward propogation of FcLayer
float vecMulti(float* vec1, float* vec2, int vecL) {
	int i;
	float m = 0;
	for (i = 0; i < vecL; i++)
		m = m + vec1[i] * vec2[i];//the element of vector multiplies.
	return m;
}

void nnFf(float* output, float* input, float** wdata, float* bias, nSize nnSize) {
	int w = nnSize.c;//the length of vector
	int h = nnSize.r;

	int i;
	for (i = 0; i < h; i++)
		output[i] = vecMulti(input, wdata[i], w) + bias[i];
}

//Clear the data in CNN
void cnnClear(CNN* cnn) {
	int j, c, r;
	//C1 Layer
	for (j = 0; j < cnn->C1->outChannels; j++) {
		for (r = 0; r < cnn->P2->inputHeight; r++) {
			for (c = 0; c < cnn->P2->inputWidth; c++) {
				cnn->C1->v[j][r][c] = (float)0.0;
				cnn->C1->y[j][r][c] = (float)0.0;
			}
		}
	}

	//S2 Layer
	for (j = 0; j < cnn->P2->outchannels; j++) {
		for (r = 0; r < cnn->C3->inputHeight; r++) {
			for (c = 0; c < cnn->C3->inputWidth; c++) {
				cnn->P2->y[j][r][c] = (float)0.0;
			}
		}
	}

	//C3 Layer
	for (j = 0; j < cnn->C3->outChannels; j++) {
		for (r = 0; r < cnn->P4->inputHeight; r++) {
			for (c = 0; c < cnn->P4->inputWidth; c++) {
				cnn->C3->v[j][r][c] = (float)0.0;
				cnn->C3->y[j][r][c] = (float)0.0;
			}
		}
	}

	//S4 Layer
	for (j = 0; j < cnn->P4->outchannels; j++) {
		for (r = 0; r < cnn->P4->inputHeight / cnn->P4->mapSize; r++) {
			for (c = 0; c < cnn->P4->inputWidth / cnn->P4->mapSize; c++) {
				cnn->P4->y[j][r][c] = (float)0.0;
			}
		}
	}

	//O5 Layer
	for (j = 0; j < cnn->O5->outputNum; j++) {
		cnn->O5->v[j] = (float)0.0;
	}
}


