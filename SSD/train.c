#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cnn.h"
#include "minst.h"
#include "mat.h"


int main()
{
	LabelArr trainLabel = read_Lable("E:\\VSproject\\tostq\\DeepLearningC-master\\CNN\\Minst\\train-labels.idx1-ubyte");
	ImgArr trainImg = read_Img("E:\\VSproject\\tostq\\DeepLearningC-master\\CNN\\Minst\\train-images.idx3-ubyte");
	LabelArr testLabel = read_Lable("E:\\VSproject\\tostq\\DeepLearningC-master\\CNN\\Minst\\test-labels.idx1-ubyte");
	ImgArr testImg = read_Img("E:\\VSproject\\tostq\\DeepLearningC-master\\CNN\\Minst\\test-images.idx3-ubyte");

	nSize inputSize = { testImg->ImgPtr[0].c,testImg->ImgPtr[0].r };
	int outSize = testLabel->LabelPtr[0].l;

	// CNN结构的初始化
	CNN* cnn = (CNN*)malloc(sizeof(CNN));
	cnnsetup(cnn, inputSize, outSize);

	// CNN训练

	CNNOpts opts;
	opts.numepochs = 1;
	opts.alpha = 1.0;
	int trainNum = 55000;
	cnntrain(cnn, trainImg, trainLabel, opts, trainNum);
	printf("train finished!!\n");
	savecnn(cnn, "minst.cnn");
	// 保存训练误差
	FILE* fp = NULL;
	fp = fopen("E:\\Code\\Matlab\\PicTrans\\cnnL.ma", "wb");
	if (fp == NULL)
		printf("write file failed\n");
	fwrite(cnn->L, sizeof(float), trainNum, fp);
	fclose(fp);



	// CNN测试
	importcnn(cnn, "minst.cnn");
	int testNum = 10000;
	float incorrectRatio = 0.0;
	incorrectRatio = cnntest(cnn, testImg, testLabel, testNum);
	printf("test finished!!\n");

	return 0;
}