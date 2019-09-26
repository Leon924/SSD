#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "cnn.h"
#include "minst.h"


int  main(){
	int testNum = 10;
	LabelArr testLabel = read_Label(testNum);
	ImgArr testImg = read_Img(testNum);

	nSize inputSize = { testImg->ImgPtr[0].c, testImg->ImgPtr[0].r };
	int outSize = testLabel->LabelPtr[0].l;

	//init and setup
	CNN* cnn = (CNN*)malloc(sizeof(CNN));
	cnnSetup(cnn, inputSize, outSize);

	//import weight
	importCnn(cnn, "minst.cnn");
	float incorrectRatio = 0.0;
	//Feed forward, Test this CNN
	incorrectRatio = cnnTest(cnn, testImg, testLabel, testNum);
	printf("%lf\n", incorrectRatio);
	cnnClear(cnn);
	printf("test finished!\n");

	return 0;
}