#ifndef __MINST__ 
#define __MINST__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
/*
	MINST��һ����дͼ�����ݿ�
*/

typedef struct MinstImg {
	int c;	//	ͼ���
	int r;	//	ͼ���
	float** ImgData; //ͼ�����ݶ�ά��̬����
}MinstImg;

typedef struct MinstImgArr {
	int ImgNum; //�洢ͼ�����Ŀ
	MinstImg* ImgPtr; //�洢ͼ�������ָ��
}*ImgArr;

typedef struct MnistLabel {
	int l;	//�����ǵĳ�
	float* LabelData; //����������
}MinstLabel;

typedef struct MinstLabelArr {
	int LabelNum; // ��ǵ���Ŀ
	MinstLabel* LabelPtr; //ָ���ǵ�ָ��
}*LabelArr;

LabelArr read_Label(int); //����ͼ����

ImgArr read_Img(int); //����ͼ��

#endif


