/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Ch03 Samples
 *
 *        Version:  1.0
 *        Created:  07/14/2021 10:41:21 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@ewha.ac.kr
 *   Organization:  EWHA Womans Unversity
 *
 * =====================================================================================
 */

#include<iostream>
#include "mkPpm.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"

using namespace std;

const int MAX_ITER = 1000;

void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	for(int i = 0; i < w * h * 3; i+=3){
		unsigned char tmpRgb = inArray[i] * 0.21f + inArray[i+1] * 0.71f + inArray[i + 2] * 0.07f;
		outArray[i] = tmpRgb;
		outArray[i+1] = tmpRgb;
		outArray[i+2] = tmpRgb;
	}
}

__global__
void gpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = (row * w + col) * 3;

	if(col < w && row < h){
		unsigned char tmpRgb = inArray[offset] * 0.21f + inArray[offset + 1] * 0.71f + inArray[offset + 2] * 0.07f;
		outArray[offset] = outArray[offset + 1] = outArray[offset + 2] = tmpRgb;
	}
}

int main(){
	int w, h;
	unsigned char *h_imageArray;
	unsigned char *h_outImageArray;
	unsigned char *d_imageArray;
	unsigned char *d_outImageArray;
	unsigned char *h_outImageArray2;

	ppmLoad("./data/ewha_picture.ppm", &h_imageArray, &w, &h);

	size_t arraySize = sizeof(unsigned char) * h * w * 3;
	h_outImageArray = (unsigned char*)malloc(arraySize);
	h_outImageArray2 = (unsigned char*)malloc(arraySize);

	cudaError_t err = cudaMalloc((void **) &d_imageArray, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_outImageArray, arraySize);
	checkCudaError(err);

	err = cudaMemcpy(d_imageArray, h_imageArray, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	const int tSize = 16;
	dim3 blockSize(tSize, tSize, 1);
	dim3 gridSize(ceil((float)w/tSize), ceil((float)h/tSize), 1);

	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	mkClockMeasure *ckGpu = new mkClockMeasure("GPU CODE");

	ckCpu->clockReset();
	ckGpu->clockReset();


	for(int i = 0; i < MAX_ITER; i++){
		
		ckCpu->clockResume();
		cpuCode(h_outImageArray, h_imageArray, w, h);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuCode<<<gridSize, blockSize>>>(d_outImageArray, d_imageArray, w, h);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);

	}
	ckCpu->clockPrint();
	ckGpu->clockPrint();

	err = cudaMemcpy(h_outImageArray2, d_outImageArray, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);
			
	ppmSave("ewha_picture_cpu.ppm", h_outImageArray, w, h);
	ppmSave("ewha_picture_gpu.ppm", h_outImageArray2, w, h);
	return 0;

		

}


