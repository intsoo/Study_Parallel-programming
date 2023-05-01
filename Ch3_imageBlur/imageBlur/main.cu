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

#define BLUR_SIZE 5

using namespace std;

const int MAX_ITER = 10;


/*
void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	int pixels = 0;
	unsigned char val=0;
	for(int row=0; row<h; row++){
		for(int col=0; col<w; col++){
			for(int rowOffset = -BLUR_SIZE; rowOffset < BLUR_SIZE+1; rowOffset++){
				for(int colOffset = -BLUR_SIZE; colOffset < BLUR_SIZE+1; colOffset++){
					int curRow = row + rowOffset;
					int curCol = col + colOffset;
	
					if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
						int curIndex = (curRow * w + curCol);
						val += inArray[curIndex];
						pixels++;
					}
				}
			}
			int index = row*w+col;
			outArray[index] = (unsigned char)(val/pixels);
			pixels = 0;
			val = 0;
		}
	}
}
*/

void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
//	int pixels = 0;
//	float avgR = avgG = avgB = 0;

	for(int row=0; row<h; row++){
		for(int col=0; col<w; col++){
			float avgR=0, avgG = 0, avgB = 0;  // 실수: 'unsigned char' type -> blur처리 안 되고 검은 화면 뜸..
			int pixels = 0, index;
			for(int rowOffset = -BLUR_SIZE; rowOffset < BLUR_SIZE+1; rowOffset++){
				for(int colOffset = -BLUR_SIZE; colOffset < BLUR_SIZE+1; colOffset++){
					int curRow = row + rowOffset;
					int curCol = col + colOffset;
	
					if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
						int curIndex = (curRow * w + curCol) * 3;
						avgR += inArray[curIndex];
						avgG += inArray[curIndex+1];
						avgB += inArray[curIndex+2];
						pixels++;
					}
				}
			}
			avgR = (unsigned char)(avgR/pixels);
			avgG = (unsigned char)(avgG/pixels);
			avgB = (unsigned char)(avgB/pixels);
			
			index = (row * w + col) * 3;
			outArray[index] = avgR;
			outArray[index+1] = avgG;
			outArray[index+2] = avgB;

			//avgR = avgG = avgB = pixels = 0;
		}
	}
}


__global__
void gpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	float avgR=0, avgG = 0, avgB = 0;  // 'unsigned char' type -> blur처리 안 되고 검은 화면 뜸..
	int pixels = 0;
	int index;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for(int rowOffset = -BLUR_SIZE; rowOffset < BLUR_SIZE+1; rowOffset++){
		for(int colOffset = -BLUR_SIZE; colOffset < BLUR_SIZE+1; colOffset++){
			int curRow = row + rowOffset;
			int curCol = col + colOffset;
	
			if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
				int curIndex = (curRow * w + curCol) * 3;
				avgR += inArray[curIndex];
				avgG += inArray[curIndex+1];
				avgB += inArray[curIndex+2];
				pixels++;
			}
		}
	}
	avgR = (unsigned char)(avgR/pixels);
	avgG = (unsigned char)(avgG/pixels);
	avgB = (unsigned char)(avgB/pixels);
			
	index = (row * w + col) * 3;
	outArray[index] = avgR;
	outArray[index+1] = avgG;
	outArray[index+2] = avgB;

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
