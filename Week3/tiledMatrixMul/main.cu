/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description: 	Matrix Multiplication (Using Shared Memory)
 *
 *        Version:  1.0
 *        Created:  2021/07/30 10:07:38
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon, myungkuk.yoon@ewha.ac.kr
 *   Organization:  Ewha Womans Universityi
 *
 * =====================================================================================
 */

#include <assert.h>
#include "mkCuda.h"
#include "mkClockMeasure.h"

#define TILE_WIDTH 16

const int A_H = 512;
const int A_W = 1024;
const int B_H = A_W;
const int B_W = 1024;
const unsigned int MAX_NUM = 10;
const int MAX_ITER = 10;

unsigned int matrixA[A_H * A_W];
unsigned int matrixB[B_H * B_W];
unsigned int cpuOut[A_H * B_W];
unsigned int gpuOut_2D[A_H * B_W];
unsigned int gpuOut_tiled[A_H * B_W];

void generateRandomValues(unsigned int *input, const int rowSize, const int colSize){
	for(int i = 0; i < rowSize; i++){
		for(int j = 0; j < colSize; j++){
			input[i * colSize + j] = (unsigned int) float(rand())/float(RAND_MAX) * MAX_NUM;
		}
	}
}

void printMatrixValue(const unsigned int *input, const int rowSize, const int colSize){
	printf("Print Matrix \n -----------\n");
	for(int i = 0; i < rowSize; i++){
		for(int j = 0; j < colSize; j++){
			printf("%u\t", input[i * colSize + j]);
		}
		printf("\n");
	}
	printf("--------\n");
}

bool compareMatrix(const unsigned int *inputA, const unsigned int *inputB, const int rowSize, const int colSize){
	bool ret = true;
	for(int i = 0; i < rowSize * colSize; i++){
		if(inputA[i] != inputB[i]){
			ret = false;
			break;
		}
	}
	return ret;
}

void cpuMatrixMul(const unsigned int *h_a, const unsigned int *h_b, unsigned int *h_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	int height = aRowSize, width = bColSize;
	int sum = 0;

	for(int row=0; row<height; row++){
		for(int col=0; col<width; col++){
			for(int k=0; k<aColSize; k++){
				sum += h_a[row*aColSize+k] * h_b[k*bColSize+col];  // sum = A[row][k] * B[k][col]
			}
			h_c[row*width+col] = sum;  // C[row][col] = sum
			sum = 0;
		}
	}
}


__global__
void gpuMatrixMul_2D(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int sum = 0;

	if((col < bColSize) && (row < aRowSize)){
		for(int k=0; k<aColSize; k++){
			sum += d_a[row*aColSize + k] * d_b[k*bColSize + col];
		}
		d_c[row*bColSize + col] = sum;
	}
}


__global__
void gpuTiledMatrixMul(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);

	// Declaration of Shared Memory Variables
	__shared__ int Ads[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Bds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
	int col = bx * TILE_WIDTH + tx;  // (?) blockDim.x == TILE_WIDTH (?)
	int row = by * TILE_WIDTH + ty;
	int sum = 0;
	int total_phase_num = ceil((float)aColSize/(float)TILE_WIDTH);

	for(int ph=0; ph<total_phase_num; ph++){
		// 1. Load matrix tiles to the shared memory.
		if((row < aRowSize) && ((ph*TILE_WIDTH+tx) < aColSize))
			Ads[ty][tx] = d_a[row*aColSize+ ph*TILE_WIDTH + tx];
		if(((ph*TILE_WIDTH+ty) < bRowSize) && (col < bColSize))
			Bds[ty][tx] = d_b[(ph*TILE_WIDTH + ty)*bColSize + col];
		
		__syncthreads();  // Wait until all the threads are finished writing on the shared memory.
		
		// 2. Do the inner product of two elements.
		for(int k=0; k<TILE_WIDTH; k++){
			sum += Ads[ty][k] * Bds[k][tx];
		}
		__syncthreads();  // Wait until all the threads are done reading the shared memory. 

		/* Oops... 각 phase에서는 sum에 내적한 값을 누적해주고, 모든phase의 연산 값들을 모두 더한 최종값만 global MEM의output matrix에 저장하는 것! -> 바깥 for문 밖에서 최종값을 저장해야 효율적이다.
		// 3. Store the sum of the inner products into the output matrix.
		if((row < aRowSize) && (col < bColSize)) {
			d_c[row*bColSize + col] = sum;  // C[row][col] = sum
		}
		*/
	}
		// 3. Store the sum of the inner products into the output matrix.
		if((row < aRowSize) && (col < bColSize)) {
			d_c[row*bColSize + col] = sum;  // C[row][col] = su
		}
}
	

int main(){
	srand((unsigned int)time(NULL));
	generateRandomValues(matrixA, A_H, A_W);
	generateRandomValues(matrixB, B_H, B_W);

	//MK: GPU Memory 
	unsigned int *d_a, *d_b, *d_c_2D, *d_c_tiled;
	size_t matrixSizeA = sizeof(unsigned int) * A_H * A_W;
	size_t matrixSizeB = sizeof(unsigned int) * B_H * B_W;
	size_t matrixSizeC = sizeof(unsigned int) * A_H * B_W;

	cudaError_t err = cudaMalloc((void **) &d_a, matrixSizeA);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_b, matrixSizeB);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_c_2D, matrixSizeC);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_c_tiled, matrixSizeC);
	checkCudaError(err);

	err = cudaMemcpy(d_a, matrixA, matrixSizeA, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(d_b, matrixB, matrixSizeB, cudaMemcpyHostToDevice);
	checkCudaError(err);

	//MK: Thread Num
	const int tbDimX_2D = TILE_WIDTH, tbDimY_2D = TILE_WIDTH;

	// 2D Thread Organization
	dim3 gridSize_2D(ceil((float)B_W/(float)tbDimX_2D), ceil((float)A_H/tbDimY_2D)); 	
	dim3 gridSize_tiled(ceil((float)B_W/(float)tbDimX_2D), ceil((float)A_H/tbDimY_2D)); 	
	dim3 blockSize_2D(tbDimX_2D, tbDimY_2D);  // block은 tile과 dimension이 같게!
	dim3 blockSize_tiled(tbDimX_2D, tbDimY_2D); 

	//MK: Time Measurement
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	mkClockMeasure *ckGpu_2D = new mkClockMeasure("GPU CODE(2D)");
	ckGpu_2D->clockReset();

	mkClockMeasure *ckGpu_tiled = new mkClockMeasure("GPU CODE(Tiled)");
	ckGpu_tiled->clockReset();

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuMatrixMul(matrixA, matrixB, cpuOut, A_H, A_W, B_H, B_W);
		ckCpu->clockPause();

		ckGpu_2D->clockResume();
		gpuMatrixMul_2D<<<gridSize_2D, blockSize_2D>>>(d_a, d_b, d_c_2D, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu_2D->clockPause();
		checkCudaError(err);

		ckGpu_tiled->clockResume();
		gpuTiledMatrixMul<<<gridSize_tiled, blockSize_tiled>>>(d_a, d_b, d_c_tiled, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu_tiled->clockPause();
		checkCudaError(err);
	}

	err = cudaMemcpy(gpuOut_2D, d_c_2D, matrixSizeC, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpuOut_tiled, d_c_tiled, matrixSizeC, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c_2D);
	cudaFree(d_c_tiled);

	if(compareMatrix(cpuOut, gpuOut_2D, A_H, B_W) && compareMatrix(cpuOut, gpuOut_tiled, A_H, B_W)){
		ckCpu->clockPrint();
		ckGpu_2D->clockPrint();
		ckGpu_tiled->clockPrint();
	}else{
		printf("ERROR: The Three Matrices are not same\n");
	}

//	printMatrixValue(matrixA, A_H, A_W);
//	printMatrixValue(matrixB, B_H, B_W);
//	printMatrixValue(matrixZero, A_H, B_W);
//	printMatrixValue(cpuOut, A_H, B_W);
//	printMatrixValue(gpuOut_2D, A_H, B_W);

}
