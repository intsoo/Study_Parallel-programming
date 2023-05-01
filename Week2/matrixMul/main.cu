/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description: 	Matrix Multiplication (No use of Shared Memory)
 *
 *        Version:  1.0
 *        Created:  2021/07/30 10:07:38
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Myung Kuk Yoon, myungkuk.yoon@ewha.ac.kr
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#include <assert.h>
#include "mkCuda.h"
#include "mkClockMeasure.h"


const int A_H = 17;
const int A_W = 16;
const int B_H = A_W;
const int B_W = 16;
const unsigned int MAX_NUM = 5;
const int MAX_ITER = 1;

unsigned int matrixA[A_H * A_W];
unsigned int matrixB[B_H * B_W];
//unsigned int matrixC[A_H * B_W];
//unsigned int matrixZero[A_H * B_W] = {0};
unsigned int cpuOut[A_H * B_W];
unsigned int gpuOut[A_H * B_W];
unsigned int gpuOut_2D[A_H * B_W];
//unsigned int gpuOut_3D[A_H * B_W];

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
void gpuMatrixMul(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	int size = aRowSize * bColSize, width = bColSize;  // ~ of Output Matrix C
	int row, col;
	int sum = 0;

	if(tId < size) {
		// tId = row * width + column
		row = tId / width;
		col = tId % width;
		for (int k=0; k<aColSize; k++) {
			sum += d_a[row*aColSize+k] * d_b[k*bColSize+col];  // sum += A[row][k] * B[k][col]
		}
		d_c[tId] = sum;  // C[row][col] = sum
		//sum = 0;
	}
}


__global__
void gpuMatrixMul_2D(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int width = bColSize, height = aRowSize;
	int sum = 0;

	if((col < width) && (row < height)) {
		for (int k=0; k<aColSize; k++) {
			sum += d_a[row*aColSize+k] * d_b[k*bColSize+col];  // sum += A[row][k] * B[k][col]
		}
		d_c[row*width+col] = sum;  // C[row][col] = sum
		//sum = 0;
	}
}

/*
__global__
void gpuMatrixMul_3D(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize){
	assert(aColSize == bRowSize);
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;
	int row = threadIdx.z + blockIdx.z * blockDim.z;
	int val;

	if(k < aColSize && col < bColSize && row < aRowSize) {
		val = d_a[row*aColSize+k] * d_b[k*bColSize+col];  // val = A[row][k] * B[k][col]	
		int index = row*bColSize+col;
	
		// 시도 2: fail
	  	// d_c[index] += val는 곧 기존의 값에 현재 val를 더해서 저장하는 것. 기존의 값이 반영된다는 점에서 thread 간 data dependency가 있다...	   
		printf("[%d]:%d\n", index, val);
		d_c[index] += val;  // C[row][col] += val
		printf("C[%d] = %d\n", index, d_c[index]);
		

		// 시도 1: fail
		// 동시에 실행되고 있는 thread들의  경우, d_c[index]>0이 모두 거짓이 됨 (순차적으로 실행되지 않으니까)	
		// 따라서 애초에 matric C를 0으로 초기화해줘야 함!
	  	if(d_c[index] > 0){
			printf("[%d]:%d\n", index, d_c[index]);
			d_c[index] += val;  // C[row][col] += val
		}
		else
			d_c[index] = 0;
	}
}
*/

int main(){
	srand((unsigned int)time(NULL));
	generateRandomValues(matrixA, A_H, A_W);
	generateRandomValues(matrixB, B_H, B_W);

	//MK: GPU Memory 
	unsigned int *d_a, *d_b, *d_c, *d_c_2D; // *d_c_3D;
	size_t matrixSizeA = sizeof(unsigned int) * A_H * A_W;
	size_t matrixSizeB = sizeof(unsigned int) * B_H * B_W;
	size_t matrixSizeC = sizeof(unsigned int) * A_H * B_W;

	cudaError_t err = cudaMalloc((void **) &d_a, matrixSizeA);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_b, matrixSizeB);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_c, matrixSizeC);
	checkCudaError(err);	
	err = cudaMalloc((void **) &d_c_2D, matrixSizeC);
	checkCudaError(err);

	/*
	err = cudaMalloc((void **) &d_c_3D, matrixSizeC);
	checkCudaError(err);
	*/

	err = cudaMemcpy(d_a, matrixA, matrixSizeA, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(d_b, matrixB, matrixSizeB, cudaMemcpyHostToDevice);
	checkCudaError(err);

	/*
	// 3D matrix는 output matrix도 미리 다 0으로 초기화해줘야 함
	err = cudaMemcpy(d_c_3D, matrixZero, matrixSizeC, cudaMemcpyHostToDevice);
	checkCudaError(err);
	*/

	//MK: Thread Num
	const int tbSize = 256;
	const int tbDimX_2D = 16, tbDimY_2D = 16;
//	const int tbDimX_3D = 4, tbDimY_3D = 4, tbDimZ_3D = 16; 

	// 1D Thread Organization
	dim3 gridSize(ceil((float)(A_H*B_W)/(float)tbSize), 1, 1);
	dim3 blockSize(tbSize, 1, 1);

	// 실수: dim3은 (x, y, z)순이다!
	// 2D Thread Organization
	dim3 gridSize_2D(ceil((float)B_W/(float)tbDimX_2D), ceil((float)A_H/tbDimY_2D)); 	
	dim3 blockSize_2D(tbDimX_2D, tbDimY_2D);

/*
	// 3D Thread Organization
	dim3 gridSize_3D(ceil((float)A_W/tbDimZ_3D), ceil((float)A_H/tbDimY_3D), ceil((float)B_W/tbDimX_3D));
	dim3 blockSize_3D(tbDimZ_3D, tbDimY_3D, tbDimX_3D);
*/

	//MK: Time Measurement
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	mkClockMeasure *ckGpu = new mkClockMeasure("GPU CODE(1D)");
	ckGpu->clockReset();

	mkClockMeasure *ckGpu_2D = new mkClockMeasure("GPU CODE(2D)");
	ckGpu_2D->clockReset();

	/*
	mkClockMeasure *ckGpu_3D = new mkClockMeasure("GPU CODE(3D)");
	ckGpu_3D->clockReset();
	*/

	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuMatrixMul(matrixA, matrixB, cpuOut, A_H, A_W, B_H, B_W);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuMatrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);

		ckGpu_2D->clockResume();
		gpuMatrixMul_2D<<<gridSize_2D, blockSize_2D>>>(d_a, d_b, d_c_2D, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu_2D->clockPause();
		checkCudaError(err);
		/*			
		ckGpu_3D->clockResume();
		gpuMatrixMul_3D<<<gridSize_3D, blockSize_3D>>>(d_a, d_b, d_c_3D, A_H, A_W, B_H, B_W);
		err=cudaDeviceSynchronize();
		ckGpu_3D->clockPause(); 
		checkCudaError(err);
		*/		
	}

	err = cudaMemcpy(gpuOut, d_c, matrixSizeC, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpuOut_2D, d_c_2D, matrixSizeC, cudaMemcpyDeviceToHost);
	checkCudaError(err);
	
	/*
	err = cudaMemcpy(gpuOut_3D, d_c_3D, matrixSizeC, cudaMemcpyDeviceToHost);
	checkCudaError(err);
	*/

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_c_2D);
	//cudaFree(d_c_3D);


	if(compareMatrix(cpuOut, gpuOut_2D, A_H, B_W)){
//	if(compareMatrix(cpuOut, gpuOut, A_H, B_W) && compareMatrix(cpuOut, gpuOut_2D, A_H, B_W)){
		ckCpu->clockPrint();
		ckGpu->clockPrint();
		ckGpu_2D->clockPrint();
	}else{
		printf("ERROR: Three Matrices are not same\n");
	}
/*
	if(compareMatrix(cpuOut, gpuOut_3D, A_H, B_W)){
		ckCpu->clockPrint();
		ckGpu_3D->clockPrint();
	}else{
		printf("ERROR: Two Matrices are not same\n");
	}
*/

//	printMatrixValue(matrixA, A_H, A_W);
//	printMatrixValue(matrixB, B_H, B_W);
//	printMatrixValue(matrixZero, A_H, B_W);
//	printMatrixValue(cpuOut, A_H, B_W);
//	printMatrixValue(gpuOut, A_H, B_W);
//	printMatrixValue(gpuOut_2D, A_H, B_W);
	//printMatrixValue(gpuOut_3D, A_H, B_W);
}
