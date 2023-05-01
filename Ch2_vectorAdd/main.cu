// Vector Addition of Two-dimensional Matrices 

/*
   dim3 DimGrid(10, 10);  // A grid is a 10*10 thread block array.
   dim3 DimBlock(100, 100);  // Each block is a 100*100 thread array.
   kernelFunc<<< DimGrid, DimBlock>>>();
*/

#include <stdio.h>
#include <cuda.h>
#define ROW 10
#define COL 10

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
	/*
	0 < blockIdx.x < gridDim.x = total number of blocks in the x dimension(column) of a grid
	0 < threadIdx.x < blockDim.x = total number of threads in the x dimension of a thread block
	WE DON'T NEED A TWO-DIMENSIONAL GRID IN THIS PROGRAM! = NO 'blockIdx.y'
	*/
	
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if(i<n) C[i] = A[i] + B[i];
}


void vecAdd(float (* h_A)[COL], float (*h_B)[COL], float (*h_C)[COL], int n)
{
	int size = n * sizeof(float);
	// float **d_A, **d_B, **d_C;  // Double pointer: dynamic allocation of two dimensional array
	float *d_A, *d_B, *d_C;
	int i;

   	dim3 DimBlock(10, 10);  // Each block is a 10*10 thread array.
	
	// Part 1: Allocate device global memory and copy the input data from host to device
	
	/* ERROR
	cudaMalloc((void ***)&d_A, row * sizeof(float *);  // Allocate the rows of matrix A
	for(i=0; i<row; i++)
		// Allocate the columns of A
		cudaMalloc((void **)d_A[i], col * sizeof(float);  // Impossible to access the element of an array in GPU from the CPU code!!
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	*/

	cudaMalloc((void **)&d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&d_B, size);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, size);


	// Part 2: Call the kernel function that computes the actual addition
	vecAddKernel<<<1, DimBlock>>>(d_A, d_B, d_C, n);


	// Part 3: Copy the result matrix from device to host and clear the device memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main()
{
	//const int ROW = 10;
	//const int COL = 10;

	// Define three matrices.
	float a[ROW][COL];	
	float b[ROW][COL];	
	float c[ROW][COL];

	// Initialize the input matrices.
	for(int i=0; i<ROW; i++){
		for(int j=0; j<COL; j++){
			float num = i*COL + j;
			a[i][j] = b[i][j] = num;
			printf("%6.2lf ", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	// Execute the vestor addition.
	vecAdd(a, b, c,ROW*COL);
	
	// Print out the result.
	for(int i=0; i<ROW; i++){
		for(int j=0; j<COL; j++){
			printf("%6.2lf ", c[i][j]);
		}
		printf("\n");
	}
}


