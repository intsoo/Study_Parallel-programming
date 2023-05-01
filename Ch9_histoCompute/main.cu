/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description: 	Matrix Multiplication
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


const unsigned int INPUT_SIZE = 2000000000;  // Size of input data
const int INTERVAL_SIZE = 4;  // Size of histogram intervals
const int HISTO_SIZE = 7;  // Size of histogram = ceil(26/INTERVAL_SIZE);

const unsigned int MAX_NUM = 26;  // a~z
const int MAX_ITER = 1;

unsigned char input[INPUT_SIZE];
unsigned int cpuOut[HISTO_SIZE];
unsigned int gpuOut_s1[HISTO_SIZE];
unsigned int gpuOut_s2[HISTO_SIZE];
unsigned int gpuOut_shared[HISTO_SIZE];
unsigned int gpuOut_aggr[HISTO_SIZE];


void generateRandomString(unsigned char *input, const unsigned int size){
	int numOfChars = size-1, i;
	for(i = 0; i < numOfChars; i++){
		input[i] = (unsigned int)(float(rand())/float(RAND_MAX) * MAX_NUM) + 'a';  // a~z
	//	input[i] = 'a';
	}
	input[i] = '\0';  // The last character should be NULL
}

void printString(const unsigned char *input, const unsigned int size){
	printf("Print String \n -----------\n");

	printf("%s\n", input);  // Print the whole string
/*
	// Print characters one by one
	for(int i = 0; i < size; i++){
		printf("%c", input[i]);
	}
	printf("\n--------\n");
*/	
}

void printHistogram(const unsigned int *histo, const unsigned int size, const char *name){
	printf("Print Histogram [%s]\n -----------\n", name);

	for(int i = 0; i < size; i++){
		printf("%u\t", histo[i]);
	}
	printf("\n--------\n");
}


bool compareHistograms(const unsigned int *histoA, const unsigned int *histoB, const unsigned int size){
	bool ret = true;
	for(int i = 0; i < size; i++){
		if(histoA[i] != histoB[i]){
			ret = false;
			break;
		}
	}
	return ret;
}



// Histogram Computation Functions
// 1. CPU Code
void cpuHistoComput(const unsigned char *input, unsigned int *histo, const unsigned int length){
	for(int i = 0; i < length; i++){
		int alphabet_pos = input[i] - 'a';
		if(alphabet_pos >= 0 && alphabet_pos < 26) {  // a~z
			histo[alphabet_pos/INTERVAL_SIZE]++;
		}
	}
}


__global__
void gpuHistoComput_strat1(const unsigned char *input, unsigned int *histo, const unsigned int length){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;  // global thread index
//	unsigned int grid_size = blockDim.x * gridDim.x;  // the total number of threads for the kernel
	//int section_size = ceil(length/grid_size);  // the number of elements assigned to each thread
	unsigned int section_size = (length-1) / (blockDim.x * gridDim.x) + 1;  // == ceil(length/grid_size)-> cannot be used in kernel
	int start = tid * section_size;  // index of the first element of the current section

	// Each thread handles consecutive elements of the section
	for(int k=0; k < section_size; k++){
		if((start+k) < length) {  // Check if the current index is out of bounds (of input data)
			int alphabet_pos = input[start+k] - 'a';  // what histogram interval index that the current char belongs to
			if(alphabet_pos >= 0 && alphabet_pos < 26){  // if the current char is an alphabet
				atomicAdd(&histo[alphabet_pos/INTERVAL_SIZE], 1);  // increase the interval value by one
			}
		}
	}	
}


__global__
void gpuHistoComput_strat2(const unsigned char *input, unsigned int *histo, const unsigned int length){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int grid_size = blockDim.x * gridDim.x;  // the total number of threads

	// Each thread handles corresponding elements of every grid_size section
	for(unsigned int i=tid; i < length; i += grid_size) {
		int alphabet_pos = input[i] - 'a';
		if(alphabet_pos >= 0 && alphabet_pos < 26) {
			atomicAdd(&(histo[alphabet_pos/INTERVAL_SIZE]), 1);
		}
	}
}


__global__
void gpuHistoComput_strat2_shared(const unsigned char *input, unsigned int *histo, const unsigned int length, const int num_intervals){  // num_intervals는 ceil(INPUT_SIZE / INTERVAL_SIZE)로 7이다!
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int grid_size = blockDim.x * gridDim.x;

	// Private Histogram in the shared MEM for the current section
	extern __shared__ unsigned int histo_s[];  // size: number of intervals
	for(unsigned int binIdx = threadIdx.x; binIdx < num_intervals; binIdx += blockDim.x) {
		histo_s[binIdx] = 0u;  // Initialize all elements of the histogram to 0	
	}
	__syncthreads();  // Wait for all threads to initialize the private histogram

	// Generate the private histogram
	for(unsigned int i=tid; i < length; i += grid_size) {
		int alphabet_pos = input[i] - 'a';
		if(alphabet_pos >= 0 && alphabet_pos < 26) {
			atomicAdd(&(histo_s[alphabet_pos/INTERVAL_SIZE]), 1);
		}
	}
	__syncthreads();  // Wait for all the threads to generate the private histogram
	
	// Commit to the original histogram in the global MEM
	for(unsigned int binIdx=threadIdx.x; binIdx < num_intervals; binIdx += blockDim.x){  // binIdx += grid_size가 아니다! Each thd in the TB is distributing the total work
		atomicAdd(&(histo[binIdx]), histo_s[binIdx]);	
	}

}


__global__
void gpuHistoComput_strat2_aggr(const unsigned char *input, unsigned int *histo, const unsigned int length, const int num_intervals ){	
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int grid_size = blockDim.x * gridDim.x;

	// Private Histogram in the shared MEM for the current section
	extern __shared__ unsigned int histo_s[];  // size: number of intervals
	for(unsigned int binIdx = threadIdx.x; binIdx < num_intervals; binIdx += blockDim.x) {
		histo_s[binIdx] = 0u;  // Initialize all elements of the histogram to 0	
	}
	__syncthreads();  // Wait for all threads to initialize the private histogram


	// Aggregation Part:
	
	int prev_interval = -1;
	unsigned int accumulator = 0;

	// Generate the private histogram
	for(unsigned int i=tid; i < length; i += grid_size) {
		int alphabet_pos = input[i] - 'a';

		if(alphabet_pos >= 0 && alphabet_pos < 26) {
			unsigned int curr_interval = alphabet_pos/INTERVAL_SIZE;

			// Accumulate the interval value if the current char belongs to the same interval of the previous char!
			if(curr_interval != prev_interval){ // If the current char and the previous one belongs to different interval: end accumulation
				if(accumulator > 0) atomicAdd(&(histo_s[prev_interval]), accumulator);
				accumulator = 1;
				prev_interval = curr_interval;
			}
			else {  // If the current char and the previous one belongs to the same interval: accumulate by one
				accumulator++;
			}
//			printf("prev interval: %d, curr interval: %d, accumulator: %d\n", prev_interval, curr_interval, accumulator);

		}
	}
	// 책에 빠진 부분! :Add the last character's accumulator
	atomicAdd(&(histo_s[prev_interval]), accumulator);

	__syncthreads();  // Wait for all the threads to generate the private histogram
	
/* Print the private histogram
	for(int i=0; i<INTERVAL_SIZE; i++){
		printf("%u\t", histo_s[i]);
	}
	printf("\n");	
*/	
	
	// Commit to the original histogram in the global MEM
	for(unsigned int binIdx=threadIdx.x; binIdx < num_intervals; binIdx += blockDim.x){  // binIdx += grid_size가 아니다! Each thd in the TB is distributing the total work
		atomicAdd(&(histo[binIdx]), histo_s[binIdx]);	
	}

/* Print the original histogram
	for(int i=0; i<INTERVAL_SIZE; i++){
		printf("%u\t", histo[i]);
	}
	printf("\n");
*/

}



int main(){
	srand((unsigned int)time(NULL));

	// Generate random alphabet string
	generateRandomString(input, INPUT_SIZE);

	//MK: GPU Memory
	unsigned char *d_input;
	unsigned int *d_histo_s1, *d_histo_s2, *d_histo_shared, *d_histo_aggr;
	size_t inputSize = sizeof(char) * INPUT_SIZE;
	size_t histoSize = sizeof(unsigned int) * HISTO_SIZE;

	// Dynamic Allocations in GPU
	cudaError_t err = cudaMalloc((void **) &d_input, inputSize);
	checkCudaError(err);

	err = cudaMalloc((void **) &d_histo_s1, histoSize);
	checkCudaError(err);

	err = cudaMalloc((void **) &d_histo_s2, histoSize);
	checkCudaError(err);

	err = cudaMalloc((void **) &d_histo_shared, histoSize);
	checkCudaError(err);

	err = cudaMalloc((void **) &d_histo_aggr, histoSize);
	checkCudaError(err);

	// Copy input data from CPU MEM to CPU MEM (input -> d_input)
	err = cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	//MK: Thread Num
//	const int num_intervals = ceil(INPUT_SIZE / INTERVAL_SIZE);  // 7 -> 실수: INPUT이 아니라 알파벳 개수(26)이 돼야 한다..!
	const int num_intervals = HISTO_SIZE;  // ceil(26/INTERVAL_SIZE);  // The size of the histogram (# of intervals)

/* TB Size */
	const int tbSize = 256;  // # of threads in each TB

/* TB Num */
	// 1. 각 스레드가 담당할 요소의 개수를 미리 정할 수 없다 -> 남는 스레드는 뒤쪽 TB들에만 존재하게 해야 하므로
	//const int sectionSize = 10;  // # of elements each thread handles
	//const int tbNum = ceil(INPUT_SIZE / (tbSize * sectionSize));  // 1 thread handles 'sectionSize' elements
		// -> 각 스레드가 몇 개의 element를 담당하게 할건지 결정하는 방식은 말이 안된다..! TB size와 TB 개수를 정하고 나서야 section size가 결정되는 것!

	// 2. 하나의 스레드가 하나의 요소를 담당하게 하는 방식
	//const int tbNum = ceil(INPUT_SIZE / tbSize);  // 1 thread handles 1 element
	
	// 3. 하나의 스레드가 여러 요소들을 담당하게 하는 방식(몇 개의 요소를 담당하느냐에 따라 TB의 개수가 달라짐)
	const int tbNum = ceil(INPUT_SIZE / (1000 * tbSize));  // 대략 한 TB가1000개씩 처리하도록-> 실수: ceil(INPUT_SIZE/10)은 한 TB에 1 thread만 있음을 전제로 함

/* Thread Organization */	
// 실수: TB(스레드)의 개수를input size를 넘어가게 너무 많이 만들면 이런 에러가 뜬다: an illegal memory access was encountered in ~
	dim3 gridSize(tbNum, 1, 1);
	dim3 blockSize(tbSize, 1, 1);
	
	//MK: Time Measurement
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	mkClockMeasure *ckGpu_s1 = new mkClockMeasure("GPU CODE(s1)");
	ckGpu_s1->clockReset();
	
	mkClockMeasure *ckGpu_s2 = new mkClockMeasure("GPU CODE(s2)");
	ckGpu_s2->clockReset();
	
	mkClockMeasure *ckGpu_shared = new mkClockMeasure("GPU CODE(shared)");
	ckGpu_shared->clockReset();
	
	mkClockMeasure *ckGpu_aggr = new mkClockMeasure("GPU CODE(aggr)");
	ckGpu_aggr->clockReset();

	// Execute the function(s)
	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuHistoComput(input, cpuOut, INPUT_SIZE);
		ckCpu->clockPause();

		ckGpu_s1->clockResume();
		gpuHistoComput_strat1<<<gridSize, blockSize>>>(d_input, d_histo_s1, INPUT_SIZE);
		err=cudaDeviceSynchronize();
		ckGpu_s1->clockPause();
		checkCudaError(err);

		ckGpu_s2->clockResume();
		gpuHistoComput_strat2<<<gridSize, blockSize>>>(d_input, d_histo_s2, INPUT_SIZE);
		err=cudaDeviceSynchronize();
		ckGpu_s2->clockPause();
		checkCudaError(err);


		ckGpu_shared->clockResume();
		gpuHistoComput_strat2_shared<<<gridSize, blockSize>>>(d_input, d_histo_shared, INPUT_SIZE, num_intervals);
		err=cudaDeviceSynchronize();
		ckGpu_shared->clockPause();
		checkCudaError(err);


		ckGpu_aggr->clockResume();
		gpuHistoComput_strat2_aggr<<<gridSize, blockSize>>>(d_input, d_histo_aggr, INPUT_SIZE, num_intervals);
		err=cudaDeviceSynchronize();
		ckGpu_aggr->clockPause();
		checkCudaError(err);
	}

	// Copy output data in the GPU MEM to CPU MEM (d_histo -> gpuOut)
	err = cudaMemcpy(gpuOut_s1, d_histo_s1, histoSize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpuOut_s2, d_histo_s2, histoSize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpuOut_shared, d_histo_shared, histoSize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpuOut_aggr, d_histo_aggr, histoSize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	// Release the memory used in the GPU
	cudaFree(d_input);
	cudaFree(d_histo_s1);
	cudaFree(d_histo_s2);
	cudaFree(d_histo_shared);
	cudaFree(d_histo_aggr);

	if(compareHistograms(cpuOut, gpuOut_s1, HISTO_SIZE) && compareHistograms(cpuOut, gpuOut_s2, HISTO_SIZE) && compareHistograms(cpuOut, gpuOut_shared, HISTO_SIZE) && compareHistograms(cpuOut, gpuOut_aggr, HISTO_SIZE)){
		ckCpu->clockPrint();
		ckGpu_s1->clockPrint();
		ckGpu_s2->clockPrint();
		ckGpu_shared->clockPrint();
		ckGpu_aggr->clockPrint();

	}else{
		printf("ERROR: Histograms are not same\n");
	}

//	printString(input, INPUT_SIZE);
	printHistogram(cpuOut, HISTO_SIZE, "CPU");
	printHistogram(gpuOut_s1, HISTO_SIZE, "GPU_Strategy 1");
	printHistogram(gpuOut_s2, HISTO_SIZE, "GPU_Strategy 2");
	printHistogram(gpuOut_shared, HISTO_SIZE, "GPU_Privatization");
	printHistogram(gpuOut_aggr, HISTO_SIZE, "GPU_Aggregation");

}
