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


const unsigned int VERTEX_NUM = 15000; // Size of input data
const unsigned int MAX_EDGE_NUM = VERTEX_NUM * (VERTEX_NUM-1) / 2;  // at most n*(n-1)/2 edges
const unsigned int MAX_NUM = 5;  // 0~4
const int MAX_ITER = 1;
const int BLOCK_SIZE = 256;
const int BLOCK_QUEUE_SIZE = 1000;

unsigned int A[VERTEX_NUM*VERTEX_NUM] = {0};  // Adjacency Matrix
unsigned int dest[MAX_EDGE_NUM];  // Destination vertex of each edge
unsigned int edges[VERTEX_NUM+1] = {0};  // Starting edge number of each vertex (Each Vertex 'i' has edges #edges[i] ~ #edges[i+1]-1
int label[VERTEX_NUM];  // Level of each vertex
unsigned int visited[VERTEX_NUM] = {0};  // checks whether the current vertex has already been visited or not

int cpuOut[VERTEX_NUM];
int gpuOut_simple[VERTEX_NUM];
//unsigned int cpuOut[HISTO_SIZE];
//unsigned int gpuOut_s1[HISTO_SIZE];


// Generate a Simple Graph via BFS
// 1. Generate a random Adjacency Matrix
void genrRndAdjacencyM(unsigned int *A, const int vertexNum) {
	for(int i=0; i<vertexNum; i++){
		for(int j=0; j<vertexNum; j++){
//			unsigned int p = 1;
			unsigned int p = (unsigned int)(float(rand()) / float(RAND_MAX) * MAX_NUM);  // 0~4
			if((p==1) && (i!=j) && (A[j*vertexNum+i]==0))
				A[i*vertexNum+j] = 1;  // 1/5의 확률&simple graph 고려해서 outgoing edge 생성
			else	
				A[i*vertexNum+j] = 0;  // 4/5의 확률로outgoing edge가 존재X
		}
	}
}

// 2. Generate a CSR(Sparse Matrix) Representation of the adjacency matrix
void genrCSR(unsigned int *A, const int vertexNum) {
	int edge_count=0;  // edge가 하나 있을 때마다count하는 누적값
	
	for(int i=0; i<vertexNum; i++) {
		for(int j=0; j<vertexNum; j++){
			if(A[i*vertexNum+j] == 1)
				dest[edge_count++] = j;  // Edge 'edge_count': Vertex i -> Vertex j
		}
		edges[i+1] = edge_count;
	}	
}
void printAdjacencyM(unsigned int *A, const int vertexNum) {
	int count = 0;
	
	printf("-----Print Adjacency Matrix Representation-----\n");
	for(int i=0; i<vertexNum; i++){
		for(int j=0; j<vertexNum; j++){
			if(A[i*vertexNum+j] == 1){	
				printf("[%d]V%d -> V%d\n", count, i, j);
				count++;
			}
		}
	}
}
void printCSR(unsigned int *dest, unsigned int *edges, const int vertexNum) {
	printf("-----Print CSR Representation-----\n");
	for(int i=0; i<vertexNum; i++) {
		for(int j=edges[i]; j<edges[i+1]; j++) {
			printf("[%d]V%d -> V%d\n", j, i, dest[j]);
		}
	}
}

void printLabel(const int *label, const unsigned int vertexNum) {
	printf("-----Print Label-----\n");
	for(int i=0; i<vertexNum; i++){
		printf("%d\t", label[i]);
	}
	printf("\n");
}

void printBFS(const int *label, const unsigned int vertexNum) {
	unsigned int Vertices[VERTEX_NUM][VERTEX_NUM] = {0};  // Maximum Depth: N, possible maximum number of vertices in one level: N-1
	unsigned int count[VERTEX_NUM] = {0};  // The number of vertices in each level
	unsigned int maxLevel=0;
	
	printf("-----Print BFS Result-----\n");
	for(int i=0; i<vertexNum; i++){
		int level = label[i];  // The level of the curent vertex
//		printf("Level %d\n", level);
		if(level >= 0) {  // 실수: label인 경우를 고려하지 않았다!! index가 -1이 되면 오류가 bus error가 난다!!
			Vertices[level][count[level]] = i;  // Add the current vertex into its level array.
//			printf("V%d stored inside Vertices[%d][%d]\n", Vertices[level][count[level]], level, count[level]);
			count[level]++; 

			// Find the maximum level value.
			if(level > maxLevel)
				maxLevel = level;
		}
	}
	for(int i=0; i <= maxLevel; i++){  // Print vertices level by level.
		//실수: i < maxLevel하면 마지막 레벨은 출력이 되지 않는다! 

		printf("[Level %d]\n", i);
		for(int j=0; j < count[i]; j++) {  // Print all the vertices of the current level.
			printf("%u\t", Vertices[i][j]);
		}
		printf("\n");
	}
}

bool compareBFS(const int *labelA, const int *labelB, const unsigned int vertexNum) {
	bool ret = true;
	for(int i = 0; i < vertexNum; i++){
		if(labelA[i] != labelB[i]){
			ret = false;
			break;
		}
	}
	return ret;
}

void insert_frontier(const unsigned int vertex, unsigned int *frontier, unsigned int *frontier_tail) {  // The address of the tail should be received.
	frontier[*frontier_tail] = vertex;  // Insert the vertex into the frontier queue.

//	printf("Vertex %u inserted in queue space #%d.\n", vertex, *frontier_tail);
	(*frontier_tail)++;  // The tail moves one space right.

//	printf("There exits %u number of vertices inside the queue.\n", *frontier_tail);
}


// Graph Search (BFS) Functions
// 1. CPU Code
void cpuBFS(const unsigned int source, const unsigned int *edges, const unsigned int *dest, int *label) {
	unsigned int frontier[VERTEX_NUM * 2];  // (= [2][VERTEX_NUM])includes current frontier queue and previous frontier queue
	unsigned int *c_frontier = &frontier[0];  // Current Frontier Queue
	unsigned int c_frontier_tail = 0;  // The tail is the index of the first blank element.
	unsigned int *p_frontier = &frontier[VERTEX_NUM];  // Previous Frontier Queue
	unsigned int p_frontier_tail = 0;  // Index of the last element
	unsigned int c_vertex;  // The index of the current vertex

	// Initialization
	insert_frontier(source, p_frontier, &p_frontier_tail);  // Insert the source vertex into the p_queue.
	label[source] = 0;  // The level of the source vertex is zero.

/*
	// 실수: Cannot initialize all the elements to -1 in declaration sentence!!
	for(int i=1; i<VERTEX_NUM; i++){
		label[i] = -1;
	}
*/

	// Do the BFS graph search 
	while(p_frontier_tail > 0) {  // while there exists one or more vertices to visit in the p_queue
		for (int f=0; f < p_frontier_tail; f++) {
			c_vertex = p_frontier[f];  // Visit each vertex stored in the p_queue (FIFO).
			
			for (int i = edges[c_vertex]; i < edges[c_vertex+1]; i++) {  // Find the index of each outgoing edge of the current vertex.
//				printf("V%d has edge #%d\n", c_vertex, i);
				unsigned int destVertex = dest[i];
//				printf("%d, label:%d", destVertex, label[destVertex]);
				if(label[destVertex] == -1) {  // If the destination vertex of such edge (neighbor vertex) has not been visited yet,
					insert_frontier(destVertex, c_frontier, &c_frontier_tail);  // Insert the current vertex into the c_queue
					label[destVertex] = label[c_vertex] + 1;  // Level of the current vertex is one greater than its predecessor
				}
			}
		}
		// Swap the address of the previous queue and the current queue (Change the role of the two queues)
		unsigned int *temp = c_frontier;
		c_frontier = p_frontier;
		p_frontier = temp;

		// Update the tails of c_queue and p_queue.
		p_frontier_tail = c_frontier_tail;  // There are 'c_frontier_tail' number of vertices in the p_queue
		c_frontier_tail = 0;  // The c_queue is renewed to store the newly visited vertices in the next iteration.
	}
}

// 2. GPU Code

__global__
void gpuBFS_initialization(unsigned int source, unsigned int *visited_d, int *label_d, unsigned int *p_frontier_d, unsigned int *c_frontier_tail_d, unsigned int *p_frontier_tail_d){
	*c_frontier_tail_d = 0;
	p_frontier_d[0] = source;  // Insert the source vertex inside the pQueue in device global MEM.
	*p_frontier_tail_d = 1;	

	label_d[source] = 0;
	visited_d[source] = 1;  // 실수: 책에 안 나와있는 부분!!!

	//printf("%d %d\n", label_d[source], label_d[1]);
}
__global__
void gpuBFS_renewTails(unsigned int *p_frontier_tail_d, unsigned int *c_frontier_tail_d){
	// Renew the value of the tail variables by using pointer parameters. 
	*p_frontier_tail_d = *c_frontier_tail_d;
	*c_frontier_tail_d = 0;
}

__global__
void gpuBFS_simple_kernel(unsigned int *p_frontier, unsigned int *p_frontier_tail, unsigned int *c_frontier, unsigned int *c_frontier_tail, const unsigned int *edges, const unsigned int *dest, int *label, unsigned int *visited){
	__shared__ unsigned int c_frontier_s[BLOCK_QUEUE_SIZE];
	__shared__ unsigned int c_frontier_tail_s, our_c_frontier_tail;
	
	if(threadIdx.x == 0)
		c_frontier_tail_s = 0;
	__syncthreads();

	// 1. Visit the neighbors of the previous vertices using shared (private) queue.
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < *p_frontier_tail) {
		const unsigned int my_vertex = p_frontier[tid];  // Vertex that the current thread should handle
		for(unsigned int i = edges[my_vertex]; i < edges[my_vertex+1]; i++) {
			const unsigned int was_visited = atomicExch(&(visited[dest[i]]), 1);  // Visit the current neighor vertex(change the value of 'visited' to 1) and get the old value.
			if(!was_visited) {  // If the current neighbor vertex hasn't already been visited
				label[dest[i]] = label[my_vertex] + 1;  // The level of the current neighbor vertex is one higher than its predecessor.
				const unsigned int my_tail = atomicAdd(&c_frontier_tail_s, 1);  // The tail moves one space right. Get the old tail value
				
				// Check if the cQueue in shared MEM is full
				if(my_tail < BLOCK_QUEUE_SIZE) {  // If not full,
					c_frontier_s[my_tail] = dest[i];  // Insert the current vertex into the shared (private) cQueue.
				}
				else {  // If full, insert it into the global cQueue.
					c_frontier_tail_s = BLOCK_QUEUE_SIZE;  // The tail of shared cQueue should come back to its original place (at the end of the queue).
					const unsigned int my_global_tail = atomicAdd(c_frontier_tail, 1);  // Tail of the global cQueue
					c_frontier[my_global_tail] = dest[i];
				}
//				printf("V%u -> V%u(label:%u): inserted in cQueue[%d]\n", my_vertex, c_frontier_s[my_tail], label[dest[i]], my_tail);
				
			}
		}
	}
	__syncthreads();
	
	// 2. Shared -> Global
	if(threadIdx.x == 0) {  // One thread in each TB updates the global tail value.
		our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);  // 실수: Be careful not to write 'c_frontier_tail' as '&c_frontier_tail'!!
	}
	__syncthreads();

	for(unsigned int i = threadIdx.x; i < c_frontier_tail_s; i += blockDim.x) {  // Each thread in TB inserts the vertices in shared queue into the global queue.
		c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
	
//		printf("%d\t", c_frontier[our_c_frontier_tail+i]);
	}
//	if(threadIdx.x == c_frontier_tail_s) printf("\n");

}

void gpuBFS_host(const unsigned int source, const unsigned int *edges_d, const unsigned int *dest_d, unsigned int *visited_d, int *label_d, unsigned int *frontier_d, unsigned int *c_frontier_tail_d, unsigned int *p_frontier_tail_d) {
	unsigned int *c_frontier_d = &frontier_d[0];
	unsigned int *p_frontier_d = &frontier_d[VERTEX_NUM];
	unsigned int p_frontier_tail = 0;

	// 1. Initialization
	// Initialization Kernel
	gpuBFS_initialization<<<1, 1 >>>(source, visited_d, label_d, p_frontier_d, c_frontier_tail_d, p_frontier_tail_d);

//	printf("Hello\n");

	// Initialization in host(CPU)

	// 실수: device MEM의 데이터는 host에서 접근 불가!!!!!
//	label_d[source] = 0;  // The level of the source vertex is 0.

	p_frontier_tail = 1;
	
	while(p_frontier_tail > 0) {
		int num_blocks = ceil(p_frontier_tail / float(BLOCK_SIZE));  // Thread Organization varies depending on the number of previous vertices to search
		
		// Launch a kernel function that searches the neighor vertices for every iteration
		gpuBFS_simple_kernel<<<num_blocks, BLOCK_SIZE>>>(p_frontier_d, p_frontier_tail_d, c_frontier_d, c_frontier_tail_d, edges_d, dest_d, label_d, visited_d);

		// Copy the value of c_frontier_tail_d in device MEM to p_frontier_tail in host MEM to check the condition of the while loop.
//		cudaError_t err = cudaMemcpy(p_frontier_tail, *c_frontier_tail_d, sizeof(unsigned int), cudaMemcpyHostToDevice);
// 실수: device에서 host로 copy하는 것이므로 cudaMemcpyDeviceToHost를 써야 한다!!!!!
//		cudaError_t err = cudaMemcpy(&p_frontier_tail, c_frontier_tail_d, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaError_t err = cudaMemcpy(&p_frontier_tail, c_frontier_tail_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		checkCudaError(err);

	// The visited vertices in current iteration (in cQueue) becomes the predecessor (in pQueue) in the next iteration.
		// In the next iteration, neighboring vertices of the previous vertices will be visited.
		// 1) Swap the roles of pQueue and cQueue
		unsigned int *temp = c_frontier_d;
		c_frontier_d = p_frontier_d;
		p_frontier_d = temp;

		// 2) Launch a kernel to renew the tails for next iteration.
		gpuBFS_renewTails<<<1, 1>>>(p_frontier_tail_d, c_frontier_tail_d);

	
	
	}


/*
	*/
}

/*
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
*/
/*
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
*/

int main(){
	srand((unsigned int)time(NULL));

	// Generate random simple graph
	genrRndAdjacencyM(A, VERTEX_NUM);
	genrCSR(A, VERTEX_NUM);
	
//	cpuBFS(0, edges, dest, label);
//	printBFS(label, VERTEX_NUM);


	// Initialize all the elements of label to -1
	// 실수: Cannot initialize all the elements to -1 in declaration sentence!!
	for(int i=0; i<VERTEX_NUM; i++){
		label[i] = -1;
	}

	//MK: GPU Memory
	unsigned int *edges_d, *dest_d, *visited_d;
	int *label_d_simple;

	// Allocate frontier queues in device global memory.
	unsigned int *frontier_d_simple;
	unsigned int *c_frontier_tail_d_simple, *p_frontier_tail_d_simple;  // The variable used in kernel function -> should be declared in device global MEM

	size_t vertexNum = sizeof(unsigned int) * VERTEX_NUM;
	size_t edgesSize = sizeof(unsigned int) * (VERTEX_NUM+1);
	size_t maxEdgeNum = sizeof(unsigned int) * MAX_EDGE_NUM;
	size_t intVar = sizeof(unsigned int);

	// Dynamic Allocations in GPU
	cudaError_t err = cudaMalloc((void **) &edges_d, edgesSize);
	checkCudaError(err);

	err = cudaMalloc((void **) &dest_d, maxEdgeNum);
	checkCudaError(err);

	err = cudaMalloc((void **) &visited_d, vertexNum);
	checkCudaError(err);

	err = cudaMalloc((void **) &label_d_simple, vertexNum);
	checkCudaError(err);

	// Allocate frontier queues and its tails in device global memory.
	err = cudaMalloc((void **) &frontier_d_simple, vertexNum);
	checkCudaError(err);

	err = cudaMalloc((void **) &c_frontier_tail_d_simple, intVar);
	checkCudaError(err);

	err = cudaMalloc((void **) &p_frontier_tail_d_simple, intVar);
	checkCudaError(err);


/* 실수: c_frontier_d와 p_frontier_d는 frontier_d를 가리키는 포인터이지, 배열명이 아니다!
	err = cudaMalloc((void **) &c_frontier_d_simple, sizeof(unsigned int));
	checkCudaError(err);

	err = cudaMalloc((void **) &p_frontier_d_simple, sizeof(unsigned int));
	checkCudaError(err);
*/

	// Copy input data from CPU MEM to GPU MEM (input -> d_input)
	err = cudaMemcpy(edges_d, edges, edgesSize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	err = cudaMemcpy(dest_d, dest, maxEdgeNum, cudaMemcpyHostToDevice);
	checkCudaError(err);

	err = cudaMemcpy(visited_d, visited, vertexNum, cudaMemcpyHostToDevice);
	checkCudaError(err);

	err = cudaMemcpy(label_d_simple, label, vertexNum, cudaMemcpyHostToDevice);
	checkCudaError(err);


	//MK: Thread Num

/* TB Size */
	const int tbSize = 256;  // # of threads in each TB

/* TB Num */
	// 1. Enough amount of threads
	const int tbNum = ceil(VERTEX_NUM / tbSize); 

//	const int tbNum = 1;


/* Thread Organization */	
// 실수: TB(스레드)의 개수를input size를 넘어가게 너무 많이 만들면 이런 에러가 뜬다: an illegal memory access was encountered in ~
	dim3 gridSize(tbNum, 1, 1);
	dim3 blockSize(tbSize, 1, 1);
	
	//MK: Time Measurement
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	ckCpu->clockReset();
	
	mkClockMeasure *ckGpu_simple = new mkClockMeasure("GPU CODE(simple)");
	ckGpu_simple->clockReset();
	
	mkClockMeasure *ckGpu_bQueue = new mkClockMeasure("GPU CODE(Block Queue)");
	ckGpu_bQueue->clockReset();
	
	mkClockMeasure *ckGpu_wQueue = new mkClockMeasure("GPU CODE(Warp Queue)");
	ckGpu_wQueue->clockReset();


	// Execute the function(s)
	for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuBFS(0, edges, dest, label);
		ckCpu->clockPause();

		ckGpu_simple->clockResume();
//		gpuBFS_host<<<gridSize, blockSize>>>(0, edges_d, dest_d, visited_d, label_d_simple, frontier_d_simple, c_frontier_tail_d_simple, p_frontier_tail_d_simple);
		gpuBFS_host(0, edges_d, dest_d, visited_d, label_d_simple, frontier_d_simple, c_frontier_tail_d_simple, p_frontier_tail_d_simple);
		err=cudaDeviceSynchronize();
		ckGpu_simple->clockPause();
		checkCudaError(err);
/*
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
*/
	}

	// Copy output data in the GPU MEM to CPU MEM (d_histo -> gpuOut)
// 실수: cudaMalloc이나 cudaMemcpy의 경우, 사이즈는 byte수이다!!!!!!!!!
//	err = cudaMemcpy(gpuOut_simple, label_d_simple, VERTEX_NUM, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(gpuOut_simple, label_d_simple, vertexNum, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	
/*
	err = cudaMemcpy(gpuOut_s2, d_histo_s2, histoSize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpuOut_shared, d_histo_shared, histoSize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpuOut_aggr, d_histo_aggr, histoSize, cudaMemcpyDeviceToHost);
	checkCudaError(err);
*/
	// Release the memory used in the GPU
	cudaFree(edges_d);
	cudaFree(dest_d);
	cudaFree(visited_d);
	cudaFree(label_d_simple);
	cudaFree(frontier_d_simple);
	cudaFree(c_frontier_tail_d_simple);
	cudaFree(p_frontier_tail_d_simple);

	if(compareBFS(label, gpuOut_simple, VERTEX_NUM)){
		ckCpu->clockPrint();
		ckGpu_simple->clockPrint();
	}else{
		printf("ERROR: The BFS results are not same\n");
	}


	
//	printAdjacencyM(A, VERTEX_NUM);

//	printBFS(label, VERTEX_NUM);

//	printBFS(gpuOut_simple, VERTEX_NUM);

//	printLabel(label, VERTEX_NUM);
//	printLabel(gpuOut_simple, VERTEX_NUM);
}
