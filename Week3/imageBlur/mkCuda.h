/*
 * =====================================================================================
 *
 *       Filename:  mkCuda.h
 *
 *    Description:  Help file for writing cuda code
 *
 *        Version:  1.0
 *        Created:  07/15/2021 10:35:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@ewha.ac.kr
 *   Organization:  EWHA Womans University 
 *
 * =====================================================================================
 */

#pragma once

#define checkCudaError(error) 			\
	if(error != cudaSuccess){ 				\
		printf("%s in %s at line %d\n", \
				cudaGetErrorString(error), 	\
				__FILE__ ,__LINE__); 				\
		exit(EXIT_FAILURE);							\
	}

