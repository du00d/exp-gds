/*
 * This program measures the total time
 * to perform NUM_READS reads over fixed array size of ARRAY_SIZE
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>


#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
}

using namespace std;
using namespace chrono;

size_t GetFileSize(int fd) {
	int ret;
	struct stat st;

	ret = fstat(fd, &st);
	return (ret == 0) ? st.st_size : -1;
}
int main(int argc, char **argv){

	cout << "MMAP EXPERIMENT" << endl;

	float *devPtr;
	int fd = open(argv[1], O_DIRECT);
	int NUM_READS = atoi(argv[2]);
	int ARRAY_SIZE = atoi(argv[3]);
	int filesize = GetFileSize(fd);
	float *file_mmap = static_cast<float *>(mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0));
	float temp[ARRAY_SIZE];
	
	cudaSetDevice(0); //use gpu:0
	cudaCheckError();

	cudaMalloc(&devPtr, filesize);
	cudaCheckError();

	vector<int> random_num; 
	//generate random read index
	for(int i = 0 ; i < NUM_READS ; i++){
		random_num.push_back(rand() % (filesize / (ARRAY_SIZE * sizeof(float))));
	}

	auto start = high_resolution_clock::now();

	for(int i =0 ; i < NUM_READS ; i++){
		memcpy(&temp, &file_mmap[random_num[i]], sizeof(float) * ARRAY_SIZE);
		cudaMemcpy(devPtr + 1 * sizeof(float), temp, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	}
	cout << (float) duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << " msec"<< endl;

	close(fd);
}
