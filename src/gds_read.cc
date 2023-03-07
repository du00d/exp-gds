/*
 * This program measures the total time
 * to perform NUM_READS reads over fixed array size of ARRAY_SIZE
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include "cufile.h"
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

	void *devPtr;

	int fd = open(argv[1], O_DIRECT);
	int NUM_READS = atoi(argv[2]);
	int ARRAY_SIZE = atoi(argv[3]);
	int filesize = GetFileSize(fd);


	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle;


	memset((void *)&cfr_descr, 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);

	cudaSetDevice(0); //use gpu:0
	cudaCheckError();

	cudaMalloc(&devPtr, filesize);
	cudaCheckError();

	status = cuFileBufRegister(devPtr, filesize, 0);

	vector<int> random_num; 
	//generate random read index
	for(int i = 0 ; i < NUM_READS ; i++){
		random_num.push_back(rand() % (filesize / (ARRAY_SIZE * sizeof(float))));
	}

	auto start = high_resolution_clock::now();

	for(int i =0 ; i < NUM_READS ; i++){
		cuFileRead(cfr_handle,
			   devPtr,
			   ARRAY_SIZE * sizeof(float), //bytes
			   random_num[i] * sizeof(float), //file offset
			   1 * sizeof(float)); //read to fixed location on gpu
	}
	cout << (float) duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << " msec"<< endl;

	close(fd);
}
