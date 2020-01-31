CUDA_NVCC_FLAGS:STRING=-arch=sm_13 

all: cuda random

cuda: kernel.cu
	nvcc kernel.cu -o cuda

random: RandomCord.c
	gcc -Wall -o random RandomCord.c

clean:
	rm cuda