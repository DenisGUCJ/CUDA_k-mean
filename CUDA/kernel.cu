#include <stdio.h>
#include <stdlib.h>
#include <string.h>     
#include <fcntl.h>
#include <unistd.h>    
#include <sys/types.h>  
#include <sys/stat.h> 
#include "file_io.c"
#include "kCuda.cu"
#include "kmeans.h"

static void usage(char *argv0) {
    char help[] =
        "Usage: %s -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -n num_clusters: number of clusters (K must > 1)\n";
     fprintf(stderr, help, argv0);
    exit(-1);
}

int main(int argc, char **argv) {

    cudaEvent_t event1,event2;
    int opt;
    int num_clusters = 0;
    int num_obj = 0;
    int num_coord;
    int *membership;
	char *filename = NULL;
    float **objects;
    float **clusters;
    float dt_ms; 
    
    while ((opt = getopt(argc,argv,"i:n:"))!= -1) {
	switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'n': num_clusters = atoi(optarg);
                      break;
            default: usage(argv[0]);
                      break;
		}
	}
	if (filename == 0 || num_clusters <= 1) usage(argv[0]);
	
	printf("...READING DATA FROM %s...\n",filename);
    objects = file_read(filename, &num_obj, &num_coord);
    
    if(objects == NULL) {
		printf("ERROR: 3D space was not found.\n");
		exit(1);
	}
	    if (num_obj < num_clusters) {
        printf("ERROR: number of clusters exeedes number of objects.\n");
        free(objects[0]);
        free(objects);
        exit(1);
	}
    
	/* allocate a 2D space for clusters[] (coordinates of cluster centers)
       this array should be the same across all processes                  */	
	clusters    = (float**) malloc(num_clusters * sizeof(float*));  assert(clusters != NULL); 
    clusters[0] = (float*)  malloc(num_clusters * num_coord * sizeof(float));   assert(clusters[0] != NULL);
    membership = (int*) malloc(num_obj * sizeof(int));  assert(membership != NULL);
	for (int i=1; i<num_clusters; i++)
		clusters[i] = clusters[i-1] + num_coord;
	
    //printf("...SELECTING %i INITIAL CLUSTERS...\n",num_clusters);
    
    printf("...COMPUTING...\n");            
    gpuErrchk(cudaEventCreate(&event1));
    gpuErrchk(cudaEventCreate(&event2));
    gpuErrchk(cudaEventRecord(event1));
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    clusters = cuda_kmeans(objects, num_obj, num_clusters, membership);
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    gpuErrchk(cudaEventRecord(event2));
    gpuErrchk(cudaEventSynchronize(event2));
    gpuErrchk(cudaDeviceSynchronize());

    cudaEventElapsedTime(&dt_ms, event1,event2);
    printf("...EXEQUTION TIME : %f  sec. ...\n", dt_ms/1000);

    file_write(filename, num_clusters, num_obj, num_coord, clusters, membership);

    free(objects[0]);
    free(objects);
    free(membership);
    free(clusters[0]);
    free(clusters);
	
	exit(0);
}