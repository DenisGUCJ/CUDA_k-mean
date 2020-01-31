#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "kmeans.h"

#define MAX_BLOCK 512
#define THREADS_PER_BLOCK 256
#define ERR 0.001
#define COORDS 3

typedef double Float;

__device__ inline static float euclid_distance(int numObjs,
                            int numClusters,
                            int objectId,
                            int clusterId,
                            float *objects,
                            float *clusters
){

    float ans = 0.0;
    for (int i = 0; i < COORDS; i++) {
        ans += (objects[3*objectId+i] - clusters[i + clusterId*3]) *
                (objects[3*objectId+i] - clusters[i + clusterId*3]);
    }

    return(ans);
}


__global__ static void find_nearest_cluster(int numObjs,
                    int numClusters,
                    float *objects,    
                    float *deviceClusters,
                    int *membership,
                    int *changedmembership,
                    float *temp_clusters,
                    int *temp_clusters_sizes
){

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    while (objectId < numObjs) {

        int index = 0;
        float dist, min_dist;
        min_dist = euclid_distance(numObjs, numClusters, objectId, 0, objects, deviceClusters);

        for (int i=1; i<numClusters; i++) {
            dist = euclid_distance(numObjs, numClusters, objectId, i, objects, deviceClusters);
            if (dist < min_dist) {
                min_dist = dist;
                index    = i;
            }
        }

        if(membership[objectId] != index){

            atomicAdd(changedmembership,1);
            membership[objectId] = index;          
        }

        atomicAdd(&temp_clusters_sizes[index],1);

        for(int j=0; j<COORDS; j++)
            atomicAdd(&temp_clusters[index*3+j], objects[objectId*3+j]);

        objectId += blockDim.x * gridDim.x;
    }
}

float** cuda_kmeans(float **objects,
    int numObjs,     
    int numClusters, 
    int *membership  
){

#pragma region declaration

    int loop = 0;
    int total_sum = 0;
    float delta;              
    int *newClusterSize; 
    float  **loopClusters;    
    float  **clusters;     
    float  **newClusters;    
    float **zero;;
    int *d_Membership;
    int *d_Changedmembership;
    float *d_Objects;
    float *d_Clusters;
    float *d_temp_clusters;
    int *d_temp_cluster_sizes;


#pragma endregion

#pragma region initialization

    gpuErrchk(cudaSetDevice(0));
 
    malloc2D(loopClusters, numClusters ,COORDS , float);
    malloc2D(zero, numClusters ,COORDS , float);

    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < COORDS; j++) {
            loopClusters[i][j] = objects[i][j];
            zero[i][j] = 0; 
        }
    }

    newClusterSize = (int*) malloc(numClusters* sizeof(int));   assert(newClusterSize != NULL);

    malloc2D(newClusters, numClusters,COORDS, float);
    memset(newClusters[0], 0, (COORDS * numClusters) * sizeof(float));
    memset(newClusterSize, 0, numClusters * sizeof(int));
    memset(membership,0,numObjs*sizeof(int)); 

    gpuErrchk(cudaMalloc(&d_Objects, numObjs*COORDS*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_Clusters, numClusters*COORDS*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_Membership, numObjs*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_Changedmembership, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_temp_clusters, numClusters*COORDS*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_temp_cluster_sizes, numClusters*sizeof(int)));

    gpuErrchk(cudaMemset(d_Changedmembership,0, sizeof(int)));
    gpuErrchk(cudaMemset(d_temp_cluster_sizes,0, numClusters*sizeof(int)));
    
    gpuErrchk(cudaMemcpy(d_Objects, objects[0], numObjs*COORDS*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Membership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));


#pragma endregion

#pragma region exeqution

    do {

        int tot_cor = 0;
        
        gpuErrchk(cudaMemcpy(d_Clusters, loopClusters[0], numClusters*COORDS*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_temp_clusters, zero[0], numClusters*COORDS*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(d_temp_cluster_sizes,0, numClusters*sizeof(int)));
        gpuErrchk(cudaMemset(d_Changedmembership,0, sizeof(int)));

        find_nearest_cluster<<<THREADS_PER_BLOCK, MAX_BLOCK >>>(numObjs, numClusters, d_Objects, d_Clusters, 
            d_Membership, d_Changedmembership, d_temp_clusters, d_temp_cluster_sizes);

        gpuErrchk(cudaMemcpy(&total_sum, d_Changedmembership, sizeof(int), cudaMemcpyDeviceToHost));
        
        delta = (float)total_sum/(float)numObjs;
    
        gpuErrchk(cudaMemcpy(membership, d_Membership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy(newClusterSize, d_temp_cluster_sizes, numClusters*sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(newClusters[0], d_temp_clusters, numClusters*COORDS*sizeof(float), cudaMemcpyDeviceToHost));
        
        /*set new cluster centers*/
        for (int i=0; i<numClusters; i++) {
            for (int j=0; j<COORDS; j++) {
                if (newClusterSize[i] > 0)
                {      
                    loopClusters[i][j] = (float)newClusters[i][j] / (float)newClusterSize[i];
                }
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            tot_cor += newClusterSize[i];
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        if(tot_cor != numObjs) {
            printf("Sum ERR \n");
            exit(-1);
        }

    } while (delta > ERR && loop++ < 500);

    /* allocate a 2D space for returning variable clusters[] (coordinates
    of cluster centers) */
    malloc2D(clusters, numClusters, COORDS, float);
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < COORDS; j++) {
            clusters[i][j] = loopClusters[i][j];
        }
    }

#pragma endregion

#pragma region free

    gpuErrchk(cudaFree(d_Membership));
    gpuErrchk(cudaFree(d_Changedmembership));
    gpuErrchk(cudaFree(d_Objects));
    gpuErrchk(cudaFree(d_Clusters));
    gpuErrchk(cudaFree(d_temp_clusters));
    gpuErrchk(cudaFree(d_temp_cluster_sizes));
    
    free(zero[0]);
    free(zero);
    free(loopClusters[0]);
    free(loopClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

#pragma endregion

    return clusters;
}
