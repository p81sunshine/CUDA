// CUDA Implementation for Mesh Distance Calculation
// Author: [Your Name]
// Date: 2025-11-05

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <stdio.h>

// Include headers
#include "../inc/vec3f.h"
#include "../inc/transf.h"
#include "../inc/pair.h"

// CUDA kernel for minimum distance calculation
// Each thread processes one vertex from model 1 and computes distances to all vertices in model 2
__global__ void computeMinDistanceKernel(
    const vec3f* vtxs1,      // Vertices from model 1
    int num_vtx1,            // Number of vertices in model 1
    const vec3f* vtxs2,      // Vertices from model 2
    int num_vtx2,            // Number of vertices in model 2
    float* min_distances,    // Output: minimum distance per block
    int* min_pairs,          // Output: (i, j) pairs per block
    transf trf               // Transformation matrix
) {
    // Step 1: Get thread and block indices
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 2: Allocate shared memory for block reduction
    extern __shared__ float shared_mem[];
    float* shared_min_dist = shared_mem;  // First part for distances
    int* shared_min_pair = (int*)&shared_mem[blockDim.x];  // Second part for pairs
    
    // Step 3: Each thread computes distances for its assigned vertex
    float local_min_dist = FLT_MAX;
    int local_min_i = -1;
    int local_min_j = -1;
    
    // Each thread processes one vertex from model 1
    if (gid < num_vtx1) {
        vec3f v1 = vtxs1[gid];
        vec3f v1_transformed = trf.getVertex(v1);
        
        // Compare with all vertices in model 2
        for (int j = 0; j < num_vtx2; j++) {
            vec3f v2 = vtxs2[j];
            vec3f diff = v1_transformed - v2;
            float dist_sq = diff.squareLength();
            
            if (dist_sq < local_min_dist) {
                local_min_dist = dist_sq;
                local_min_i = gid;
                local_min_j = j;
            }
        }
    }
    
    // Store in shared memory
    shared_min_dist[tid] = local_min_dist;
    shared_min_pair[tid * 2] = local_min_i;
    shared_min_pair[tid * 2 + 1] = local_min_j;
    __syncthreads();
    
    // Step 4: Perform block-level reduction to find minimum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_min_dist[tid + s] < shared_min_dist[tid]) {
                shared_min_dist[tid] = shared_min_dist[tid + s];
                shared_min_pair[tid * 2] = shared_min_pair[(tid + s) * 2];
                shared_min_pair[tid * 2 + 1] = shared_min_pair[(tid + s) * 2 + 1];
            }
        }
        __syncthreads();
    }
    
    // Step 5: Write block result to global memory
    if (tid == 0) {
        min_distances[blockIdx.x] = shared_min_dist[0];
        min_pairs[blockIdx.x * 2] = shared_min_pair[0];
        min_pairs[blockIdx.x * 2 + 1] = shared_min_pair[1];
    }
}

// Host function that manages GPU memory and kernel execution
extern "C" 
float computeDistanceGPU(
    const vec3f* h_vtxs1,    // Host pointer to model 1 vertices
    int num_vtx1,
    const vec3f* h_vtxs2,    // Host pointer to model 2 vertices
    int num_vtx2,
    const transf& trf,       // Transformation
    int& min_i,              // Output: vertex index in model 1
    int& min_j               // Output: vertex index in model 2
) {
    // Step 1: Allocate device memory
    vec3f *d_vtxs1 = nullptr;
    vec3f *d_vtxs2 = nullptr;
    
    cudaMalloc(&d_vtxs1, num_vtx1 * sizeof(vec3f));
    cudaMalloc(&d_vtxs2, num_vtx2 * sizeof(vec3f));
    
    // Step 2: Copy data from host to device
    cudaMemcpy(d_vtxs1, h_vtxs1, num_vtx1 * sizeof(vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vtxs2, h_vtxs2, num_vtx2 * sizeof(vec3f), cudaMemcpyHostToDevice);
    
    // Step 3: Configure kernel launch parameters
    int threadsPerBlock = 256;  // Typical choice: 256 or 512
    int numBlocks = (num_vtx1 + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("GPU Configuration: %d blocks x %d threads = %d total threads\n", 
           numBlocks, threadsPerBlock, numBlocks * threadsPerBlock);
    printf("Processing: %d vertices from model 1, %d vertices from model 2\n", 
           num_vtx1, num_vtx2);
    
    // Step 4: Allocate memory for per-block results
    float *d_min_distances = nullptr;
    int *d_min_pairs = nullptr;
    
    cudaMalloc(&d_min_distances, numBlocks * sizeof(float));
    cudaMalloc(&d_min_pairs, numBlocks * 2 * sizeof(int));
    
    // Step 5: Launch kernel
    // Calculate shared memory size: floats for distances + ints for pairs
    size_t shared_mem_size = threadsPerBlock * sizeof(float) + threadsPerBlock * 2 * sizeof(int);
    
    computeMinDistanceKernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(
        d_vtxs1, num_vtx1,
        d_vtxs2, num_vtx2,
        d_min_distances, d_min_pairs,
        trf
    );
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Step 6: Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return FLT_MAX;
    }
    
    // Step 7: Copy results back to host
    float *h_min_distances = new float[numBlocks];
    int *h_min_pairs = new int[numBlocks * 2];
    
    cudaMemcpy(h_min_distances, d_min_distances, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_pairs, d_min_pairs, numBlocks * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Step 8: Final reduction on CPU (merge results from all blocks)
    float final_min_dist = FLT_MAX;
    min_i = -1;
    min_j = -1;
    
    for (int i = 0; i < numBlocks; i++) {
        if (h_min_distances[i] < final_min_dist) {
            final_min_dist = h_min_distances[i];
            min_i = h_min_pairs[i * 2];
            min_j = h_min_pairs[i * 2 + 1];
        }
    }
    
    printf("GPU Result: min_dist_sq = %.6f, vertex pair = (%d, %d)\n", 
           final_min_dist, min_i, min_j);
    
    // Step 9: Cleanup
    delete[] h_min_distances;
    delete[] h_min_pairs;
    
    cudaFree(d_vtxs1);
    cudaFree(d_vtxs2);
    cudaFree(d_min_distances);
    cudaFree(d_min_pairs);
    
    return final_min_dist;
}

