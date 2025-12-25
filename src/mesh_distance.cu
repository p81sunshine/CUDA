// CUDA Implementation for Mesh Distance Calculation
// Optimized GPU path inspired by the CPU reference implementation.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <stdio.h>
#include <vector>

// Include headers
#include "../inc/vec3f.h"
#include "../inc/transf.h"
#include "../inc/pair.h"

namespace {

__constant__ transf c_trf;  // Cached transformation for fast reuse

struct DistPair {
    float dist;
    int i;
    int j;
};

__device__ __forceinline__ DistPair make_dist_pair(float dist, int i, int j) {
    DistPair p{dist, i, j};
    return p;
}

__device__ __forceinline__ DistPair pick_min(const DistPair& a, const DistPair& b) {
    return (b.dist < a.dist) ? b : a;
}

// Warp-level reduction for minimum distance and corresponding pair
__device__ __forceinline__ DistPair warp_reduce_min(DistPair val) {
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        DistPair other;
        other.dist = __shfl_down_sync(mask, val.dist, offset);
        other.i = __shfl_down_sync(mask, val.i, offset);
        other.j = __shfl_down_sync(mask, val.j, offset);
        val = pick_min(val, other);
    }
    return val;
}

}  // namespace

// CUDA kernel for minimum distance calculation
// Each block processes a tile of vertices from model 1 while collaboratively
// reusing tiles of model 2 vertices in shared memory to reduce global traffic.
__global__ void computeMinDistanceKernel(
    const vec3f* vtxs1,      // Vertices from model 1
    int num_vtx1,            // Number of vertices in model 1
    const vec3f* vtxs2,      // Vertices from model 2
    int num_vtx2,            // Number of vertices in model 2
    float* min_distances,    // Output: minimum distance per block
    int2* min_pairs          // Output: (i, j) pairs per block
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Shared memory for a tile of model 2 vertices
    extern __shared__ unsigned char shared_raw[];
    vec3f* shared_vtx2 = reinterpret_cast<vec3f*>(shared_raw);

    DistPair local = make_dist_pair(FLT_MAX, -1, -1);

    if (gid < num_vtx1) {
        const vec3f v1 = vtxs1[gid];
        const vec3f v1_transformed = c_trf.getVertex(v1);

        const int tiles = (num_vtx2 + blockDim.x - 1) / blockDim.x;
        for (int tile = 0; tile < tiles; ++tile) {
            const int j = tile * blockDim.x + tid;
            if (j < num_vtx2) {
                shared_vtx2[tid] = vtxs2[j];
            }
            __syncthreads();

            const int tile_size = min(blockDim.x, num_vtx2 - tile * blockDim.x);
            #pragma unroll 4
            for (int k = 0; k < tile_size; ++k) {
                const vec3f diff = v1_transformed - shared_vtx2[k];
                const float dist_sq = diff.squareLength();
                if (dist_sq < local.dist) {
                    local = make_dist_pair(dist_sq, gid, tile * blockDim.x + k);
                }
            }
            __syncthreads();
        }
    }

    // Warp-level reduction
    local = warp_reduce_min(local);

    // Shared scratch for warp minima
    __shared__ DistPair warp_mins[32];
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;

    if (lane == 0) {
        warp_mins[warp_id] = local;
    }
    __syncthreads();

    // First warp reduces warp-level minima
    if (warp_id == 0) {
        DistPair block_val = (lane < (blockDim.x + warpSize - 1) / warpSize)
                                 ? warp_mins[lane]
                                 : make_dist_pair(FLT_MAX, -1, -1);
        block_val = warp_reduce_min(block_val);

        if (lane == 0) {
            min_distances[blockIdx.x] = block_val.dist;
            min_pairs[blockIdx.x] = make_int2(block_val.i, block_val.j);
        }
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
    if (num_vtx1 == 0 || num_vtx2 == 0) {
        min_i = min_j = -1;
        return FLT_MAX;
    }

    auto check_cuda = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    // Step 1: Allocate device memory
    vec3f *d_vtxs1 = nullptr;
    vec3f *d_vtxs2 = nullptr;

    if (!check_cuda(cudaMalloc(&d_vtxs1, num_vtx1 * sizeof(vec3f)), "cudaMalloc d_vtxs1")) {
        return FLT_MAX;
    }
    if (!check_cuda(cudaMalloc(&d_vtxs2, num_vtx2 * sizeof(vec3f)), "cudaMalloc d_vtxs2")) {
        cudaFree(d_vtxs1);
        return FLT_MAX;
    }

    // Step 2: Copy data from host to device
    if (!check_cuda(cudaMemcpy(d_vtxs1, h_vtxs1, num_vtx1 * sizeof(vec3f), cudaMemcpyHostToDevice),
                    "cudaMemcpy vtxs1")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        return FLT_MAX;
    }
    if (!check_cuda(cudaMemcpy(d_vtxs2, h_vtxs2, num_vtx2 * sizeof(vec3f), cudaMemcpyHostToDevice),
                    "cudaMemcpy vtxs2")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        return FLT_MAX;
    }

    // Cache transformation in constant memory
    if (!check_cuda(cudaMemcpyToSymbol(c_trf, &trf, sizeof(transf)), "cudaMemcpyToSymbol c_trf")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        return FLT_MAX;
    }

    // Step 3: Configure kernel launch parameters
    const int threads_per_block = 256;
    const int num_blocks = (num_vtx1 + threads_per_block - 1) / threads_per_block;

    // Step 4: Allocate memory for per-block results
    float* d_min_distances = nullptr;
    int2* d_min_pairs = nullptr;

    if (!check_cuda(cudaMalloc(&d_min_distances, num_blocks * sizeof(float)), "cudaMalloc d_min_distances")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        return FLT_MAX;
    }
    if (!check_cuda(cudaMalloc(&d_min_pairs, num_blocks * sizeof(int2)), "cudaMalloc d_min_pairs")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        cudaFree(d_min_distances);
        return FLT_MAX;
    }

    // Step 5: Launch kernel with shared memory sized for one tile of model 2
    const size_t shared_mem_size = threads_per_block * sizeof(vec3f);

    computeMinDistanceKernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_vtxs1, num_vtx1,
        d_vtxs2, num_vtx2,
        d_min_distances, d_min_pairs
    );

    // Wait for kernel to complete
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        cudaFree(d_min_distances);
        cudaFree(d_min_pairs);
        return FLT_MAX;
    }

    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        cudaFree(d_min_distances);
        cudaFree(d_min_pairs);
        return FLT_MAX;
    }

    // Step 7: Copy results back to host
    std::vector<float> h_min_distances(num_blocks, FLT_MAX);
    std::vector<int2> h_min_pairs(num_blocks, make_int2(-1, -1));

    if (!check_cuda(cudaMemcpy(h_min_distances.data(), d_min_distances, num_blocks * sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy min_distances")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        cudaFree(d_min_distances);
        cudaFree(d_min_pairs);
        return FLT_MAX;
    }
    if (!check_cuda(cudaMemcpy(h_min_pairs.data(), d_min_pairs, num_blocks * sizeof(int2), cudaMemcpyDeviceToHost),
                    "cudaMemcpy min_pairs")) {
        cudaFree(d_vtxs1);
        cudaFree(d_vtxs2);
        cudaFree(d_min_distances);
        cudaFree(d_min_pairs);
        return FLT_MAX;
    }

    // Step 8: Final reduction on CPU (merge results from all blocks)
    float final_min_dist = FLT_MAX;
    min_i = -1;
    min_j = -1;

    for (int idx = 0; idx < num_blocks; ++idx) {
        if (h_min_distances[idx] < final_min_dist) {
            final_min_dist = h_min_distances[idx];
            min_i = h_min_pairs[idx].x;
            min_j = h_min_pairs[idx].y;
        }
    }

    // Step 9: Cleanup
    cudaFree(d_vtxs1);
    cudaFree(d_vtxs2);
    cudaFree(d_min_distances);
    cudaFree(d_min_pairs);

    return final_min_dist;
}

