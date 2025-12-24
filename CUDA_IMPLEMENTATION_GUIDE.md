# CUDA Implementation Guide - Mesh Distance Calculation

## 📝 Assignment Overview
**Goal**: Implement GPU version using CUDA to achieve optimal performance (results must match CPU version)

**Current Status**: 
- ✅ CPU version with OpenMP (0.19s for small models, 7.65s for large models)
- ✅ CUDA environment configured
- ❌ GPU kernel implementation needed

---

## 🎯 Implementation Steps

### Step 1: Create CUDA Kernel File

**File**: `src/mesh_distance.cu` (NEW FILE)

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../inc/vec3f.h"
#include "../inc/transf.h"
#include "../inc/pair.h"
#include <cfloat>

// CUDA kernel for distance calculation
__global__ void computeMinDistanceKernel(
    const vec3f* vtxs1,    // Model 1 vertices
    int num_vtx1,
    const vec3f* vtxs2,    // Model 2 vertices  
    int num_vtx2,
    float* min_distances,  // Per-block minimum distances
    int* min_pairs,        // Per-block minimum pairs (i, j)
    const transf trf       // Transformation matrix
) {
    // TODO: Implement kernel
    // Strategy: Each thread computes distances for a subset of vertex pairs
    // Use shared memory for block-level reduction
}

// Host function to call kernel
extern "C" 
float computeDistanceGPU(
    const vec3f* h_vtxs1, int num_vtx1,
    const vec3f* h_vtxs2, int num_vtx2,
    const transf& trf,
    int& min_i, int& min_j
);
```

### Step 2: Modify CMakeLists.txt

**File**: `CMakeLists.txt`

**Change line 14-18** from:
```cmake
set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/obj-viewer.cpp
    ${CMAKE_SOURCE_DIR}/src/cmodel.cpp
    ${CMAKE_SOURCE_DIR}/src/crigid.cpp
```

**To**:
```cmake
set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/obj-viewer.cpp
    ${CMAKE_SOURCE_DIR}/src/cmodel.cpp
    ${CMAKE_SOURCE_DIR}/src/crigid.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh_distance.cu  # NEW CUDA FILE
```

**And modify line 158-162**:
```cmake
# After add_executable, set CUDA properties
set_target_properties(meshDistCPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"  # Adjust for your GPU
)
```

### Step 3: Implement Kernel - Detailed Algorithm

#### 3.1 Parallel Strategy

**Grid Configuration**:
```cpp
dim3 threadsPerBlock(256);  // Typical: 256 or 512
dim3 numBlocks((num_vtx1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
```

**Thread Mapping**:
- Each thread handles one vertex from model 1
- Each thread computes distances to ALL vertices in model 2
- Use shared memory for block-level minimum reduction

#### 3.2 Kernel Implementation

```cuda
__global__ void computeMinDistanceKernel(
    const vec3f* vtxs1, int num_vtx1,
    const vec3f* vtxs2, int num_vtx2,
    float* min_distances,
    int* min_pairs,
    const transf trf
) {
    // Shared memory for block-level reduction
    __shared__ float shared_min_dist[256];
    __shared__ int shared_min_pair[256 * 2];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
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
    
    // Block-level reduction
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
    
    // Thread 0 writes block result
    if (tid == 0) {
        min_distances[blockIdx.x] = shared_min_dist[0];
        min_pairs[blockIdx.x * 2] = shared_min_pair[0];
        min_pairs[blockIdx.x * 2 + 1] = shared_min_pair[1];
    }
}
```

#### 3.3 Host Function

```cuda
extern "C" 
float computeDistanceGPU(
    const vec3f* h_vtxs1, int num_vtx1,
    const vec3f* h_vtxs2, int num_vtx2,
    const transf& trf,
    int& min_i, int& min_j
) {
    // Allocate device memory
    vec3f *d_vtxs1, *d_vtxs2;
    cudaMalloc(&d_vtxs1, num_vtx1 * sizeof(vec3f));
    cudaMalloc(&d_vtxs2, num_vtx2 * sizeof(vec3f));
    
    // Copy data to device
    cudaMemcpy(d_vtxs1, h_vtxs1, num_vtx1 * sizeof(vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vtxs2, h_vtxs2, num_vtx2 * sizeof(vec3f), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (num_vtx1 + threadsPerBlock - 1) / threadsPerBlock;
    
    float *d_min_distances;
    int *d_min_pairs;
    cudaMalloc(&d_min_distances, numBlocks * sizeof(float));
    cudaMalloc(&d_min_pairs, numBlocks * 2 * sizeof(int));
    
    computeMinDistanceKernel<<<numBlocks, threadsPerBlock>>>(
        d_vtxs1, num_vtx1,
        d_vtxs2, num_vtx2,
        d_min_distances, d_min_pairs,
        trf
    );
    
    // Copy results back
    float *h_min_distances = new float[numBlocks];
    int *h_min_pairs = new int[numBlocks * 2];
    
    cudaMemcpy(h_min_distances, d_min_distances, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_pairs, d_min_pairs, numBlocks * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Final CPU reduction
    float final_min_dist = FLT_MAX;
    for (int i = 0; i < numBlocks; i++) {
        if (h_min_distances[i] < final_min_dist) {
            final_min_dist = h_min_distances[i];
            min_i = h_min_pairs[i * 2];
            min_j = h_min_pairs[i * 2 + 1];
        }
    }
    
    // Cleanup
    delete[] h_min_distances;
    delete[] h_min_pairs;
    cudaFree(d_vtxs1);
    cudaFree(d_vtxs2);
    cudaFree(d_min_distances);
    cudaFree(d_min_pairs);
    
    return final_min_dist;
}
```

### Step 4: Modify cmodel.cpp

**File**: `src/cmodel.cpp`

Add GPU function declaration at top:
```cpp
extern "C" float computeDistanceGPU(
    const vec3f* h_vtxs1, int num_vtx1,
    const vec3f* h_vtxs2, int num_vtx2,
    const transf& trf,
    int& min_i, int& min_j
);
```

Modify `check()` function (around line 475):
```cpp
REAL check(kmesh* m1, kmesh* m2, const transf& trfA, const transf& trfB, std::vector<id_pair>& pairs)
{
    const transf trfA2B = trfB.inverse() * trfA;
    
#ifdef USE_CUDA
    // GPU version
    int min_i, min_j;
    float dist_sq = computeDistanceGPU(
        m1->getVtxs(), m1->getNbVertices(),
        m2->getVtxs(), m2->getNbVertices(),
        trfA2B,
        min_i, min_j
    );
    
    pairs.clear();
    pairs.push_back(id_pair(min_i, min_j, false));
    return dist_sq;
#else
    // CPU version
    return m1->distNaive(m2, trfA2B, pairs);
#endif
}
```

### Step 5: Compile and Test

```bash
cd build
rm -rf *
cmake ..
make -j$(nproc)

# Test
./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj
```

---

## 🚀 Optimization Tips

### Level 1: Basic (Get it working)
- ✅ Simple thread mapping (1 thread per vertex1)
- ✅ Basic reduction
- **Expected speedup**: 5-10x

### Level 2: Intermediate (Better performance)
- Use constant memory for transformation matrix
- Coalesced memory access
- Optimize block size (try 256, 512)
- **Expected speedup**: 20-50x

### Level 3: Advanced (Optimal performance)
- Shared memory for model2 vertices (if fits)
- Warp-level primitives (__shfl_down_sync)
- Atomic operations for global minimum
- Stream processing for very large models
- **Expected speedup**: 50-100x

### Level 4: Expert (Maximum performance)
- Tile-based processing
- BVH (Bounding Volume Hierarchy) acceleration
- Multiple CUDA streams
- Tensor cores (if applicable)
- **Expected speedup**: 100-500x

---

## 📊 Performance Metrics

**Test with**: `my-bunny.obj` (34,834 vertices) + `alien-animal.obj` (23,385 vertices)

| Version | Time | Speedup |
|---------|------|---------|
| CPU (single-thread) | ~1.0s | 1x |
| CPU (OpenMP) | ~0.19s | 5x |
| GPU (basic) | ~0.02s | 50x (target) |
| GPU (optimized) | ~0.005s | 200x (possible) |

---

## ⚠️ Common Pitfalls

1. **Incorrect memory transfer**: Make sure host-to-device copy is correct
2. **Synchronization bugs**: Always use `__syncthreads()` properly
3. **Shared memory size**: Don't exceed 48KB per block
4. **Race conditions**: Use atomics when needed
5. **Result mismatch**: Always verify output matches CPU version

---

## 📚 Helpful Resources

- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- Parallel Reduction: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- Your GPU architecture: Run `nvidia-smi` to check

---

## ✅ Checklist

- [ ] Create `src/mesh_distance.cu`
- [ ] Implement `computeMinDistanceKernel`
- [ ] Implement `computeDistanceGPU`
- [ ] Modify `CMakeLists.txt`
- [ ] Modify `src/cmodel.cpp`
- [ ] Compile successfully
- [ ] Test: Results match CPU version
- [ ] Measure: GPU faster than CPU
- [ ] Optimize: Try different configurations
- [ ] Document: Add performance analysis

---

Good luck with your assignment! 🚀

