# ✅ mesh_distance.cu 已完成！

## 🎉 已实现的功能

### CUDA Kernel (`computeMinDistanceKernel`)
- ✅ 每个线程处理模型1的一个顶点
- ✅ 计算与模型2所有顶点的距离
- ✅ 使用共享内存进行块级规约
- ✅ 找到每个块的最小距离

### Host 函数 (`computeDistanceGPU`)
- ✅ GPU 内存分配
- ✅ 主机到设备的数据传输
- ✅ Kernel 启动配置（256线程/块）
- ✅ 结果回传和最终规约
- ✅ 内存清理

---

## 🚀 接下来需要做的事

### 步骤 1: 修改 CMakeLists.txt

需要添加两处修改：

#### 修改 1: 添加 CUDA 源文件（第14-18行附近）

**找到这部分**：
```cmake
set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/obj-viewer.cpp
    ${CMAKE_SOURCE_DIR}/src/cmodel.cpp
    ${CMAKE_SOURCE_DIR}/src/crigid.cpp
```

**改为**：
```cmake
set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/obj-viewer.cpp
    ${CMAKE_SOURCE_DIR}/src/cmodel.cpp
    ${CMAKE_SOURCE_DIR}/src/crigid.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh_distance.cu
```

#### 修改 2: 添加 CUDA 属性（第30行之后）

**在 `add_executable(meshDistCPU ${SRC_FILES})` 之后添加**：
```cmake
add_executable(meshDistCPU ${SRC_FILES})

# 配置 CUDA 属性
set_target_properties(meshDistCPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
)
```

> **注意**：根据你的 GPU 调整 `CUDA_ARCHITECTURES`
> - RTX 30系列：86
> - RTX 20系列：75
> - V100/A100：70/80
> 运行 `nvidia-smi` 查看你的 GPU 型号

---

### 步骤 2: 修改 src/cmodel.cpp

#### 修改 1: 添加 GPU 函数声明（文件顶部，includes 之后）

```cpp
// 在 #include 之后添加
#ifdef USE_CUDA
extern "C" float computeDistanceGPU(
    const vec3f* h_vtxs1, int num_vtx1,
    const vec3f* h_vtxs2, int num_vtx2,
    const transf& trf,
    int& min_i, int& min_j
);
#endif
```

#### 修改 2: 修改 check() 函数（第475行附近）

**找到**：
```cpp
REAL check(kmesh* m1, kmesh* m2, const transf& trfA, const transf& trfB, std::vector<id_pair>& pairs)
{
    const transf trfA2B = trfB.inverse() * trfA;
    return	m1->distNaive(m2, trfA2B, pairs);
}
```

**改为**：
```cpp
REAL check(kmesh* m1, kmesh* m2, const transf& trfA, const transf& trfB, std::vector<id_pair>& pairs)
{
    const transf trfA2B = trfB.inverse() * trfA;
    
#ifdef USE_CUDA
    // GPU 版本
    printf("=== Using GPU version ===\n");
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
    // CPU 版本
    printf("=== Using CPU version ===\n");
    return m1->distNaive(m2, trfA2B, pairs);
#endif
}
```

---

### 步骤 3: 编译

```bash
cd /home/jiaxingliu/workspace/gpuhw/build
rm -rf *
cmake ..
make -j$(nproc)
```

**预期输出**：
```
...
[ 27%] Building CUDA object CMakeFiles/meshDistCPU.dir/src/mesh_distance.cu.o
...
[100%] Built target meshDistCPU
```

---

### 步骤 4: 测试

```bash
# 小模型测试
./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj

# 大模型测试
./bin/meshDistCPU --headless data/buddha.obj data/wheeler.obj
```

**预期输出**：
```
=== Headless Mode (No GUI) ===
Model 1: data/my-bunny.obj
Model 2: data/alien-animal.obj
Loading models...
Computing minimum distance...
=== Using GPU version ===
GPU Configuration: 137 blocks x 256 threads = 35072 total threads
Processing: 34834 vertices from model 1, 23385 vertices from model 2
GPU Result: min_dist_sq = 54.582797, vertex pair = (21886, 20194)
MinDistance = 7.387997 (1 pairs) at 0.01234 s
(21886, 20194): (0.771866, 0.816832, 0.578291) - (8.148194, 1.023990, 0.218609) = 7.387997
=== Done ===
```

---

## ✅ 验证清单

检查以下几点确保实现正确：

- [ ] **编译成功**：无错误和警告
- [ ] **结果一致**：GPU 和 CPU 版本输出相同的最短距离
- [ ] **顶点对正确**：(21886, 20194) 对于 bunny+alien
- [ ] **性能提升**：GPU 版本明显快于 CPU（目标 >20x）

---

## 🐛 常见问题

### 问题 1: 编译错误 "undefined reference to computeDistanceGPU"

**原因**：没有正确链接 CUDA 文件

**解决**：检查 `CMakeLists.txt` 是否添加了 `mesh_distance.cu`

### 问题 2: 结果不一致

**原因**：浮点精度或算法错误

**解决**：
1. 检查 kernel 中的距离计算
2. 确认使用 `squareLength()` 而不是 `length()`
3. 验证变换矩阵应用正确

### 问题 3: CUDA 错误

**解决**：查看错误信息
```cpp
cudaError_t err = cudaGetLastError();
printf("CUDA Error: %s\n", cudaGetErrorString(err));
```

### 问题 4: 性能没有提升

**可能原因**：
- 数据传输开销太大（对小模型）
- 未启用 GPU 版本（检查 `USE_CUDA` 宏）
- GPU 利用率不足

**优化方向**：
1. 增加 `threadsPerBlock` (试试 512)
2. 优化内存访问模式
3. 使用 nvprof 分析

---

## 📊 性能对比

| 模型 | 顶点数 | CPU 时间 | GPU 预期时间 | 预期加速比 |
|------|--------|----------|------------|-----------|
| bunny + alien | 34k × 23k | 0.19s | 0.01s | 20x |
| buddha + wheeler | 427k × 72k | 7.65s | 0.15s | 50x |

---

## 🎯 下一步优化（可选）

如果想进一步提升性能：

1. **使用常量内存** 存储变换矩阵
2. **Warp 级规约** 替代块级规约
3. **原子操作** 替代两阶段规约
4. **共享内存优化** 存储模型2的顶点
5. **多流并发** 处理超大模型

---

开始修改文件吧！有问题随时查看这个文档 🚀

