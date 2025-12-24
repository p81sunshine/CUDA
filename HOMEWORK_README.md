# 🎯 CUDA 作业：网格距离计算 GPU 加速

## 📋 作业要求

**目标**：使用 CUDA 在 GPU 上实现最短距离计算，获得最优性能，结果必须与 CPU 版本一致。

**输入**：两个 3D 网格模型（.obj 文件）  
**输出**：两个模型之间的最短距离及对应的顶点对  
**性能目标**：GPU 版本应比 CPU 版本快 20-100 倍

---

## 🎓 核心算法

### CPU 版本（已实现）
```cpp
// 在 src/cmodel.cpp 的 kmesh::distNaive()
for (int j = 0; j < num_vtx1; j++) {
    vec3f v1_transformed = transform(vtx1[j]);
    for (int i = 0; i < num_vtx2; i++) {
        float dist = distance(v1_transformed, vtx2[i]);
        if (dist < min_dist) {
            min_dist = dist;
            min_pair = (j, i);
        }
    }
}
```

**复杂度**：O(N × M)，N 和 M 是两个模型的顶点数  
**CPU 时间**：34k × 23k 顶点 ≈ 0.19 秒（OpenMP）

### GPU 版本（待实现）

**并行策略**：
- 每个线程处理模型1的一个顶点
- 每个线程计算与模型2所有顶点的距离
- 使用共享内存进行块级规约
- 最后在 CPU 上合并各块结果

---

## 📂 需要修改的文件

### 1. ✅ **`src/mesh_distance.cu`** (新建，模板已创建)
- 实现 CUDA kernel：`computeMinDistanceKernel`
- 实现 host 函数：`computeDistanceGPU`

### 2. ⚠️ **`CMakeLists.txt`** (需要修改)

**第 14-18 行**，添加 CUDA 文件：
```cmake
set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/obj-viewer.cpp
    ${CMAKE_SOURCE_DIR}/src/cmodel.cpp
    ${CMAKE_SOURCE_DIR}/src/crigid.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh_distance.cu  # 添加这一行
    ...
)
```

**第 30-31 行**之后，添加 CUDA 属性：
```cmake
add_executable(meshDistCPU ${SRC_FILES})

# 添加 CUDA 配置
set_target_properties(meshDistCPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"  # 根据你的 GPU 调整
)
```

### 3. ⚠️ **`src/cmodel.cpp`** (需要修改)

**文件顶部**添加 GPU 函数声明：
```cpp
// Add after includes
#ifdef USE_CUDA
extern "C" float computeDistanceGPU(
    const vec3f* h_vtxs1, int num_vtx1,
    const vec3f* h_vtxs2, int num_vtx2,
    const transf& trf,
    int& min_i, int& min_j
);
#endif
```

**第 475-479 行**，修改 `check()` 函数：
```cpp
REAL check(kmesh* m1, kmesh* m2, const transf& trfA, const transf& trfB, std::vector<id_pair>& pairs)
{
    const transf trfA2B = trfB.inverse() * trfA;
    
#ifdef USE_CUDA
    // GPU 版本
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
    // CPU 版本（保持不变）
    return m1->distNaive(m2, trfA2B, pairs);
#endif
}
```

---

## 🚀 快速开始

### 步骤 1：查看模板代码
```bash
cd /home/jiaxingliu/workspace/gpuhw
cat src/mesh_distance.cu  # 查看 TODO 标记的地方
```

### 步骤 2：实现 CUDA kernel

参考 `CUDA_IMPLEMENTATION_GUIDE.md` 中的详细实现。

### 步骤 3：修改 CMakeLists.txt

按照上面的说明修改配置文件。

### 步骤 4：编译测试
```bash
cd build
rm -rf *
cmake ..
make -j$(nproc)

# 测试 GPU 版本
./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj
```

### 步骤 5：性能对比

记录不同配置下的性能：
```bash
# 小模型
./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj

# 大模型
./bin/meshDistCPU --headless data/buddha.obj data/wheeler.obj
```

---

## 📊 性能指标

| 模型 | 顶点数 | CPU 时间 | GPU 目标时间 | 加速比目标 |
|------|--------|----------|--------------|-----------|
| 兔子 vs 外星生物 | 34k × 23k | 0.19s | < 0.01s | > 20x |
| 佛像 vs 轮子 | 427k × 72k | 7.65s | < 0.2s | > 40x |

---

## 🔧 调试技巧

### 1. 检查 CUDA 错误
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

### 2. 验证结果
```bash
# GPU 和 CPU 结果应该一致
echo "Testing GPU version:"
./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj

# 如果结果不一致，需要调试
```

### 3. 性能分析
```bash
# 使用 CUDA profiler
nvprof ./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj
```

---

## 📚 关键知识点

### 1. 线程组织
- **Grid**: 整个任务
- **Block**: 一组线程（通常 256 或 512）
- **Thread**: 单个执行单元

### 2. 内存层次
- **Global Memory**: 最慢，但容量大
- **Shared Memory**: 快，块内共享（48KB 限制）
- **Register**: 最快，线程私有

### 3. 同步
- `__syncthreads()`: 块内同步
- 原子操作：用于全局变量的安全更新

### 4. 规约（Reduction）
- 关键技术：将多个值合并为一个
- 用于找最小值、求和等操作

---

## ⚠️ 常见错误

1. **忘记 cudaMalloc/cudaFree** → 内存泄漏
2. **未检查错误** → kernel 失败但不知道
3. **共享内存越界** → 未定义行为
4. **缺少 __syncthreads()** → 数据竞争
5. **结果不一致** → 浮点精度或算法错误

---

## 📝 提交清单

- [ ] `src/mesh_distance.cu` 完整实现
- [ ] `CMakeLists.txt` 正确配置
- [ ] `src/cmodel.cpp` 正确集成
- [ ] 编译成功，无警告
- [ ] 结果与 CPU 版本一致
- [ ] GPU 比 CPU 快（记录加速比）
- [ ] 代码有注释，说明算法
- [ ] 性能分析报告

---

## 🎯 优化建议

### 基础版（先让它工作）
- 简单的线程映射
- 基本的块级规约
- **目标加速比**: 20x

### 进阶版（提升性能）
- 优化内存访问模式
- 使用常量内存
- 调整块大小
- **目标加速比**: 50x

### 高级版（极致性能）
- Warp 级原语
- 原子操作优化
- 多流并发
- **目标加速比**: 100x+

---

## 📖 参考资料

- **详细实现指南**: `CUDA_IMPLEMENTATION_GUIDE.md`
- **CUDA 文档**: https://docs.nvidia.com/cuda/
- **并行规约**: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- **GPU 信息**: 运行 `nvidia-smi` 查看

---

## 💡 提示

1. 先实现**最简单的版本**，确保结果正确
2. 然后逐步**优化性能**
3. 每次修改后都**测试结果一致性**
4. 记录**每个优化的性能提升**
5. 写**清晰的注释**说明你的设计

---

祝你作业顺利！有问题随时参考 `CUDA_IMPLEMENTATION_GUIDE.md` 🚀

