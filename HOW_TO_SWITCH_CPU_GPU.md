# 🔄 如何在 CPU 和 GPU 版本之间切换

## 📝 概述

现在项目支持通过 CMake 选项编译不同的可执行文件：
- **meshDistGPU**：使用 CUDA 并行计算（GPU 版本）
- **meshDistCPU**：使用 OpenMP 多线程（CPU 版本）

**✅ 两个版本可以共存，无需重新编译！**

---

## 🚀 方法1：一键编译两个版本（推荐）⭐

### 编译 GPU 版本 (meshDistGPU)

```bash
cd /home/jiaxingliu/workspace/gpuhw/build
rm -rf *
export CUDACXX=/home/jiaxingliu/miniconda3/bin/nvcc
cmake -DUSE_GPU=ON ..
make -j$(nproc)
# 生成: build/bin/meshDistGPU
```

### 编译 CPU 版本 (meshDistCPU)

```bash
cd /home/jiaxingliu/workspace/gpuhw
mkdir -p build_cpu && cd build_cpu
cmake -DUSE_GPU=OFF ..
make -j$(nproc)
# 生成: build_cpu/bin/meshDistCPU

# 复制到主目录（可选）
cp bin/meshDistCPU ../build/bin/
```

### 现在你同时拥有两个版本！

```bash
ls -lh /home/jiaxingliu/workspace/gpuhw/build/bin/
# meshDistCPU (1.3M)  - CPU版本
# meshDistGPU (2.1M)  - GPU版本
```

---

## 📋 方法2：修改 CMakeLists.txt

找到第 12 行：

```cmake
option(USE_GPU "Use GPU (CUDA) version for distance calculation" ON)
```

**使用 CPU**：改为 `OFF`
```cmake
option(USE_GPU "Use GPU (CUDA) version for distance calculation" OFF)
```

**使用 GPU**：改为 `ON`
```cmake
option(USE_GPU "Use GPU (CUDA) version for distance calculation" ON)
```

然后重新编译：
```bash
cd build
rm -rf *
cmake ..
make -j$(nproc)
```

---

## ✅ 测试运行

### 测试小模型

```bash
cd /home/jiaxingliu/workspace/gpuhw/build

# CPU 版本
./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj
# 输出: MinDistance = 7.387997 (1 pairs) at 0.236 s

# GPU 版本
./bin/meshDistGPU --headless data/my-bunny.obj data/alien-animal.obj
# 输出: MinDistance = 7.387997 (1 pairs) at 0.254 s
```

### 测试大模型

```bash
# CPU 版本
./bin/meshDistCPU --headless data/buddha.obj data/wheeler.obj
# 输出: MinDistance = 1.642743 (1 pairs) at 7.64 s

# GPU 版本
./bin/meshDistGPU --headless data/buddha.obj data/wheeler.obj
# 输出: MinDistance = 1.642743 (1 pairs) at 0.32 s
```

**🚀 GPU加速：23.9倍！**

---

## 📊 性能对比

| 模型 | 顶点数 | CPU 时间 | GPU 时间 | GPU 加速比 |
|------|--------|----------|----------|-----------|
| bunny + alien | 34k × 23k | 0.19s | 0.32s | 0.6x |
| buddha + wheeler | 427k × 100k | 7.65s | 0.32s | 24x |

**说明**：
- 小模型：GPU数据传输开销大，CPU更快
- 大模型：GPU并行优势明显，大幅提速

---

## 🔍 如何确认当前版本

运行程序时会输出：

**CPU 版本**：
```
=== Using CPU version ===
```

**GPU 版本**：
```
=== Using GPU version ===
GPU Configuration: ...
```

或者查看编译时的输出：

**CPU 版本**：
```
-- CPU version enabled (USE_GPU=OFF)
```

**GPU 版本**：
```
-- GPU version enabled (USE_GPU=ON)
```

---

## 💡 快捷脚本

创建两个脚本方便切换：

### build_cpu.sh
```bash
#!/bin/bash
cd /home/jiaxingliu/workspace/gpuhw/build
rm -rf *
cmake -DUSE_GPU=OFF ..
make -j$(nproc)
echo "CPU version built successfully!"
```

### build_gpu.sh
```bash
#!/bin/bash
cd /home/jiaxingliu/workspace/gpuhw/build
rm -rf *
export CUDACXX=/home/jiaxingliu/miniconda3/bin/nvcc
cmake -DUSE_GPU=ON ..
make -j$(nproc)
echo "GPU version built successfully!"
```

使用方法：
```bash
chmod +x build_cpu.sh build_gpu.sh
./build_cpu.sh  # 编译CPU版本
./build_gpu.sh  # 编译GPU版本
```

---

## 🎯 常见问题

### Q: 为什么小模型GPU反而更慢？

**A:** 因为数据传输开销（主机↔设备）。GPU适合大规模并行计算。

**建议**：
- 小模型（< 10万顶点对）：用CPU
- 大模型（> 100万顶点对）：用GPU

### Q: GPU版本编译失败

**A:** 确保：
1. 设置了 CUDACXX 环境变量
2. 有可用的 NVIDIA GPU
3. 安装了 CUDA toolkit

### Q: 能否运行时动态切换？

**A:** 目前不行，需要重新编译。可以考虑编译两个不同的可执行文件。

---

## 📚 相关文件

- `src/mesh_distance.cu` - GPU kernel 实现
- `src/cmodel.cpp` - CPU/GPU 切换逻辑
- `CMakeLists.txt` - 编译配置

---

享受CPU和GPU的灵活切换！🚀

