#!/bin/bash
# GPU版本编译脚本
# 编译输出目录: build/

echo "================================"
echo "  编译 GPU 版本 (meshDistGPU)"
echo "================================"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置 CUDA 编译器路径
export CUDACXX=/home/jiaxingliu/miniconda3/bin/nvcc

# 检查 nvcc 是否存在
if [ ! -f "$CUDACXX" ]; then
    echo "❌ 错误: 未找到 nvcc 编译器"
    echo "   路径: $CUDACXX"
    echo "   请检查 CUDA 是否正确安装"
    exit 1
fi

echo "使用 CUDA 编译器: $CUDACXX"

# 创建并进入 build 目录
mkdir -p build
cd build

echo ""
echo "[1/3] 清理旧的构建文件..."
rm -rf *

echo ""
echo "[2/3] 配置 CMake (USE_GPU=ON)..."
cmake -DUSE_GPU=ON .. || {
    echo "❌ CMake 配置失败！"
    exit 1
}

echo ""
echo "[3/3] 编译中..."
make -j$(nproc) || {
    echo "❌ 编译失败！"
    exit 1
}

echo ""
echo "[4/4] 验证可执行文件..."
if [ -f "bin/meshDistGPU" ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "可执行文件位置:"
    ls -lh bin/meshDistGPU
    echo ""
    echo "运行方式:"
    echo "  cd $SCRIPT_DIR/build"
    echo "  ./bin/meshDistGPU --headless data/my-bunny.obj data/alien-animal.obj"
else
    echo "❌ 未找到可执行文件！"
    exit 1
fi

echo ""
echo "================================"
echo "  GPU 版本编译完成！"
echo "================================"

