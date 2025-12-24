#!/bin/bash
# CPU版本编译脚本
# 编译输出目录: build_cpu/

echo "================================"
echo "  编译 CPU 版本 (meshDistCPU)"
echo "================================"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 创建并进入 build_cpu 目录
mkdir -p build_cpu
cd build_cpu

echo ""
echo "[1/3] 配置 CMake (USE_GPU=OFF)..."
cmake -DUSE_GPU=OFF .. || {
    echo "❌ CMake 配置失败！"
    exit 1
}

echo ""
echo "[2/3] 编译中..."
make -j$(nproc) || {
    echo "❌ 编译失败！"
    exit 1
}

echo ""
echo "[3/3] 验证可执行文件..."
if [ -f "bin/meshDistCPU" ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "可执行文件位置:"
    ls -lh bin/meshDistCPU
    echo ""
    echo "运行方式:"
    echo "  cd $SCRIPT_DIR/build_cpu"
    echo "  ./bin/meshDistCPU --headless data/my-bunny.obj data/alien-animal.obj"
else
    echo "❌ 未找到可执行文件！"
    exit 1
fi

echo ""
echo "================================"
echo "  CPU 版本编译完成！"
echo "================================"

