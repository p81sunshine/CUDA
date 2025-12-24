#!/bin/bash
# 性能对比测试脚本
# 对 data/ 目录下所有 .obj 文件进行两两组合测试
# 对比 CPU 和 GPU 版本的性能

# 注意：不使用 set -e，允许单个测试失败后继续

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         CPU vs GPU 性能对比测试                           ║"
echo "║         Mesh Distance Benchmark                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查可执行文件是否存在
CPU_BIN="$SCRIPT_DIR/build_cpu/bin/meshDistCPU"
GPU_BIN="$SCRIPT_DIR/build/bin/meshDistGPU"

if [ ! -f "$CPU_BIN" ]; then
    echo "❌ CPU 版本未找到: $CPU_BIN"
    echo "   请先运行: ./build_cpu.sh"
    exit 1
fi

if [ ! -f "$GPU_BIN" ]; then
    echo "❌ GPU 版本未找到: $GPU_BIN"
    echo "   请先运行: ./build_gpu.sh"
    exit 1
fi

echo "✅ 找到可执行文件:"
echo "   CPU: $CPU_BIN"
echo "   GPU: $GPU_BIN"
echo ""

# 找到所有 .obj 文件
DATA_DIR="$SCRIPT_DIR/data"
OBJ_FILES=($(find "$DATA_DIR" -name "*.obj" -type f | sort))

if [ ${#OBJ_FILES[@]} -eq 0 ]; then
    echo "❌ 未找到 .obj 文件"
    exit 1
fi

echo "📁 找到 ${#OBJ_FILES[@]} 个 OBJ 文件:"
for f in "${OBJ_FILES[@]}"; do
    basename "$f"
    size=$(du -h "$f" | cut -f1)
    echo "   └─ 大小: $size"
done
echo ""

# 创建结果目录
RESULT_DIR="$SCRIPT_DIR/benchmark_results"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="$RESULT_DIR/benchmark_${TIMESTAMP}.txt"
CSV_FILE="$RESULT_DIR/benchmark_${TIMESTAMP}.csv"

echo "📊 测试结果将保存到:"
echo "   文本: $RESULT_FILE"
echo "   CSV:  $CSV_FILE"
echo ""

# 初始化 CSV 文件
echo "模型1,模型2,顶点数1,顶点数2,总顶点对,CPU时间(s),GPU时间(s),加速比,最小距离,顶点对" > "$CSV_FILE"

# 初始化结果文件
{
    echo "═══════════════════════════════════════════════════════════"
    echo "  CPU vs GPU 性能对比测试报告"
    echo "  测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
} > "$RESULT_FILE"

# 统计变量
total_tests=0
cpu_faster=0
gpu_faster=0
total_cpu_time=0
total_gpu_time=0

echo "════════════════════════════════════════════════════════════"
echo "  开始测试 (${#OBJ_FILES[@]} 个文件，两两组合)"
echo "════════════════════════════════════════════════════════════"
echo ""

# 生成所有两两组合（不重复）
test_num=0
for ((i=0; i<${#OBJ_FILES[@]}; i++)); do
    for ((j=i+1; j<${#OBJ_FILES[@]}; j++)); do
        file1="${OBJ_FILES[$i]}"
        file2="${OBJ_FILES[$j]}"
        
        name1=$(basename "$file1" .obj)
        name2=$(basename "$file2" .obj)
        
        ((test_num++))
        total_tests=$test_num
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "测试 $test_num: $name1 vs $name2"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 获取相对路径
        rel_file1="data/$(basename "$file1")"
        rel_file2="data/$(basename "$file2")"
        
        # ============ CPU 版本测试 ============
        echo ""
        echo "🔹 CPU 版本测试中..."
        
        # 运行 CPU 版本并捕获输出
        cpu_start=$(date +%s.%N)
        cpu_output=$("$CPU_BIN" --headless "$rel_file1" "$rel_file2" 2>&1)
        cpu_end=$(date +%s.%N)
        cpu_time=$(echo "$cpu_end - $cpu_start" | bc)
        
        # 提取关键信息
        cpu_min_dist=$(echo "$cpu_output" | grep -oP "MinDistance = \K[0-9.]+" | head -1)
        cpu_pair=$(echo "$cpu_output" | grep -oP "\(\K[0-9]+, [0-9]+" | head -1)
        cpu_calc_time=$(echo "$cpu_output" | grep -oP "at \K[0-9.]+" | head -1)
        
        echo "   计算时间: ${cpu_calc_time}s (总时间: ${cpu_time}s)"
        echo "   最小距离: $cpu_min_dist"
        echo "   顶点对: ($cpu_pair)"
        
        # ============ GPU 版本测试 ============
        echo ""
        echo "🔹 GPU 版本测试中..."
        
        # 运行 GPU 版本并捕获输出
        gpu_start=$(date +%s.%N)
        gpu_output=$("$GPU_BIN" --headless "$rel_file1" "$rel_file2" 2>&1)
        gpu_end=$(date +%s.%N)
        gpu_time=$(echo "$gpu_end - $gpu_start" | bc)
        
        # 提取关键信息
        gpu_min_dist=$(echo "$gpu_output" | grep -oP "MinDistance = \K[0-9.]+" | head -1)
        gpu_pair=$(echo "$gpu_output" | grep -oP "\(\K[0-9]+, [0-9]+" | head -1)
        gpu_calc_time=$(echo "$gpu_output" | grep -oP "at \K[0-9.]+" | head -1)
        gpu_config=$(echo "$gpu_output" | grep "GPU Configuration" | head -1)
        gpu_vtx=$(echo "$gpu_output" | grep "Processing:" | grep -oP "[0-9]+ vertices" | head -2)
        
        echo "   配置: $gpu_config"
        echo "   计算时间: ${gpu_calc_time}s (总时间: ${gpu_time}s)"
        echo "   最小距离: $gpu_min_dist"
        echo "   顶点对: ($gpu_pair)"
        
        # ============ 对比结果 ============
        echo ""
        echo "📊 性能对比:"
        
        # 计算加速比（使用计算时间）
        if [ -n "$cpu_calc_time" ] && [ -n "$gpu_calc_time" ]; then
            speedup=$(echo "scale=2; $cpu_calc_time / $gpu_calc_time" | bc)
            echo "   CPU 计算时间: ${cpu_calc_time}s"
            echo "   GPU 计算时间: ${gpu_calc_time}s"
            echo "   加速比: ${speedup}x"
            
            # 判断谁更快
            result=$(echo "$speedup > 1.0" | bc)
            if [ "$result" -eq 1 ]; then
                echo "   🚀 GPU 更快!"
                ((gpu_faster++))
            else
                echo "   💻 CPU 更快!"
                ((cpu_faster++))
            fi
            
            # 累计时间
            total_cpu_time=$(echo "$total_cpu_time + $cpu_calc_time" | bc)
            total_gpu_time=$(echo "$total_gpu_time + $gpu_calc_time" | bc)
        fi
        
        # 验证结果一致性
        if [ "$cpu_min_dist" != "$gpu_min_dist" ]; then
            echo "   ⚠️  警告: 最小距离不一致!"
        else
            echo "   ✅ 结果验证: 一致"
        fi
        
        # 提取顶点数
        vtx1=$(echo "$gpu_output" | grep -oP "Processing: \K[0-9]+" | head -1)
        vtx2=$(echo "$gpu_output" | grep -oP "vertices from model 1, \K[0-9]+" | head -1)
        total_pairs=$(echo "$vtx1 * $vtx2" | bc)
        
        # 写入 CSV
        echo "$name1,$name2,$vtx1,$vtx2,$total_pairs,$cpu_calc_time,$gpu_calc_time,$speedup,$cpu_min_dist,$cpu_pair" >> "$CSV_FILE"
        
        # 写入文本结果
        {
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "测试 $test_num: $name1 vs $name2"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "顶点数: $vtx1 x $vtx2 = $total_pairs 对"
            echo ""
            echo "CPU 时间: ${cpu_calc_time}s"
            echo "GPU 时间: ${gpu_calc_time}s"
            echo "加速比:   ${speedup}x"
            echo ""
            echo "最小距离: $cpu_min_dist"
            echo "顶点对:   ($cpu_pair)"
            echo ""
        } >> "$RESULT_FILE"
        
        echo ""
    done
done

# ============ 生成总结报告 ============
echo "════════════════════════════════════════════════════════════"
echo "  测试完成！"
echo "════════════════════════════════════════════════════════════"
echo ""

overall_speedup=$(echo "scale=2; $total_cpu_time / $total_gpu_time" | bc)

echo "📈 总结:"
echo "   总测试数:     $total_tests"
echo "   GPU 更快:     $gpu_faster 次"
echo "   CPU 更快:     $cpu_faster 次"
echo "   CPU 总时间:   ${total_cpu_time}s"
echo "   GPU 总时间:   ${total_gpu_time}s"
echo "   平均加速比:   ${overall_speedup}x"
echo ""

# 写入总结到文件
{
    echo "═══════════════════════════════════════════════════════════"
    echo "  测试总结"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "总测试数:     $total_tests"
    echo "GPU 更快:     $gpu_faster 次"
    echo "CPU 更快:     $cpu_faster 次"
    echo "CPU 总时间:   ${total_cpu_time}s"
    echo "GPU 总时间:   ${total_gpu_time}s"
    echo "平均加速比:   ${overall_speedup}x"
    echo ""
    if [ "$gpu_faster" -gt "$cpu_faster" ]; then
        echo "🎉 GPU 版本在大多数情况下表现更好！"
    elif [ "$cpu_faster" -gt "$gpu_faster" ]; then
        echo "💻 CPU 版本在大多数情况下表现更好！"
    else
        echo "⚖️  CPU 和 GPU 表现相当。"
    fi
    echo ""
    echo "测试完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "$RESULT_FILE"

echo "✅ 详细结果已保存:"
echo "   文本报告: $RESULT_FILE"
echo "   CSV 数据: $CSV_FILE"
echo ""

# 显示 CSV 预览（使用 column 命令格式化，如果可用）
if command -v column &> /dev/null; then
    echo "📋 CSV 数据预览:"
    echo "─────────────────────────────────────────────────────────────"
    cat "$CSV_FILE" | column -t -s ','
    echo "─────────────────────────────────────────────────────────────"
else
    echo "💡 提示: 安装 column 命令可以查看格式化的表格"
    echo "   sudo apt-get install bsdmainutils"
fi

echo ""
echo "🎉 基准测试完成！"

