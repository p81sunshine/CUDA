#!/usr/bin/env python3
"""
基准测试结果可视化脚本
读取 benchmark CSV 文件并生成性能对比图表
"""

import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 使用非交互式后端
matplotlib.use('Agg')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def find_latest_csv():
    """查找最新的 benchmark CSV 文件"""
    csv_files = glob.glob('benchmark_results/benchmark_*.csv')
    if not csv_files:
        return None
    return max(csv_files, key=os.path.getctime)

def load_data(csv_file):
    """加载 CSV 数据"""
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def create_visualizations(df, output_dir='benchmark_results'):
    """创建多个可视化图表"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试名称
    df['Test'] = df['模型1'] + ' vs\n' + df['模型2']
    
    # ========== 图表 1: CPU vs GPU 时间对比 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['CPU时间(s)'], width, label='CPU', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['GPU时间(s)'], width, label='GPU', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Test Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('CPU vs GPU Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Test'], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 在柱子上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}s',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/time_comparison.png")
    plt.close()
    
    # ========== 图表 2: 加速比 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#27ae60' if x > 1.0 else '#e67e22' for x in df['加速比']]
    bars = ax.bar(x, df['加速比'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加参考线
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Break-even (1x)')
    
    ax.set_xlabel('Test Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Speedup Over CPU', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Test'], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 在柱子上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        speedup = df['加速比'].iloc[i]
        label = f'{speedup:.2f}x'
        if speedup > 1.0:
            label += ' 🚀'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/speedup_comparison.png")
    plt.close()
    
    # ========== 图表 3: 按顶点对数量的加速比散点图 ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scatter = ax.scatter(df['总顶点对'], df['加速比'], 
                        s=200, c=df['加速比'], cmap='RdYlGn',
                        alpha=0.7, edgecolors='black', linewidth=2)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speedup', fontsize=11, fontweight='bold')
    
    # 添加参考线
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Break-even')
    
    ax.set_xlabel('Total Vertex Pairs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Speedup vs Problem Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加标注
    for i, row in df.iterrows():
        ax.annotate(f"{row['模型1'][:6]}\nvs\n{row['模型2'][:6]}",
                   xy=(row['总顶点对'], row['加速比']),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_vs_size.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/speedup_vs_size.png")
    plt.close()
    
    # ========== 图表 4: 性能效率对比（对数尺度）==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['CPU时间(s)'], width, label='CPU', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['GPU时间(s)'], width, label='GPU', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Test Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('CPU vs GPU Time Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Test'], fontsize=9)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_comparison_log.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/time_comparison_log.png")
    plt.close()

def generate_summary_report(df, output_file='benchmark_results/summary.txt'):
    """生成文字摘要报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Tests: {len(df)}\n")
        f.write(f"GPU Faster: {(df['加速比'] > 1.0).sum()} times\n")
        f.write(f"CPU Faster: {(df['加速比'] < 1.0).sum()} times\n\n")
        
        f.write(f"Average Speedup: {df['加速比'].mean():.2f}x\n")
        f.write(f"Median Speedup: {df['加速比'].median():.2f}x\n")
        f.write(f"Max Speedup: {df['加速比'].max():.2f}x ({df.loc[df['加速比'].idxmax(), '模型1']} vs {df.loc[df['加速比'].idxmax(), '模型2']})\n")
        f.write(f"Min Speedup: {df['加速比'].min():.2f}x ({df.loc[df['加速比'].idxmin(), '模型1']} vs {df.loc[df['加速比'].idxmin(), '模型2']})\n\n")
        
        f.write(f"Total CPU Time: {df['CPU时间(s)'].sum():.2f}s\n")
        f.write(f"Total GPU Time: {df['GPU时间(s)'].sum():.2f}s\n")
        f.write(f"Total Time Saved: {df['CPU时间(s)'].sum() - df['GPU时间(s)'].sum():.2f}s\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 60 + "\n\n")
        
        for i, row in df.iterrows():
            f.write(f"Test {i+1}: {row['模型1']} vs {row['模型2']}\n")
            f.write(f"  Vertex Pairs: {row['总顶点对']:,}\n")
            f.write(f"  CPU Time: {row['CPU时间(s)']:.4f}s\n")
            f.write(f"  GPU Time: {row['GPU时间(s)']:.4f}s\n")
            f.write(f"  Speedup: {row['加速比']:.2f}x\n")
            f.write(f"  Winner: {'GPU 🚀' if row['加速比'] > 1.0 else 'CPU 💻'}\n\n")
    
    print(f"✅ Saved: {output_file}")

def main():
    print("=" * 60)
    print("  Benchmark Visualization Tool")
    print("=" * 60)
    print()
    
    # 查找 CSV 文件
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = find_latest_csv()
    
    if not csv_file or not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Usage: python3 visualize_benchmark.py [csv_file]")
        sys.exit(1)
    
    print(f"📊 Reading data from: {csv_file}")
    
    # 加载数据
    df = load_data(csv_file)
    if df is None:
        print("❌ Failed to load data")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df)} test results")
    print()
    
    # 创建可视化
    print("🎨 Generating visualizations...")
    create_visualizations(df)
    print()
    
    # 生成摘要报告
    print("📝 Generating summary report...")
    generate_summary_report(df)
    print()
    
    print("=" * 60)
    print("  ✅ All visualizations complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  - benchmark_results/time_comparison.png")
    print("  - benchmark_results/speedup_comparison.png")
    print("  - benchmark_results/speedup_vs_size.png")
    print("  - benchmark_results/time_comparison_log.png")
    print("  - benchmark_results/summary.txt")
    print()

if __name__ == "__main__":
    main()


