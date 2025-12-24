#!/usr/bin/env python3
"""
OBJ文件查看器 - 命令行版本
用于查看和分析3D模型文件的基本信息
"""

import sys
import numpy as np
import argparse

def load_obj(filename):
    """加载OBJ文件，返回顶点和面数据"""
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # 顶点坐标 (v x y z)
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # 面 (f v1 v2 v3 或 f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
            elif parts[0] == 'f':
                face = []
                for i in range(1, len(parts)):
                    # 处理不同格式：v, v/vt, v/vt/vn, v//vn
                    vertex_idx = int(parts[i].split('/')[0])
                    face.append(vertex_idx - 1)  # OBJ索引从1开始
                faces.append(face)
    
    return np.array(vertices), faces

def analyze_obj(filename, show_stats=True, show_bbox=True, show_sample=False):
    """分析OBJ文件并显示统计信息"""
    print(f"📦 加载文件: {filename}")
    print("=" * 60)
    
    vertices, faces = load_obj(filename)
    
    # 基本统计
    if show_stats:
        print(f"\n📊 模型统计:")
        print(f"  顶点数量: {len(vertices):,}")
        print(f"  面数量:   {len(faces):,}")
        
        # 面的类型统计
        face_types = {}
        for face in faces:
            n = len(face)
            face_types[n] = face_types.get(n, 0) + 1
        
        print(f"\n  面类型:")
        for n_verts, count in sorted(face_types.items()):
            face_name = {3: "三角形", 4: "四边形"}.get(n_verts, f"{n_verts}边形")
            print(f"    {face_name}: {count:,}")
    
    # 边界框
    if show_bbox and len(vertices) > 0:
        min_coord = vertices.min(axis=0)
        max_coord = vertices.max(axis=0)
        center = (min_coord + max_coord) / 2
        size = max_coord - min_coord
        
        print(f"\n📐 边界框 (Bounding Box):")
        print(f"  最小坐标: ({min_coord[0]:.4f}, {min_coord[1]:.4f}, {min_coord[2]:.4f})")
        print(f"  最大坐标: ({max_coord[0]:.4f}, {max_coord[1]:.4f}, {max_coord[2]:.4f})")
        print(f"  中心位置: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"  尺寸大小: ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")
    
    # 显示样本顶点
    if show_sample and len(vertices) > 0:
        print(f"\n🔍 前5个顶点坐标:")
        for i in range(min(5, len(vertices))):
            v = vertices[i]
            print(f"  顶点 {i}: ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")
    
    return vertices, faces

def compute_distance(file1, file2):
    """计算两个模型之间的最短距离（简化版本）"""
    print("🔢 计算最短距离...")
    print("=" * 60)
    
    vertices1, _ = load_obj(file1)
    vertices2, _ = load_obj(file2)
    
    print(f"模型1: {len(vertices1):,} 顶点")
    print(f"模型2: {len(vertices2):,} 顶点")
    
    # 警告：对大模型会很慢
    if len(vertices1) * len(vertices2) > 1000000:
        print("\n⚠️  警告: 顶点数量较大，计算可能需要较长时间...")
        print("   建议使用编译后的C++版本: ./bin/meshDistCPU --headless")
        response = input("   是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    import time
    start = time.time()
    
    # 计算最短距离
    min_dist = float('inf')
    min_pair = (0, 0)
    
    for i, v1 in enumerate(vertices1):
        for j, v2 in enumerate(vertices2):
            dist = np.linalg.norm(v1 - v2)
            if dist < min_dist:
                min_dist = dist
                min_pair = (i, j)
        
        # 显示进度
        if (i + 1) % 1000 == 0:
            print(f"  处理进度: {i+1}/{len(vertices1)} ({100*(i+1)/len(vertices1):.1f}%)")
    
    elapsed = time.time() - start
    
    print(f"\n✅ 计算完成!")
    print(f"最短距离: {min_dist:.6f}")
    print(f"顶点对: ({min_pair[0]}, {min_pair[1]})")
    print(f"  模型1顶点 {min_pair[0]}: {vertices1[min_pair[0]]}")
    print(f"  模型2顶点 {min_pair[1]}: {vertices2[min_pair[1]]}")
    print(f"计算时间: {elapsed:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(
        description='OBJ文件查看和分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看单个模型
  python3 view_obj.py data/my-bunny.obj
  
  # 查看详细信息（包括样本顶点）
  python3 view_obj.py data/my-bunny.obj -s
  
  # 计算两个模型之间的距离（Python版，较慢）
  python3 view_obj.py data/my-bunny.obj data/alien-animal.obj -d
  
  # 比较两个模型的统计信息
  python3 view_obj.py data/my-bunny.obj data/buddha.obj
        """
    )
    
    parser.add_argument('files', nargs='+', help='OBJ文件路径（1个或2个）')
    parser.add_argument('-s', '--sample', action='store_true', 
                       help='显示样本顶点坐标')
    parser.add_argument('-d', '--distance', action='store_true',
                       help='计算两个模型之间的最短距离（需要2个文件）')
    parser.add_argument('--no-bbox', action='store_true',
                       help='不显示边界框信息')
    
    args = parser.parse_args()
    
    # 单个文件：显示信息
    if len(args.files) == 1:
        analyze_obj(args.files[0], 
                   show_bbox=not args.no_bbox,
                   show_sample=args.sample)
    
    # 两个文件
    elif len(args.files) == 2:
        if args.distance:
            # 计算距离
            compute_distance(args.files[0], args.files[1])
        else:
            # 显示两个模型的对比信息
            print("📦 模型 1")
            analyze_obj(args.files[0], 
                       show_bbox=not args.no_bbox,
                       show_sample=args.sample)
            
            print("\n" + "=" * 60)
            print("📦 模型 2")
            analyze_obj(args.files[1], 
                       show_bbox=not args.no_bbox,
                       show_sample=args.sample)
    else:
        print("❌ 错误: 请提供1个或2个OBJ文件")
        sys.exit(1)

if __name__ == '__main__':
    main()

