#!/usr/bin/env python3
"""
OBJ File Visualization Tool - Render to Image
Render 3D models to PNG images in command line environment
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_obj(filename):
    """Load OBJ file"""
    vertices = []
    faces = []
    
    print(f"📂 Loading file: {filename}")
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face = []
                for i in range(1, len(parts)):
                    vertex_idx = int(parts[i].split('/')[0])
                    face.append(vertex_idx - 1)
                faces.append(face)
    
    vertices = np.array(vertices)
    print(f"✅ Loaded: {len(vertices):,} vertices, {len(faces):,} faces")
    return vertices, faces

def render_model(vertices, faces, output_file, title="3D Model", 
                 view_angle=(30, 45), figsize=(12, 10), 
                 show_wireframe=False, show_points=False,
                 face_color='lightblue', edge_color='black',
                 alpha=0.8, max_faces=None):
    """Render 3D model and save as image"""
    
    print(f"\n🎨 Rendering model...")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample faces if too many
    if max_faces and len(faces) > max_faces:
        print(f"⚠️  Too many faces ({len(faces):,}), sampling to {max_faces:,}")
        indices = np.random.choice(len(faces), max_faces, replace=False)
        faces_to_draw = [faces[i] for i in indices]
    else:
        faces_to_draw = faces
    
    # Draw faces
    if not show_wireframe:
        print(f"   Drawing {len(faces_to_draw):,} faces...")
        poly_collection = []
        for face in faces_to_draw:
            if len(face) >= 3:  # At least triangle
                poly = [vertices[i] for i in face]
                poly_collection.append(poly)
        
        poly3d = Poly3DCollection(poly_collection, 
                                  facecolors=face_color, 
                                  linewidths=0.1,
                                  edgecolors=edge_color,
                                  alpha=alpha)
        ax.add_collection3d(poly3d)
    
    # Draw wireframe
    if show_wireframe:
        print(f"   Drawing wireframe...")
        for face in faces_to_draw:
            if len(face) >= 3:
                face_verts = vertices[face]
                # Close the face
                face_verts = np.vstack([face_verts, face_verts[0]])
                ax.plot(face_verts[:, 0], face_verts[:, 1], face_verts[:, 2], 
                       'k-', linewidth=0.3, alpha=0.6)
    
    # Draw points
    if show_points:
        print(f"   Drawing vertices...")
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='red', marker='o', s=1, alpha=0.5)
    
    # Set axis limits
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=16, pad=20)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Save image
    print(f"💾 Saving image to: {output_file}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Done!")

def render_two_models(file1, file2, output_file, title="Model Comparison"):
    """Render two models to one image"""
    
    vertices1, faces1 = load_obj(file1)
    vertices2, faces2 = load_obj(file2)
    
    print(f"\n🎨 Rendering two models...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Model 1
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Limit face count
    max_faces = 50000
    faces1_draw = faces1 if len(faces1) <= max_faces else [faces1[i] for i in np.random.choice(len(faces1), max_faces, replace=False)]
    
    poly_collection1 = []
    for face in faces1_draw:
        if len(face) >= 3:
            poly = [vertices1[i] for i in face]
            poly_collection1.append(poly)
    
    poly3d1 = Poly3DCollection(poly_collection1, 
                              facecolors='lightblue', 
                              linewidths=0.1,
                              edgecolors='darkblue',
                              alpha=0.8)
    ax1.add_collection3d(poly3d1)
    
    # Set axis limits
    max_range1 = np.array([
        vertices1[:, 0].max() - vertices1[:, 0].min(),
        vertices1[:, 1].max() - vertices1[:, 1].min(),
        vertices1[:, 2].max() - vertices1[:, 2].min()
    ]).max() / 2.0
    
    mid_x1 = (vertices1[:, 0].max() + vertices1[:, 0].min()) * 0.5
    mid_y1 = (vertices1[:, 1].max() + vertices1[:, 1].min()) * 0.5
    mid_z1 = (vertices1[:, 2].max() + vertices1[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x1 - max_range1, mid_x1 + max_range1)
    ax1.set_ylim(mid_y1 - max_range1, mid_y1 + max_range1)
    ax1.set_zlim(mid_z1 - max_range1, mid_z1 + max_range1)
    ax1.set_title(f"Model 1\n{len(vertices1):,} vertices", fontsize=14)
    ax1.view_init(elev=30, azim=45)
    
    # Model 2
    ax2 = fig.add_subplot(122, projection='3d')
    
    faces2_draw = faces2 if len(faces2) <= max_faces else [faces2[i] for i in np.random.choice(len(faces2), max_faces, replace=False)]
    
    poly_collection2 = []
    for face in faces2_draw:
        if len(face) >= 3:
            poly = [vertices2[i] for i in face]
            poly_collection2.append(poly)
    
    poly3d2 = Poly3DCollection(poly_collection2, 
                              facecolors='lightcoral', 
                              linewidths=0.1,
                              edgecolors='darkred',
                              alpha=0.8)
    ax2.add_collection3d(poly3d2)
    
    # Set axis limits
    max_range2 = np.array([
        vertices2[:, 0].max() - vertices2[:, 0].min(),
        vertices2[:, 1].max() - vertices2[:, 1].min(),
        vertices2[:, 2].max() - vertices2[:, 2].min()
    ]).max() / 2.0
    
    mid_x2 = (vertices2[:, 0].max() + vertices2[:, 0].min()) * 0.5
    mid_y2 = (vertices2[:, 1].max() + vertices2[:, 1].min()) * 0.5
    mid_z2 = (vertices2[:, 2].max() + vertices2[:, 2].min()) * 0.5
    
    ax2.set_xlim(mid_x2 - max_range2, mid_x2 + max_range2)
    ax2.set_ylim(mid_y2 - max_range2, mid_y2 + max_range2)
    ax2.set_zlim(mid_z2 - max_range2, mid_z2 + max_range2)
    ax2.set_title(f"Model 2\n{len(vertices2):,} vertices", fontsize=14)
    ax2.view_init(elev=30, azim=45)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    
    print(f"💾 Saving image to: {output_file}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Done!")

def main():
    parser = argparse.ArgumentParser(
        description='Render OBJ files as images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render single model
  python3 render_obj.py data/my-bunny.obj -o bunny.png
  
  # Render as wireframe
  python3 render_obj.py data/my-bunny.obj -o bunny_wireframe.png --wireframe
  
  # Render two models for comparison
  python3 render_obj.py data/my-bunny.obj data/alien-animal.obj -o comparison.png
  
  # Custom view angle and color
  python3 render_obj.py data/my-bunny.obj -o bunny.png --angle 45 60 --color lightgreen
        """
    )
    
    parser.add_argument('files', nargs='+', help='OBJ file paths (1 or 2)')
    parser.add_argument('-o', '--output', default='output.png', 
                       help='Output image filename (default: output.png)')
    parser.add_argument('--angle', nargs=2, type=float, default=[30, 45],
                       metavar=('ELEV', 'AZIM'),
                       help='View angle: elevation and azimuth (default: 30 45)')
    parser.add_argument('--wireframe', action='store_true',
                       help='Wireframe mode')
    parser.add_argument('--points', action='store_true',
                       help='Show vertices')
    parser.add_argument('--color', default='lightblue',
                       help='Face color (default: lightblue)')
    parser.add_argument('--max-faces', type=int, default=50000,
                       help='Maximum faces to render (sample if exceeded, default: 50000)')
    parser.add_argument('--size', nargs=2, type=int, default=[12, 10],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Image size in inches (default: 12 10)')
    
    args = parser.parse_args()
    
    if len(args.files) == 1:
        # Single model
        vertices, faces = load_obj(args.files[0])
        render_model(vertices, faces, args.output,
                    title=args.files[0],
                    view_angle=args.angle,
                    figsize=tuple(args.size),
                    show_wireframe=args.wireframe,
                    show_points=args.points,
                    face_color=args.color,
                    max_faces=args.max_faces)
    
    elif len(args.files) == 2:
        # Two models comparison
        render_two_models(args.files[0], args.files[1], args.output,
                         title="Model Comparison")
    
    else:
        print("❌ Error: Please provide 1 or 2 OBJ files")
        sys.exit(1)
    
    print(f"\n🖼️  Image saved: {args.output}")
    print(f"   View with: xdg-open {args.output}")

if __name__ == '__main__':
    main()
