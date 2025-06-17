import os
import numpy as np
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.interpolate import splprep, splev

os.makedirs("output", exist_ok=True)

cyl_radius = 2.0
cyl_height = 10.0
pitch = 0.04

mesh_color = 'white'
mesh_opacity = 0.3


cylinder = trimesh.creation.cylinder(radius=cyl_radius, height=cyl_height, sections=64)
vox = cylinder.voxelized(pitch=pitch).fill()
points = vox.points
sparse_indices = vox.sparse_indices
mask = np.ones(points.shape[0], dtype=bool)


def random_ellipsoid(center, scale, rotation):
    p_shifted = points - center
    p_rotated = p_shifted @ rotation.T
    dist = ((p_rotated[:, 0] / scale[0]) ** 2 +
            (p_rotated[:, 1] / scale[1]) ** 2 +
            (p_rotated[:, 2] / scale[2]) ** 2)
    return dist <= 1.0

print("Generate irregular ellipsoids...")

num_ellipsoids = 4
for _ in range(num_ellipsoids):
    center = np.random.uniform([-cyl_radius * 0.5, -cyl_radius * 0.5, -cyl_height * 0.3],
                               [cyl_radius * 0.5, cyl_radius * 0.5, cyl_height * 0.3])
    scale = np.random.uniform([0.3, 0.4, 0.8], [0.8, 1.0, 2.0])
    angles = np.random.uniform(0, 2 * np.pi, size=3)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation = Rz @ Ry @ Rx

    ellipsoid_mask = random_ellipsoid(center, scale, rotation)
    mask[ellipsoid_mask] = False


def generate_smooth_curved_tube(radius_min, radius_max):
    num_points = 300
    z = np.linspace(-cyl_height / 2, cyl_height / 2, num_points)
    xy_noise = np.cumsum(np.random.uniform(-0.2, 0.2, size=(2, num_points)), axis=1)
    x_raw = xy_noise[0] * cyl_radius * 0.4
    y_raw = xy_noise[1] * cyl_radius * 0.4


    tck, u = splprep([x_raw, y_raw, z], s=0)
    x_smooth, y_smooth, z_smooth = splev(np.linspace(0, 1, num_points), tck)

    for i in range(num_points):
        radius = radius_min + (radius_max - radius_min) * (i / num_points)
        point = np.array([x_smooth[i], y_smooth[i], z_smooth[i]])
        dist = np.linalg.norm(points - point, axis=1)
        mask[dist <= radius] = False

print("Generate curved blood vessel paths...")

num_vessels = 5
for _ in range(num_vessels):
    generate_smooth_curved_tube(radius_min=0.06, radius_max=0.12)


new_sparse = sparse_indices[mask]
dense = np.zeros(vox.matrix.shape, dtype=bool)
dense[tuple(new_sparse.T)] = True

new_vox = trimesh.voxel.VoxelGrid(dense, transform=vox.transform)
mesh_with_voids = new_vox.marching_cubes

volume = dense.astype(np.float32)
volume[volume == 0] = 0.001
volume[volume == 1] = 0.05

np.save("phantom_dense.npy", volume)
print("Saved dense volume to phantom_dense.npy")
print("dense shape:", volume.shape)

nz, ny, nx = dense.shape
x_center = nx // 2
y_center = ny // 2
z_center = nz // 2

slice_xy = dense[z_center, :, :]
slice_xz = dense[:, y_center, :]
slice_yz = dense[:, :, x_center]

plt.figure(figsize=(10, 10))
plt.imshow(slice_xy, cmap='gray_r', origin='lower')
plt.title("Slice in XY plane (z center)")
plt.axis('off')
plt.savefig("output/slice_xy.png", dpi=600)
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(slice_xz, cmap='gray_r', origin='lower')
plt.title("Slice in XZ plane (y center)")
plt.axis('off')
plt.savefig("output/slice_xz.png", dpi=600)
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(slice_yz, cmap='gray_r', origin='lower')
plt.title("Slice in YZ plane (x center)")
plt.axis('off')
plt.savefig("output/slice_yz.png", dpi=600)
plt.close()

proj_xy_sum = np.sum(dense, axis=0)
proj_xz_sum = np.sum(dense, axis=1)
proj_yz_sum = np.sum(dense, axis=2)

plt.figure(figsize=(10, 10))
plt.imshow(proj_xy_sum, cmap='inferno', origin='lower')
plt.title("Sum Projection in XY plane (sum over z)")
plt.axis('off')
plt.savefig("output/proj_xy_sum.png", dpi=600)
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(proj_xz_sum, cmap='inferno', origin='lower')
plt.title("Sum Projection in XZ plane (sum over y)")
plt.axis('off')
plt.savefig("output/proj_xz_sum.png", dpi=600)
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(proj_yz_sum, cmap='inferno', origin='lower')
plt.title("Sum Projection in YZ plane (sum over x)")
plt.axis('off')
plt.savefig("output/proj_yz_sum.png", dpi=600)
plt.close()


faces_np = mesh_with_voids.faces
faces_pv = np.c_[np.full(len(faces_np), 3, dtype=np.int64), faces_np].ravel()
mesh_pv = pv.PolyData(mesh_with_voids.vertices, faces_pv)

plotter = pv.Plotter(off_screen=False)
plotter.set_background('black')
light = pv.Light(position=(5, 5, 10), focal_point=(0, 0, 0), color='white')
light.intensity = 1.0
plotter.add_light(light)
plotter.add_mesh(
    mesh_pv,
    color=mesh_color,
    opacity=mesh_opacity,
    smooth_shading=True,
    specular=0.5,
    specular_power=15
)
plotter.camera_position = 'xy'
plotter.show(title="Phantom: Ellipsoids & Curved Vessels")

print("Completed.")
