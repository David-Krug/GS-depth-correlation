import numpy as np
import sys
import os
from argparse import ArgumentParser

sys.path.append("../scene")
import colmap_loader as cl
import imageio.v3 as iio
import open3d as o3d
import glob

def  convert_depth_map_to_point_cloud(depth_map, intrinsic_matrix, depth_scale=1.0):
    """
    Convert a depth map to a point cloud.
    Parameters
    ----------
    depth_map : np.ndarray
        The depth map to convert.
    intrinsic_matrix : np.ndarray
        The intrinsic matrix of the camera.
    depth_scale : float
        The depth scale of the depth map.
    Returns
    -------
    np.ndarray
        The point cloud.
    """
    height, width = depth_map.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    depth_map = depth_map.flatten() * depth_scale
    xx = (xx - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    yy = (yy - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    point_cloud = np.stack([xx, yy, depth_map], axis=1)
    return point_cloud

if __name__ == "__main__":
  parser = ArgumentParser(description="Parameters for depth map to point cloud conversion.")
  parser.add_argument('--dataset_path', type=str, default="S:/Programme/GaussianSplatting/gaussian-splatting/training_data/playroom_depth")
  parser.add_argument('--filename', type=str)
  args = parser.parse_args()

  dataset_path = args.dataset_path
  intrinsic_path = os.path.join(dataset_path, "sparse", "0", "cameras.bin")
  os.makedirs(os.path.join(dataset_path, "pcs_from_adjusted_depth_maps"), exist_ok=True)

  intrinsic = cl.read_intrinsics_binary(intrinsic_path)[1]
  intrinsic_matrix = np.eye(4)
  intrinsic_matrix[0, 0] = intrinsic.params[0]
  intrinsic_matrix[1, 1] = intrinsic.params[1]
  intrinsic_matrix[0, 2] = intrinsic.params[2]
  intrinsic_matrix[1, 2] = intrinsic.params[3]
  print("Intrinsic matrix:")
  print(intrinsic_matrix)

  if args.filename:
    depth_map_path = os.path.join(args.dataset_path, "depth_adjusted", args.filename)
  else:
    depth_map_path = os.path.join(args.dataset_path, "depth_adjusted", "*.png")

  final_point_cloud = None
  for img_path in glob.glob(depth_map_path):
    filename = os.path.basename(img_path)
    depth_map = iio.imread(img_path)
    point_cloud = convert_depth_map_to_point_cloud(depth_map, intrinsic_matrix)

    if final_point_cloud is None:
      final_point_cloud = point_cloud
    else:
      final_point_cloud = np.concatenate([final_point_cloud, point_cloud], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcf = os.path.join(dataset_path, "pcs_from_adjusted_depth_maps", filename.split(".")[0] + ".ply")
    with open(pcf, 'w+') as file:
      o3d.io.write_point_cloud(pcf, pcd)
    print("Point cloud saved as", pcf)
  fpcd = o3d.geometry.PointCloud()
  fpcd.points = o3d.utility.Vector3dVector(final_point_cloud)
  if not args.filename:
    fpcf = os.path.join(dataset_path, "pcs_from_adjusted_depth_maps", "combined.ply")
    with open(fpcf, 'w+') as file:
      o3d.io.write_point_cloud(fpcf, fpcd)
    print("Point cloud saved as", fpcf)
  o3d.visualization.draw_geometries([fpcd])