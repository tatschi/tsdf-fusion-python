__global__ void find_voxels_for_points(
                                  float * vol_origin,
                                  float * other_params,
                                  float * point_cloud_x,
                                  float * point_cloud_y,
                                  int * voxels_x,
                                  int * voxels_y) {
  int gpu_loop_idx = (int) other_params[0];
  int max_threads_per_block = blockDim.x;
  int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
  int point_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;

  // get point values
  float point_x = point_cloud_x[point_idx];
  float point_y = point_cloud_y[point_idx];

  // point coordinates to voxel grid coordinates
  float voxel_size = other_params[1];
  voxels_x[point_idx] = (int) floorf((float)(point_x - vol_origin[0]) / (float) voxel_size);
  voxels_y[point_idx] = (int) floorf((float)(point_y - vol_origin[1]) / (float) voxel_size);
}


__global__ void integrate_point_cloud(
                          float * vol_origin,
                          float * other_params,
                          int * voxels_z,
                          float * points_z,
                          float * dists) {
  int gpu_loop_idx = (int) other_params[0];
  int max_threads_per_block = blockDim.x;
  int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
  int dists_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;

  float voxel_size = other_params[1];
  float trunc_margin = other_params[2];

  float voxel_z = voxels_z[dists_idx];
  float point_z = points_z[dists_idx];

  // Voxel grid z-coordinate to world coordinate
  float voxel_world_z = vol_origin[2]+voxel_z*voxel_size;

  float depth_diff = point_z - voxel_world_z;
  if (0 < depth_diff || depth_diff < -trunc_margin){
      return;
  }
  dists[dists_idx] = fmin(1.0f,depth_diff/trunc_margin);
}