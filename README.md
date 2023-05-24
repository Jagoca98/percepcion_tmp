# PointCloud ICP merger

### Description

Generate a merged point cloud of the point clouds found in the directory `assets/input`. It is important that they are sorted alphabetically for proper `ICP` functionality. When executed, the result is displayed, and the generated point cloud is saved as `assets/output/merged_pointcloud.pcd`.

### USAGE

To compile:

```bash
mkdir build
cd build
cmake ..
make
```

To execute (from build directory):

```bash
./PointCloud
```
