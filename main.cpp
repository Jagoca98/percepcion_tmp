#include <iostream>
#include <set>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/filesystem.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>


int main()
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds;
    boost::filesystem::path dir_root = boost::filesystem::current_path().parent_path() / "assets/";
    std::string directoryPath = dir_root.string() + "input/";
    boost::filesystem::path result_root = "output/merged_pointcloud.pcd";
    boost::filesystem::path result_root_joined = dir_root / result_root;

    if (!boost::filesystem::is_directory(directoryPath)) {
        std::cout << "Invalid directory path: " << directoryPath << std::endl;
        return 1;
    }

    std::set<std::string> sortedFiles;

    for (const auto& entry : boost::filesystem::directory_iterator(directoryPath)) {
        if (boost::filesystem::is_regular_file(entry.path())) {
            sortedFiles.insert(entry.path().filename().string());
        }
    }

    for (const auto& filename : sortedFiles) {
        std::cout << "Loading from " << filename << std::endl;
        std::string path_joined = directoryPath + filename;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(path_joined, *cloud) == -1) {
            std::cerr << "Couldn't read PCD file: " << path_joined << std::endl;
        } else {
            pointClouds.push_back(cloud);
            }
    }


    // Declare a point cloud object
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr n_referencia(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr n_nueva(new pcl::PointCloud<pcl::PointXYZRGB>);

    n_referencia = pointClouds[0];
    n_nueva = pointClouds[1];

    // 3. Downsample the two point clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr n_referencia_reducida(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr n_nueva_reducida(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(n_referencia);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
    sor.filter(*n_referencia_reducida);

    sor.setInputCloud(n_nueva);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
    sor.filter(*n_nueva_reducida);

    // 4. Calculate the transformation to align the point clouds
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputSource(n_nueva_reducida);
    icp.setInputTarget(n_referencia_reducida);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr n_nueva_transformada(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nubes_fusionadas(new pcl::PointCloud<pcl::PointXYZRGB>);
    icp.align(*n_nueva_transformada);

    // Get the transformation
    Eigen::Matrix4f transformacion = icp.getFinalTransformation();

    // 5. Apply the accumulated transformation to the new point cloud
    pcl::transformPointCloud(*n_nueva, *n_nueva_transformada, transformacion);

    // 6. Merge the transformed new point cloud with the fusion result
    *nubes_fusionadas = *n_referencia + *n_nueva_transformada;

    // Eigen::Matrix4f transformacion_acumulada = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f transformacion_acumulada = transformacion;

    // 7. Loop through the remaining point clouds
    for (int i = 2; i < pointClouds.size(); i++)
    {   
        std::cout << "Mergin " << i << "..." << std::endl;

        // 2. Load a new point cloud
        n_nueva = pointClouds[i];

        // 3. Use the previous point cloud as reference to align the new one
        pcl::copyPointCloud(*n_nueva_reducida, *n_referencia_reducida);

        // 4. Downsample the new point cloud
        sor.setInputCloud(n_nueva);
        sor.setLeafSize(0.1f, 0.1f, 0.1f);
        sor.filter(*n_nueva_reducida);

        // 5. Calculate the transformation to align the point clouds
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setMaximumIterations(1000);
        icp.setInputSource(n_nueva_reducida);
        icp.setInputTarget(n_referencia_reducida);
        icp.align(*n_nueva_transformada);

        // Get the transformation
        transformacion = icp.getFinalTransformation();

        // 6. Accumulate the transformations
        transformacion_acumulada =  transformacion * transformacion_acumulada;

        // 7. Apply the accumulated transformation to the new point cloud
        pcl::transformPointCloud(*n_nueva, *n_nueva_transformada, transformacion_acumulada);

        // 8. Merge the transformed new point cloud with the fusion result
        *nubes_fusionadas += *n_nueva_transformada;
    }

    std::cout << "Nubes fusionadas con esito" << std::endl;

    // Display the fusion result
    pcl::visualization::CloudViewer viewer("Escena fusionada");
    viewer.showCloud(nubes_fusionadas);

    while(!viewer.wasStopped()){

    }

    pcl::io::savePCDFileASCII(result_root_joined.string(), *nubes_fusionadas);

    return 0;
}