#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <map>
#include <thread>
#include <sys/time.h>

#include <Eigen/StdVector>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/distances.h>
#include <pcl/common/geometry.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

#define MAXN (720000)

mutex mtx_buffer;
condition_variable sig_buffer;
MeasureGroup Measures;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic,imu_topic;
double last_timestamp_lidar = 0,last_timestamp_imu= -1.0,s_plot11[MAXN];
double firstLidarFrame;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
double first_lidar_time = 0.0;
double lidar_start_time_point = 0.0, lidar_end_time = 0,time_diff_lidar_to_imu = 0.0;
bool isInit = false ,flg_exit = false,flg_first_scan = true;
bool isFIrstGetLidar = true;
bool  lidar_pushed;
double max_acc = 1,gravity_ = 9.71;
double weight = 3, depth=0;
float evalu_dis=0;
double split_min_x[20];

deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
PointCloudXYZI::Ptr lidar_part(new PointCloudXYZI());
vector< pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > evalu_mode(16);
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
shared_ptr<Preprocess> p_pre(new Preprocess());
pcl::ModelCoefficients::Ptr cross_section(new pcl::ModelCoefficients);
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1, kdtree2, kdtree3, kdtree4, kdtree5, kdtree6, kdtree7, kdtree8, kdtree9, kdtree10, kdtree11, kdtree12, kdtree13, kdtree14, kdtree15, kdtree16;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (isFIrstGetLidar)
    {
        firstLidarFrame = msg->header.stamp.toSec();
        isFIrstGetLidar = false;
    }
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    ROS_INFO("CURRENT received lidar point at time (msg):%lf", msg->header.stamp.toSec());

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    ROS_INFO("time buffer size is :%lu", time_buffer.size());
    if (!isInit)
    {
        ROS_INFO("lidar_start_time_point is %lf ", last_timestamp_lidar);
        lidar_start_time_point = last_timestamp_lidar;
        isInit = true;
    }

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 )
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}
void get_gravity(const MeasureGroup &meas)
{
    if (meas.imu.empty())
    {
        ROS_INFO("IMU measures is empty,return!");
    };
    ROS_ASSERT(meas.lidar != nullptr);
    double g=0;
    int num=0;
    for (auto it_imu = meas.imu.begin(); it_imu < (meas.imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        g += head->linear_acceleration.z;
        num+=1;
    }
    g=g/num;
    gravity_= (gravity_+g)/2;
}
bool livox_move(const MeasureGroup &meas)
{
    if (meas.imu.empty())
    {
        ROS_INFO("IMU measures is empty,return!");
        return false;
    };
    ROS_ASSERT(meas.lidar != nullptr);
    double p_max = max_acc;
    for (auto it_imu = meas.imu.begin(); it_imu < (meas.imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        
        /*double roll, pitch, yaw;
        tf2::Quaternion orientation;
        tf2::fromMsg(head->orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        */
        if(p_max < head->linear_acceleration.x &&head->linear_acceleration.x>0)
        {
            p_max=head->linear_acceleration.x;
            cout<<"x:   "<<head->linear_acceleration.x<<endl;
        }
        else if(p_max < head->linear_acceleration.y &&head->linear_acceleration.y>0)
        {
            p_max=head->linear_acceleration.y;
            cout<<"y:   "<<head->linear_acceleration.y<<endl;
        }
        else if(p_max < head->linear_acceleration.z- gravity_ &&head->linear_acceleration.z>0)
        {
            p_max=head->linear_acceleration.z- gravity_;
            cout<<"z:   "<<head->linear_acceleration.z<<endl;
        }
        else if(p_max < head->linear_acceleration.x*(-1) &&head->linear_acceleration.x<0)
        {
            p_max=head->linear_acceleration.x*(-1);
            cout<<"x:   "<<head->linear_acceleration.x<<endl;
        }
        else if(p_max < head->linear_acceleration.y*(-1) &&head->linear_acceleration.y<0)
        {
            p_max=head->linear_acceleration.y*(-1);
            cout<<"y:   "<<head->linear_acceleration.y<<endl;
        }
        else if(p_max < (head->linear_acceleration.z- gravity_)*(-1) &&(head->linear_acceleration.z- gravity_)<0)
        {
            p_max=(head->linear_acceleration.z- gravity_)*(-1);
            cout<<"z:   "<<head->linear_acceleration.z<<endl;
        }
    }
    
    if(max_acc<p_max){
        cout<<"part acc:"<< p_max<<"  max acc:"<< max_acc<<endl;
        return true;
    }
    return false;
}
void lider_mesh(const MeasureGroup &meas)
{
    if (meas.imu.empty())
    {
        ROS_INFO("IMU measures is empty,return!");
    }
    else{
        ROS_ASSERT(meas.lidar != nullptr);
        *lidar_part += *meas.lidar;
    }
    
}
pcl::PointCloud<pcl::PointXYZ>::Ptr smooth_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
    const double dist_weigh =0.04;
    const double normal_weigh = 0.02;
    const int k = 20;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    //kdtree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_in);
    //计算法线
    pcl::NormalEstimation<pcl::PointXYZ , pcl::Normal> ne ;//法线估计对象
    pcl::PointCloud<pcl::Normal> cloud_normals;
    ne.setInputCloud(cloud_in);
    ne.setSearchMethod(tree);
    ne.setKSearch(k);
    ne.compute(cloud_normals);
    for (size_t i = 0; i < cloud_in->size(); ++i)
    {
        std::vector<int> indices;
        std::vector<float>dist_square;
        indices.reserve(k);
        indices.reserve(k);
        //
        double zeta = 0.0;
        double sum = 0.0;
        if (tree->nearestKSearch(cloud_in->points[i], k, indices, dist_square) > 0)
        {
            for (size_t j = 1; j < indices.size(); ++j)
            {
                double diffx = cloud_in->points[indices[j]].x - cloud_in->points[i].x;
                double diffy = cloud_in->points[indices[j]].y - cloud_in->points[i].y;
                double diffz = cloud_in->points[indices[j]].z - cloud_in->points[i].z;
                double dn = cloud_normals.at(i).normal_x * diffx +
                    cloud_normals.at(i).normal_y * diffy +
                    cloud_normals.at(i).normal_z * diffz;
                double w = exp(-1*dn * dn / (2*normal_weigh*normal_weigh)) * exp(-1*dist_square[j] /(2*dist_weigh*dist_weigh));
                zeta += w * dn;
                sum += w;
            }

        }
        if (sum < 1e-10)
        {
            zeta = 0.0;
        }
        else
        {
            zeta /= sum;
        }
        pcl::PointXYZ smoothed_point;
        smoothed_point.x = cloud_in->points[i].x + cloud_normals.at(i).normal_x * zeta;
        smoothed_point.y = cloud_in->points[i].y + cloud_normals.at(i).normal_y * zeta;
        smoothed_point.z = cloud_in->points[i].z + cloud_normals.at(i).normal_z * zeta;
        cloud_out->push_back(smoothed_point);
    }
    cloud_out->width = cloud_out->size();
    cloud_out->height = 1;
    cloud_out->resize(cloud_out->width * cloud_out->height);
    return cloud_out;
}
Eigen::Vector3d down_plane;
pcl::PointCloud<pcl::PointXYZ>::Ptr split_cloudpoint(double max,bool first)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->points.resize(lidar_part->size());
    for (size_t i1 = 0; i1 < lidar_part->points.size(); i1++) {
        cloud->points[i1].x = lidar_part->points[i1].x;
        cloud->points[i1].y = lidar_part->points[i1].y;
        cloud->points[i1].z = lidar_part->points[i1].z;
    }
	if(first){
        //点云提取
        for (auto& point : *cloud) {
            if (point.z < 0 && 3 < (max - point.x))cloud_plane1->push_back(point);
        }
        for (auto& point : *cloud) {
            if (point.y<0.5 && point.z<0.5 && point.z>0 && 3 < (max - point.x))cloud_plane2->push_back(point);
        }
        pcl::ModelCoefficients::Ptr coefficients1(new pcl::ModelCoefficients);//创建分割时所需的模型系数对象
        pcl::PointIndices::Ptr inliers1(new pcl::PointIndices);//创建储存内点的点索引集合对象
        pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients);//创建分割时所需的模型系数对象
        pcl::PointIndices::Ptr inliers2(new pcl::PointIndices);//创建储存内点的点索引集合对象

        pcl::SACSegmentation<pcl::PointXYZ> seg;// 创建分割对象
        seg.setOptimizeCoefficients(true);// 设置模型系数需要优化

        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);//分割的模型类型
        seg.setMethodType(pcl::SAC_RANSAC);//随机参数的估计方法
        seg.setDistanceThreshold(0.01);//阈值距离

        seg.setInputCloud(cloud_plane1);
        seg.segment(*inliers1, *coefficients1);//输出

        for (std::size_t i = 0; i < inliers1->indices.size(); ++i) {
            outputcloud1->push_back(cloud_plane1->points[inliers1->indices[i]]);
            //cloud_plane1->erase(cloud_plane1->begin() + inliers1->indices[i]);
        }
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);//分割的模型类型
        seg.setMethodType(pcl::SAC_RANSAC);//随机参数的估计方法
        seg.setMaxIterations(300);
        seg.setDistanceThreshold(0.01);//阈值距离

        seg.setInputCloud(cloud_plane2);
        seg.segment(*inliers2, *coefficients2);//输出

        for (std::size_t i = 0; i < inliers2->indices.size(); ++i)outputcloud2->push_back(cloud_plane2->points[inliers2->indices[i]]);

        Eigen::Vector3d plane1(coefficients1->values[0], coefficients1->values[1], coefficients1->values[2]);
        Eigen::Vector3d plane2(coefficients2->values[0], coefficients2->values[1], coefficients2->values[2]);
        Eigen::Vector3d director;
        down_plane = plane1;
        director = plane1.cross(plane2);
        //cout << " " << director[0] << " " << director[1] << " " << director[2];
        double p1 = max - weight;
        double p2 = (p1 / director[0]) * director[1];
        double p3 = (p1 / director[0]) * director[2];
        for (auto& point : *cloud) {
            double d = director[0] * point.x + director[1] * point.y + director[2] * point.z - director[0] * p1 - director[1] * p2 - director[2] * p3;
            if (d > 0)result->push_back(point);
        }

        cross_section->values.resize(4);
        cross_section->values[0]=director[0];
        cross_section->values[1]=director[1];
        cross_section->values[2]=director[2];
        cross_section->values[3]= - director[0] * (p1) - director[1] * p2 - director[2] * p3;
    }    
	else{
        for (auto& point : *cloud) {
            double d = cross_section->values[0] * point.x + cross_section->values[1] * point.y + cross_section->values[2] * point.z +cross_section->values[3];
            if (d > 0)result->push_back(point);
        }
    }
    return result;
}
bool ju_flag=true;
bool judge_barrier(pcl::PointCloud<pcl::PointXYZ>::Ptr result,int num)
{
    for (size_t i1 = 0; i1 < result->points.size(); i1++) {
        if(result->points[i1].x<split_min_x[num]-0.2){
            cout<<num<<" : barriers...."<<endl;
            return true;
        }
        //if(sqrt(result->points[i1].x*result->points[i1].x+result->points[i1].y*result->points[i1].y+result->points[i1].z*result->points[i1].z)<split_min_x[num]-0.2)return true;
    }
    return false;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr lastpart_cloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::KdTreeFLANN<pcl::PointXYZ> l_kdtree;
int d_num=0;
bool* judge_dynamic(PointCloudXYZI::Ptr lidarcloud,double x_max,double depth, double top ,double left, double right)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr r(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr real_dynamic_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    r->points.resize(lidarcloud->size());//都需要进行监测
    for (size_t i1 = 0; i1 < lidarcloud->points.size(); i1++) {
        pcl::PointXYZ o_result(lidarcloud->points[i1].x,lidarcloud->points[i1].y,lidarcloud->points[i1].z);
        if(o_result.z<depth)continue;
        r->push_back(o_result);
    }
    //对评估点云体素化
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
    voxel_grid_filter.setInputCloud(r);
    voxel_grid_filter.setLeafSize(0.02, 0.02, 0.02);
    voxel_grid_filter.filter(*downsampled_cloud);
    //体素对比，看相邻帧是否存在同一点，以此为判断基准
    bool* jd = new bool[16];
    int a[16]={0};
    for(int i=0;i<16;i++)jd[i]=false;
    if(lastpart_cloud->points.size()<10){
        lastpart_cloud = downsampled_cloud;
        l_kdtree.setInputCloud(downsampled_cloud);
    }
    else{
        for(const auto& point : downsampled_cloud->points){
            std::vector<int> indices;
            std::vector<float>k_dis;
            float radius = 0.2;
            if(l_kdtree.radiusSearch(point, radius, indices, k_dis)>1)continue;
            else{
                dynamic_cloud->push_back(point);
            }
        }
        l_kdtree.setInputCloud(downsampled_cloud);
        for(const auto& point : lastpart_cloud->points){
            std::vector<int> indices;
            std::vector<float>k_dis;
            float radius = 0.2;
            if(l_kdtree.radiusSearch(point, radius, indices, k_dis)>1)continue;
            else{
                dynamic_cloud->push_back(point);
            }
        }
        //对动态点密度评估，去除密度小的点
        pcl::KdTreeFLANN<pcl::PointXYZ> m_kdtree;
        m_kdtree.setInputCloud(dynamic_cloud);
        std::vector<int> indices;
        std::vector<int> d;
        std::vector<float>k_dis;
        for(int i=0;i<dynamic_cloud->points.size();i++)
        {
            pcl::PointXYZ o_result(dynamic_cloud->points[i].x,dynamic_cloud->points[i].y,dynamic_cloud->points[i].z);
            if(m_kdtree.radiusSearch(o_result, 0.2, indices, k_dis)>5)real_dynamic_cloud->push_back(o_result);
        }
        //动态点投影以及保存
        if(real_dynamic_cloud->points.size()>20){
            string file_name = to_string(d_num) +"_dynamic.pcd";
            string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(all_points_dir, *real_dynamic_cloud);
            for(const auto& point : real_dynamic_cloud->points){
                double x = (x_max-weight+1)/point.x;//(x_max-weight+1)/point.x;
                double y = point.y*x, z = point.z*x;
                double z_dis = (top-depth)/4;
                double y_dis = (left-right)/4;
                for(int i=0;(depth+i*z_dis)<top;i++){
                    for(int j=0;(right+j*y_dis)<left;j++){
                        if(z>(depth+i*z_dis)&&z<(depth+(i+1)*z_dis)){
                                if(y>(right+j*y_dis)&&y<(right+(j+1)*y_dis)){
                                    jd[i*4+j]=true;
                                }
                        }
                    }
                }
            }
            //for(int i=0;i<16;i++)if(a[i]==1)jd[i]=true;
        }
        d_num+=1;
        lastpart_cloud = downsampled_cloud;
    }
    return jd;
}
//对分块的KDTREE进行操作
vector< pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > k_cloud(16);
void kdtree_in(int i,pcl::PointCloud<pcl::PointXYZ>::Ptr k){
    switch(i){
        case 1: {kdtree1.setInputCloud(k);break;}
        case 2: {kdtree2.setInputCloud(k);break;}
        case 3: {kdtree3.setInputCloud(k);break;}
        case 4: {kdtree4.setInputCloud(k);break;}
        case 5: {kdtree5.setInputCloud(k);break;}
        case 6: {kdtree6.setInputCloud(k);break;}
        case 7: {kdtree7.setInputCloud(k);break;}
        case 8: {kdtree8.setInputCloud(k);break;}
        case 9: {kdtree9.setInputCloud(k);break;}
        case 10: {kdtree10.setInputCloud(k);break;}
        case 11: {kdtree11.setInputCloud(k);break;}
        case 12: {kdtree12.setInputCloud(k);break;}
        case 13: {kdtree13.setInputCloud(k);break;}
        case 14: {kdtree14.setInputCloud(k);break;}
        case 15: {kdtree15.setInputCloud(k);break;}
        case 16: {kdtree16.setInputCloud(k);break;}
        default:{cout<<"bad kdcreat"<<endl;}
    }
}
int* kdtree_search(int i, int K ,pcl::PointXYZ s_point){
    std::vector<int> indices(K);
    std::vector<float>k_dis(K);
    int* k_inx = new int[K] ;
    k_inx[0] = -1;
    switch(i){
        case 1: {kdtree1.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 2: {kdtree2.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 3: {kdtree3.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 4: {kdtree4.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 5: {kdtree5.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 6: {kdtree6.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 7: {kdtree7.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 8: {kdtree8.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 9: {kdtree9.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 10: {kdtree10.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 11: {kdtree11.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 12: {kdtree12.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 13: {kdtree13.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 14: {kdtree14.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 15: {kdtree15.nearestKSearch(s_point,K,indices,k_dis);break;}
        case 16: {kdtree16.nearestKSearch(s_point,K,indices,k_dis);break;}
        default:{cout<<"bad kdsearch"<<endl;}
    }
    if(k_dis[0]>0.06)return k_inx;
    int* k_int= new int[K] ;
    for(int j=0;j<K;j++)k_int[j]=indices[j];
    return k_int;
}
//创建多变形网格，用于存储结果
vector<vector<vector<int> > >polygon(16);
int v_num[16]={0};
vector< pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > E2(16);
void init_evalumode(pcl::PointCloud<pcl::PointXYZ>::Ptr all_plane, int num,int max,int num_max,int i)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr E(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (size_t i1 = 0; i1 < all_plane->points.size(); i1++) {
        pcl::PointXYZ p_1(all_plane->points[i1].x,all_plane->points[i1].y,all_plane->points[i1].z);
        E->push_back(p_1);
    }
    *E2[i]+=*E;
    //cout<<"E2.size["<<i<<"] ="<<E2[i]->points.size()<<endl;
    v_num[i]+=1;
    if(v_num[i]==2){
        pcl::UniformSampling<pcl::PointXYZ> US;
        US.setInputCloud(E2[i]);
        US.setRadiusSearch(0.01f);
        US.filter(*downsampled_cloud);
        /*pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
        voxel_grid_filter.setInputCloud(E2[i]);
        voxel_grid_filter.setLeafSize(0.02, 0.02, 0.02);
        voxel_grid_filter.filter(*downsampled_cloud);*/
        *evalu_mode[i] += *E2[i];
        E2[i]->clear();
        v_num[i]=0;

    }
    
    if(num==num_max){
        // 计算法向量
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>); //法向量点云对象指针
        pcl::NormalEstimation<pcl::PointXYZ , pcl::Normal> n ;//法线估计对象
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>) ;//存储估计的法线的指针
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>) ;
        tree->setInputCloud(evalu_mode[i]) ;
        n.setInputCloud(evalu_mode[i]) ;
        n.setSearchMethod(tree) ;
        n.setKSearch(20);
        n.compute(*normals); //计算法线，结果存储在normals中
        cout<<"normal"<<endl;
        //将点云和法线放到一起
        pcl::concatenateFields(*evalu_mode[i] , *normals , *cloud_with_normals) ;

        //创建搜索树
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>) ;
        tree2->setInputCloud(cloud_with_normals) ;
        //创建Poisson对象，并设置参数
        pcl::Poisson<pcl::PointNormal> pn ;
        pcl::PolygonMesh mesh ;
        pn.setConfidence(false); //是否使用法向量的大小作为置信信息。如果false，所有法向量均归一化。
        pn.setDegree(2.8); //设置参数degree[1,5],值越大越精细，耗时越久。
        pn.setDepth(8); //树的最大深度，求解2^d x 2^d x 2^d立方体元。由于八叉树自适应采样密度，指定值仅为最大深度。
        pn.setIsoDivide(8); //用于提取ISO等值面的算法的深度
        pn.setManifold(false); //是否添加多边形的重心，当多边形三角化时。 设置流行标志，如果设置为true，则对多边形进行细分三角话时添加重心，设置false则不添加
        pn.setOutputPolygons(false); //是否输出多边形网格（而不是三角化移动立方体的结果）
        pn.setSamplesPerNode(3); //设置落入一个八叉树结点中的样本点的最小数量。无噪声，[1.0-5.0],有噪声[15.-20.]平滑
        pn.setScale(1.25); //设置用于重构的立方体直径和样本边界立方体直径的比率。
        pn.setSolverDivide(8); //设置求解线性方程组的Gauss-Seidel迭代方法的深度

        //设置搜索方法和输入点云
        pn.setSearchMethod(tree2);
        pn.setInputCloud(cloud_with_normals);
        //执行重构
        pn.performReconstruction(mesh);
        fromPCLPointCloud2(mesh.cloud, *k_cloud[i]);
        kdtree_in(i+1,k_cloud[i]);
        vector<vector<int> > v2(k_cloud[i]->points.size()+1);
        //遍历一次，将面片顶点改为顺序存储
        for(int j=0;j<mesh.polygons.size();j++){
            int a = mesh.polygons[j].vertices[0], b =mesh.polygons[j].vertices[1], c=mesh.polygons[j].vertices[2];
            v2[a].push_back(b),v2[a].push_back(c);
            v2[b].push_back(a),v2[b].push_back(c);
            v2[c].push_back(b),v2[c].push_back(a);
        }
        polygon[i]=v2;
        cout<<"evalu_mode.size["<<i<<"] ="<<evalu_mode[i]->points.size()<<"   k_cloud.size:"<<k_cloud[i]->points.size()<<endl;
        string file_name = to_string(i) +"_evalu_mode.pcd";
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *k_cloud[i]);
        string file_nameply = to_string(i) +"_evalu_mode.ply";
        string all_points_dirply(string(string(ROOT_DIR) + "PCD/") + file_nameply);
        pcl::io::savePLYFile(all_points_dirply, mesh);
        evalu_mode[i]->clear();
        cout<<"ply end--------------------------------"<<endl;
    }
    
}
float maxdistance=0, u=0, variance=0;//计算最远点,均值，方差
int bad_num=0, split_num;
pcl::PointCloud<pcl::PointXYZ>::Ptr bad_p(new pcl::PointCloud<pcl::PointXYZ>);
bool plane_evalu(pcl::PointCloud<pcl::PointXYZ>::Ptr split_plane, int num)
{
    //计算evalu_mode和split_plane之间点的最大距离
    //cout<< "polygon["<<num<<"].size:"<<polygon[num].size()<<"  split_plane->points.size:"<<split_plane->points.size()<<endl;
    std::vector<float> split_point;
    for(const auto& point : split_plane->points)
    {
        //计算最近点的面片
        pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_p(new pcl::PointCloud<pcl::PointXYZ>);
        
        pcl::PointXYZ s_point(point.x,point.y,point.z);
        if(s_point.x==0&&s_point.y==0&&s_point.z==0)continue;
        float r=0,dis=1;
        int K=5;
        int *k_inx = kdtree_search(num+1, K, s_point);
        if(k_inx[0] == -1){
            continue;
        }
        int k_min = k_inx[0];
        for(int i=1;i<K;i++)
            if(k_min>k_inx[K])k_min = k_inx[K];
        //cout<< k_inx<<endl;
        
        int a, b, c;
        for(int j=0;j<K;j++){
            for(int i1=0;i1<polygon[num][k_inx[j]].size();i1+=2){
                a= polygon[num][k_inx[j]][i1], b =polygon[num][k_inx[j]][i1+1];
                mesh_p->push_back(k_cloud[num]->points[a]);
                mesh_p->push_back(k_cloud[num]->points[b]);
            }
        }
        
        //计算每个面片的距离并求最小值
        for(size_t i = 0; i < mesh_p->points.size(); i+=3){
            // 计算3点云所在平面以及距离
            float p1x = mesh_p->points[i].x, p1y = mesh_p->points[i].y, p1z = mesh_p->points[i].z;
            float p2x = mesh_p->points[i+1].x, p2y = mesh_p->points[i+1].y, p2z = mesh_p->points[i+1].z;
            float p3x = mesh_p->points[i+2].x, p3y = mesh_p->points[i+2].y, p3z = mesh_p->points[i+2].z;
            float p4[4];
            p4[0] = ( (p2y-p1y)*(p3z-p1z)-(p2z-p1z)*(p3y-p1y) );
            p4[1] = ( (p2z-p1z)*(p3x-p1x)-(p2x-p1x)*(p3z-p1z) );
            p4[2] = ( (p2x-p1x)*(p3y-p1y)-(p2y-p1y)*(p3x-p1x) );
            p4[3] = ( 0-(p4[0]*p1x+p4[1]*p1y+p4[2]*p1z) );
            float pca_sqrt = sqrt(p4[0]*p4[0]+p4[1]*p4[1]+p4[2]*p4[2]);
            r = fabs(p4[0] * s_point.x + p4[1] * s_point.y + p4[2]* s_point.z + p4[3])/pca_sqrt;
            if(r<dis)dis=r;
        }
        split_point.push_back(dis);
        u+=dis;
        if(dis>0.01){
            bad_num+=1;
            bad_p->push_back(s_point);
        }
        
    }
    u=u/split_point.size();
    for(int j=0;j<split_point.size();j++){
        variance+=(split_point[j]-u)*(split_point[j]-u);
    }
    variance=variance/split_point.size();
    split_num += split_plane->points.size();
    if(evalu_dis<maxdistance){
        evalu_dis=maxdistance;
        cout<< "maxdistance:"<<maxdistance<<"  evalu_dis:"<<evalu_dis<<" bad_num: "<<bad_num<<endl;
        /*string file_name1 = to_string(0) +"_split.pcd";
        string all_points_dir1(string(string(ROOT_DIR) + "PCD/") + file_name1);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir1, *split_plane);*/
    }
    //else cout<< "maxdistance:"<<maxdistance<<"  evalu_dis:"<<evalu_dis<<endl;
    
    return false;
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "up_rviz");
    ros::NodeHandle nh;

    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");

    ros::Subscriber sub_pcl = nh.subscribe(lid_topic, 100, livox_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 2000, imu_cbk);
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();

    
    PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

    sensor_msgs::PointCloud2 pcl;
    int datafile_num=0, LIDAR_NUM=0;
    string datafile_name = "data_log.txt";
    string datafile_dir(string(string(ROOT_DIR) + "Log/") + datafile_name);
    ofstream fout(datafile_dir);
    int lidar_i=0,g_num=0,init_num[16];
    int save_n=0;
    double lidar_end_time=0;
    bool re_evalu[16], first_evalu[16],first_split_flag=true;
    clock_t t1, t2, t3, t4,t5;
    for(int i=0;i<16;i++){
        evalu_mode[i].reset(new pcl::PointCloud<pcl::PointXYZ>);
        k_cloud[i].reset(new pcl::PointCloud<pcl::PointXYZ>);
        E2[i].reset(new pcl::PointCloud<pcl::PointXYZ>);
        init_num[i]=0;
        re_evalu[i]=true;
        first_evalu[i]= true;
    } 
    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_output",1000);
    while (status)
    {
        if (flg_exit)
            break;
        // 调用该函数后可以继续执行下面的代码
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                cout << "start " << endl;
                continue;
            }
            //if(g_num<10){//调整初始重力
            //    get_gravity(Measures);
            //    g_num++;
            //}
            if(livox_move(Measures)){//判断传感器是否移动 
                for(int i=0;i<16;i++){
                    first_split_flag=true;
                    init_num[i]=0;
                    first_evalu[i]= true;
                    re_evalu[i]=true;
                    evalu_mode[i] ->clear();
                    k_cloud[i] ->clear();
                    E2[i] ->clear();
                }
                continue;
            }
            if(lidar_i<10){//累积多帧点云
                lider_mesh(Measures);
                lidar_i++;
                continue;
            }
            lidar_i=0;
            
            double max = 0, down=0, left=0, right=0, top=0;
            double z_dis=0, y_dis=0;
            t1 = clock();
            
	        for (auto& point : *lidar_part) { //最深点云点提取,为了进行分块
		        if (point.x > max)max = point.x;
                if (point.z < down)down = point.z;
                if (point.y > left)left = point.y;
                if (point.z > top)top = point.z;
                if (point.y < right)right = point.y;
	        }
            depth = down+1;
            z_dis = (top-depth)/4;
            y_dis = (left-right)/4;
            cout<<"z_dis"<<z_dis<<"  y_dis"<<y_dis<<endl;/**/
            //点云分割
            pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
            vector< pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > result_split;
	        vector< pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > result_evalu;
            if(max>weight){
                cout<<"------------far--------------------"<<endl;
                //result=split_cloudpoint(max,first_split_flag);
                for (size_t i1 = 0; i1 < lidar_part->points.size(); i1++) {
                    pcl::PointXYZ o_result(lidar_part->points[i1].x,lidar_part->points[i1].y,lidar_part->points[i1].z);
                    if(o_result.x>max-3)result->push_back(o_result);
                }
                if(first_split_flag)first_split_flag=false;
                //对分割点云分块处理
                for(int i=0;(depth+i*z_dis)<top;i++){
                    for(int j=0;(right+j*y_dis)<left;j++){
                        pcl::PointCloud<pcl::PointXYZ>::Ptr result_s(new pcl::PointCloud<pcl::PointXYZ>);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr result_e(new pcl::PointCloud<pcl::PointXYZ>);
                        for (size_t i1 = 0; i1 < result->points.size(); i1++) {
                            if(result->points[i1].z>(depth+i*z_dis)&&result->points[i1].z<(depth+(i+1)*z_dis)){
                                if(result->points[i1].y>(right+j*y_dis)&&result->points[i1].y<(right+(j+1)*y_dis)){
                                    //double d = cross_section->values[0] * result->points[i1].x + cross_section->values[1] * result->points[i1].y + cross_section->values[2] * result->points[i1].z + cross_section->values[3];
		                            //if (d > 1)result_s->push_back(result->points[i1]);
                                    //if (d > 0)result_e->push_back(result->points[i1]);
                                    if (result->points[i1].x > max-weight+1)result_s->push_back(result->points[i1]);
                                    if (result->points[i1].x > max-weight)result_e->push_back(result->points[i1]);
                                }
                            }    
                        }
                        result_split.push_back(result_s);
                        result_evalu.push_back(result_e);
                    }
                }
                if(result_evalu.size()==0){
                    lidar_part ->clear();
                    continue;
                }
            }
            else{
                cout<<"------------close--------------------"<<endl;
                //result->points.resize(lidar_part->size());//都需要进行监测
                for (size_t i1 = 0; i1 < lidar_part->points.size(); i1++) {
                    pcl::PointXYZ o_result(lidar_part->points[i1].x,lidar_part->points[i1].y,lidar_part->points[i1].z);
                    if(o_result.y>-0.385&&o_result.y<2.91)result->push_back(o_result);
                }
                for (auto& point : *result) { //最深点云点提取,为了进行分块
                    if (point.x > max)max = point.x;
                    if (point.z < down)down = point.z;
                    if (point.y > left)left = point.y;
                    if (point.z > top)top = point.z;
                    if (point.y < right)right = point.y;
                }
                top = top - 0.5;
                depth = down+0.5;
                z_dis = (top-depth)/4;
                y_dis = (left-right)/4;
                cout<<"z_dis"<<z_dis<<"  y_dis"<<y_dis<<endl;
                //对分割点云分块处理
                for(int i=0;(depth+i*z_dis)<top;i++){
                    for(int j=0;(right+j*y_dis)<left;j++){
                        pcl::PointCloud<pcl::PointXYZ>::Ptr result_s(new pcl::PointCloud<pcl::PointXYZ>);
                        pcl::PointCloud<pcl::PointXYZ>::Ptr result_e(new pcl::PointCloud<pcl::PointXYZ>);
                        for (size_t i1 = 0; i1 < result->points.size(); i1++) {
                            if(result->points[i1].z>(depth+i*z_dis)&&result->points[i1].z<(depth+(i+1)*z_dis)){
                                if(result->points[i1].y>(right+j*y_dis)&&result->points[i1].y<(right+(j+1)*y_dis)){
                                    //double d = cross_section->values[0] * result->points[i1].x + cross_section->values[1] * result->points[i1].y + cross_section->values[2] * result->points[i1].z + cross_section->values[3];
		                            if (result->points[i1].x > max-weight+1)result_s->push_back(result->points[i1]);
                                    if (result->points[i1].x > max-weight)result_e->push_back(result->points[i1]);
                                }
                            }
                            
                        }
                        result_split.push_back(result_s);
                        result_evalu.push_back(result_e);
                    }
                }
                if(result_evalu.size()==0){
                    lidar_part ->clear();
                    continue;
                }
            }
            pcl::PCDWriter pcd_writer;
            cout<<"---------"<<LIDAR_NUM<<"----------------"<<endl;
            string file_name = to_string(LIDAR_NUM) +"_lidar.pcd";
            LIDAR_NUM+=1;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
            pcd_writer.writeBinary(all_points_dir, *lidar_part);
            //更新评估模型
            t3 = clock();
            bool* jd;
            bool splitsave=false;
            jd = judge_dynamic(lidar_part,max,depth,top,left,right);//动态点判断
            t5 = clock();
            for(int f=0;f<result_evalu.size();f++){
                if(re_evalu[f]){
                    if(jd[f]){
                        init_num[f]=0,v_num[f]=0;
                        k_cloud[f]->clear(),E2[f]->clear(),evalu_mode[f]->clear();
                        continue;
                    }
                    else{if(judge_barrier(result_evalu[f],f)){
                            init_num[f]=0,v_num[f]=0;
                            k_cloud[f]->clear(),E2[f]->clear(),evalu_mode[f]->clear();
                            continue;
                        }
                    }
                    init_num[f]+=1;
                    init_evalumode(result_evalu[f],init_num[f],max,5,f);
                    if(init_num[f]==5){
                        re_evalu[f]=false;
                        splitsave=true;
                        if(first_evalu[f]==true)first_evalu[f]= false;
                        for (size_t i1 = 0; i1 < k_cloud[f]->points.size(); i1++) {//计算遮挡的距离参数
                            //double min_d = sqrt(k_cloud->points[i1].x*k_cloud->points[i1].x+k_cloud->points[i1].y*k_cloud->points[i1].y+k_cloud->points[i1].z*k_cloud->points[i1].z)；
                            if(split_min_x[f]>k_cloud[f]->points[i1].x)split_min_x[f]=k_cloud[f]->points[i1].x;
                        }
                        init_num[f]=0,v_num[f]=0;
                        E2[f]->clear(),evalu_mode[f]->clear();
                    }
                }
            }
            t4 = clock();
            //通过前方是否有人判断是否开启监测不开启监测
            for(int i=0;i<result_split.size();i++){
                if(re_evalu[i]){
                    result_split[i]->clear();
                    continue;
                }
                if(jd[i]||judge_barrier(result_split[i],i)){
                    result_split[i]->clear();
                    k_cloud[i]->clear();
                    re_evalu[i] = true;
                    cout<< i <<"need re_evalu------"<<endl;
                    continue;
                }
            }
            //挖掘面监测
            double all_u=0,all_variance=0,all_maxdistance=0;
            bool re_init = true;
            for(int i=0;i<result_split.size();i++){
                if(result_split[i]->points.size()<10)continue;
                re_init = false;
                pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
                downsampled_cloud = smooth_cloud(result_split[i]);
                if(plane_evalu(downsampled_cloud,i)){
                    ROS_INFO("WARNING");
                }
                all_u +=u;
                if(all_maxdistance<maxdistance)all_maxdistance=maxdistance;
                all_variance+=variance;
                maxdistance=0, u=0, variance=0;
            }
            if(re_init){
                lidar_part ->clear();
                continue;
            }
            //数据日志存储
            all_variance = all_variance/result_split.size();
            all_u = all_u/result_split.size();
            if(datafile_num<100){
                fout<<all_u<<"  "<<all_variance<<"  "<<bad_num<<"  "<<split_num<<endl;
                //<< all_maxdistance<<"  "
                datafile_num+=1;
            }
            else fout.close();
            all_maxdistance=0, all_u=0, all_variance=0;
            bad_num=0, split_num=0;
            
            //for(int i=0;i<result_split.size();i++)if(result_split[i]->points.size()<10)splitsave=false;
            if(splitsave){
                for(int i=0;i<result_split.size();i++){
                    if(result_split[i]->points.size()<10)continue;
                    string file_name1 = to_string(i+1) +"_split.pcd";
                    string all_points_dir1(string(string(ROOT_DIR) + "PCD/") + file_name1);
                    pcd_writer.writeBinary(all_points_dir1, *(result_split[i]));
                }
                splitsave= false;
            }
            if(save_n<2){
                string file_nameb = to_string(save_n) +"_badp.pcd";
                string all_points_dirb(string(string(ROOT_DIR) + "PCD/") + file_nameb);
                pcd_writer.writeBinary(all_points_dirb, *bad_p);
            }
            save_n++;/**/
            //判断是否出现错误
            bool flag_error= false;
            for(int i=0;i<bad_p->points.size()-2;i++){
                int d_num=0;
                for(int j=i+1;j<bad_p->points.size();j++){
                    float a = bad_p->points[i].x-bad_p->points[j].x;
                    float b = bad_p->points[i].y-bad_p->points[j].y;
                    float c = bad_p->points[i].z-bad_p->points[j].z;
                    if(sqrt(a*a+b*b+c*c)<0.05)d_num+=1;
                }
                if(d_num>4){
                    flag_error= true;
                    string file_nameb = to_string(3) +"_badp.pcd";
                    string all_points_dirb(string(string(ROOT_DIR) + "PCD/") + file_nameb);
                    pcd_writer.writeBinary(all_points_dirb, *bad_p);
                }
            }
            if(flag_error){cout<<"----------------------something bad--------------------------"<<endl;}
            bad_p->clear();
            lidar_part ->clear();
            t2 = clock();
            cout<<"  t12:"<<(double)(t2-t1)/CLOCKS_PER_SEC<< "  t34:"<<(double)(t4-t3)/CLOCKS_PER_SEC<< "  t35:"<<(double)(t5-t3)/CLOCKS_PER_SEC<< "  t54:"<<(double)(t4-t5)/CLOCKS_PER_SEC<<endl;
        }
    }
}
