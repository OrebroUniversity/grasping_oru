// The HiQP Control Framework, an optimal control framework targeted at robotics
// Copyright (C) 2016 Marcus A Johansson
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <limits>
#include <random>

#include <hiqp/utilities.h>

#include <hiqp/tasks/tdyn_policy.h>
#include <pluginlib/class_list_macros.h>

#include <ros/ros.h>

namespace hiqp
{
namespace tasks
{

    TDynPolicy::TDynPolicy(
      std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
      std::shared_ptr<Visualizer> visualizer)
     : TaskDynamics(geom_prim_map, visualizer) {}


  int TDynPolicy::init(const std::vector<std::string>& parameters,
                               RobotStatePtr robot_state,
                               const Eigen::VectorXd& e_initial,
                               const Eigen::VectorXd& e_final) {

    int size = parameters.size();
    if (size != 3) {
      printHiqpWarning("TDynPolicy requires 3 parameters, got " 
        + std::to_string(size) + "! Initialization failed!");

      return -1;
    }

      client_NN_ = nh_.serviceClient<grasp_learning::QueryNN>("query_NN",true);
      joint_effort_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("joint_effort", 1000);


    // lambda_ = std::stod( parameters.at(1) );
    // ROS_INFO("Initializing normal distrtibution with mean %lf and variance %lf",std::stod(parameters.at(1)),std::stod(parameters.at(2)));
    std::normal_distribution<double> d2(std::stod(parameters.at(1)),std::stod(parameters.at(2)));
    this->dist.param(d2.param());
    e_dot_star_.resize(e_initial.rows());
    performance_measures_.resize(e_initial.rows());

    return 0;
  }

  int TDynPolicy::update(RobotStatePtr robot_state,
                                 const Eigen::VectorXd& e,
                                 const Eigen::MatrixXd& J) {
    e_dot_star_.resize(e.size());

    // const KDL::JntArray jointpositions = robot_state->kdl_jnt_array_vel_.value();


    grasp_learning::QueryNN srv_;

    std::vector<double> vec(e.data(), e.data() + e.size());
    srv_.request.task_measures = vec;

//    srv_.request.joint_angles.push_back(0);  //yumi_joint_1_r

    // srv_.request.joint_angles.push_back(jointpositions(9));  //yumi_joint_1_r
    // srv_.request.joint_angles.push_back(jointpositions(10)); //yumi_joint_2_r
    // srv_.request.joint_angles.push_back(jointpositions(12)); //yumi_joint_3_r
    // srv_.request.joint_angles.push_back(jointpositions(13)); //yumi_joint_4_r
    // srv_.request.joint_angles.push_back(jointpositions(14)); //yumi_joint_5_r
    // srv_.request.joint_angles.push_back(jointpositions(15)); //yumi_joint_6_r
    // srv_.request.joint_angles.push_back(jointpositions(11)); //yumi_joint_7_r

    std::vector<double> sample;
    if (client_NN_.call(srv_)){
      sample = srv_.response.task_dynamics;
    }
    else{
      std::cout<<"Calling NN server failed"<<std::endl;
      sample.push_back(this->dist(this->generator));
    }
    e_dot_star_ = Eigen::VectorXd::Map(sample.data(), sample.size());
    // e_dot_star_ << sample[0], sample[0];

    return 0;
  }

  int TDynPolicy::monitor() {
    return 0;
  }

} // namespace tasks

} // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDynPolicy,
                       hiqp::TaskDynamics)
