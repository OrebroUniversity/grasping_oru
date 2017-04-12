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

#include <hiqp/tasks/tdyn_random.h>
#include <pluginlib/class_list_macros.h>

#include <ros/ros.h>

namespace hiqp
{
namespace tasks
{

    TDynRandom::TDynRandom(
      std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
      std::shared_ptr<Visualizer> visualizer)
     : TaskDynamics(geom_prim_map, visualizer) {}


  int TDynRandom::init(const std::vector<std::string>& parameters,
                               RobotStatePtr robot_state,
                               const Eigen::VectorXd& e_initial,
                               const Eigen::VectorXd& e_final) {

    int size = parameters.size();
    if (size != 3) {
      printHiqpWarning("TDynRandom requires 3 parameters, got " 
        + std::to_string(size) + "! Initialization failed!");

      return -1;
    }

        starting_pub_ =
      nh_.advertise<std_msgs::String>("random_dyn_start", 1);
      msg_.data = "updating";


    // lambda_ = std::stod( parameters.at(1) );
    // ROS_INFO("Initializing normal distrtibution with mean %lf and variance %lf",std::stod(parameters.at(1)),std::stod(parameters.at(2)));
    std::normal_distribution<double> d2(std::stod(parameters.at(1)),std::stod(parameters.at(2)));
    this->dist.param(d2.param());
    e_dot_star_.resize(e_initial.rows());
    performance_measures_.resize(e_initial.rows());

    return 0;
  }

  int TDynRandom::update(RobotStatePtr robot_state,
                                 const Eigen::VectorXd& e,
                                 const Eigen::MatrixXd& J) {
    //e_dot_star_.resize(e.size());
    // e_dot_star_ = -lambda_ * e;
    Eigen::VectorXd random_e_;
    random_e_.resize(e.size());
    double sample = 0;
      starting_pub_.publish(msg_);
    for(unsigned int i=0;i<random_e_.size();i++){
      sample = this->dist(this->generator);
      random_e_[i] = sample;
    }
    e_dot_star_ = random_e_;

    return 0;
  }

  int TDynRandom::monitor() {
    return 0;
  }

} // namespace tasks

} // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDynRandom,
                       hiqp::TaskDynamics)
