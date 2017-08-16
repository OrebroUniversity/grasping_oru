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

#ifndef HIQP_TDYN_RBFN_H
#define HIQP_TDYN_RBFN_H

#include <hiqp/robot_state.h>
#include <hiqp/task_dynamics.h>
#include <ros/ros.h>
#include "std_msgs/String.h"
#include <pluginlib/class_loader.h>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treejnttojacsolver.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/frames_io.hpp>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <grasp_learning/PolicySearch.h>
#include <grasp_learning/AddNoise.h>
#include <grasp_learning/RobotState.h>
#include <std_srvs/Empty.h>

// #include <hiqp/tasks/RBFNetwork.h>
// #include <hiqp/tasks/power.h>
#include <grasp_learning/CallRBFN.h>
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include <mutex>


namespace hiqp
{
  namespace tasks
  {

    struct KinematicQuantities {
      std::string frame_id_;
      KDL::Jacobian ee_J_;
      KDL::Frame ee_frame_;
      KDL::Vector ee_p_;
    };

  /*! \brief A general first-order task dynamics implementation that enforces an exponential decay of the task performance value.
   *  \author Marcus A Johansson */  
    class TDynRBFN : public TaskDynamics {
    public:

      inline TDynRBFN() : TaskDynamics() {
      }
      
      TDynRBFN(std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
       std::shared_ptr<Visualizer> visualizer);

      ~TDynRBFN() noexcept {}

      int init(const std::vector<std::string>& parameters,
       RobotStatePtr robot_state,
       const Eigen::VectorXd& e_initial,
       const Eigen::VectorXd& e_final);

      int update(RobotStatePtr robot_state,
       const Eigen::VectorXd& e,
       const Eigen::MatrixXd& J);

      int monitor();

      int pointForwardKinematics(
        std::vector<KinematicQuantities>& kin_q_list,
        const std::shared_ptr<geometric_primitives::GeometricPoint>& point,
        RobotStatePtr const robot_state) const;
  /*! Helper function which computes ee pose and Jacobian w.r.t. a given frame*/
      int forwardKinematics(KinematicQuantities& kin_q,
        RobotStatePtr const robot_state) const;

    private:
      TDynRBFN(const TDynRBFN& other) = delete;
      TDynRBFN(TDynRBFN&& other) = delete;
      TDynRBFN& operator=(const TDynRBFN& other) = delete;
      TDynRBFN& operator=(TDynRBFN&& other) noexcept = delete;

      std::vector<double> rollout_noise;
      bool explore = true;
      int kernels = 0;
      double lambda_ = 0;
      std::vector<double> global_pos;

      ros::ServiceClient client_NN_;
      ros::ServiceClient add_noise_clt_;

      ros::Publisher starting_pub_;
      ros::NodeHandle nh_;

      ros::Publisher state_pub;

      std::default_random_engine generator;
      std::normal_distribution<double> dist;
      std_msgs::String msg_;
      std::shared_ptr<KDL::TreeFkSolverPos_recursive> fk_solver_pos_;
      std::shared_ptr<KDL::TreeJntToJacSolver> fk_solver_jac_;

      double iter = 0;

      grasp_learning::CallRBFN srv_;
      grasp_learning::RobotState stateMsg;
      std::vector<double> vec;
      std::vector<double> RBFNOutput;
      double sampling = 0;

      KDL::JntArray jointarray;

      std::shared_ptr<GeometricPrimitiveMap> gpm; 
      std::shared_ptr<GeometricPoint> point;
      std::vector<KinematicQuantities> kin_q_list;
      KinematicQuantities kin_q;
      std::vector<double> qdot;
      std_srvs::Empty empty_srv_;

    };

} // namespace tasks

} // namespace hiqp

#endif // include guard