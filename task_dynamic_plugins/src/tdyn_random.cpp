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
      fk_solver_pos_ =
      std::make_shared<KDL::TreeFkSolverPos_recursive>(robot_state->kdl_tree_);
      fk_solver_jac_ =
      std::make_shared<KDL::TreeJntToJacSolver>(robot_state->kdl_tree_);

      return 0;
    }

    int TDynRandom::update(RobotStatePtr robot_state,
     const Eigen::VectorXd& e,
     const Eigen::MatrixXd& J) {

      const KDL::JntArray jointpositions = robot_state->kdl_jnt_array_vel_.value();

      Eigen::VectorXd random_e_;
      random_e_.resize(e.size());
      double sample = 0;
      starting_pub_.publish(msg_);
      for(unsigned int i=0;i<random_e_.size();i++){
        sample = this->dist(this->generator);
        random_e_[i] = sample;
      }
      e_dot_star_ = -0.1*e; random_e_;

      std::shared_ptr<GeometricPrimitiveMap> gpm = this->getGeometricPrimitiveMap();
      std::shared_ptr<GeometricPoint> point = gpm->getGeometricPrimitive<GeometricPoint>("point_eef");

      std::vector<KinematicQuantities> kin_q_list;
      KinematicQuantities kin_q;
      kin_q.ee_J_.resize(robot_state->getNumJoints());
      kin_q.frame_id_ = point->getFrameId();
      pointForwardKinematics(kin_q_list, point, robot_state);


      return 0;
    }

    int TDynRandom::monitor() {
      return 0;
    }

    int TDynRandom::pointForwardKinematics(
      std::vector<KinematicQuantities>& kin_q_list,
      const std::shared_ptr<geometric_primitives::GeometricPoint>& point,
      RobotStatePtr const robot_state) const {
      kin_q_list.clear();

      KDL::Vector coord = point->getPointKDL();
      KinematicQuantities kin_q;
      kin_q.ee_J_.resize(robot_state->getNumJoints());
      kin_q.frame_id_ = point->getFrameId();

      if (forwardKinematics(kin_q, robot_state) < 0) {
        printHiqpWarning(
          "TDefAvoidCollisionsSDF::primitiveForwardKinematics, primitive "
          "forward kinematics for GeometricPoint primitive '" +
          point->getName() + "' failed.");
        return -2;
      }

    // shift the Jacobian reference point
      kin_q.ee_J_.changeRefPoint(kin_q.ee_frame_.M * coord);
    // compute the ee position in the base frame
      kin_q.ee_p_ = kin_q.ee_frame_.p + kin_q.ee_frame_.M * coord;

      kin_q_list.push_back(kin_q);
    // DEBUG =========================================
        // std::cerr<<"Point coordinates: ["<<coord[0]<<" ,"<<coord[1]<<" ,"<<coord[2]<<"]"<<std::endl;

      // std::cerr<<"frame id: "<<kin_q.frame_id_<<std::endl;
      // std::cerr<<"After ref change - J:  "<<std::endl<<kin_q.ee_J_.data<<std::endl;
  // std::cerr<<"After ref change - ee: "<<kin_q.ee_p_[0]<<" ,"<<kin_q.ee_p_[1]<<" ,"<<kin_q.ee_p_[2]<<"]"<<std::endl;
    // DEBUG END =========================================

      return 0;
    }

    int TDynRandom::forwardKinematics(
      KinematicQuantities& kin_q, RobotStatePtr const robot_state) const {

      if (fk_solver_pos_->JntToCart(robot_state->kdl_jnt_array_vel_.q,
        kin_q.ee_frame_, kin_q.frame_id_) < 0) {
        printHiqpWarning(
          "TDefAvoidCollisionsSDF::forwardKinematics, end-effector FK for link "
          "'" +
          kin_q.frame_id_ + "' failed.");
      return -2;
    }

  // std::cout<<"ee_pose: "<<std::endl<<kin_q.ee_pose_<<std::endl;
    if (fk_solver_jac_->JntToJac(robot_state->kdl_jnt_array_vel_.q, kin_q.ee_J_,
     kin_q.frame_id_) < 0) {
      printHiqpWarning(
        "TDefAvoidCollisionsSDF::forwardKinematics, Jacobian computation for "
        "link '" +
        kin_q.frame_id_ + "' failed.");
    return -2;
  }

  // std::cout<<"ee_J: "<<std::endl<<kin_q.ee_J_<<std::endl;

  // Not necesserily all joints between the end-effector and base are
  // controlled, therefore the columns in the jacobian corresponding to these
  // joints must be masked to zero to avoid unwanted contributions
  for (unsigned int i = 0; i < robot_state->getNumJoints(); i++)
    if (!robot_state->isQNrWritable(i))
      kin_q.ee_J_.setColumn(i, KDL::Twist::Zero());

    return 0;
  }


} // namespace tasks

} // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDynRandom,
 hiqp::TaskDynamics)
