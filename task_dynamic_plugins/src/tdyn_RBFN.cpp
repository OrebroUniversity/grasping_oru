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

#include <hiqp/tasks/tdyn_RBFN.h>
#include <pluginlib/class_list_macros.h>

#include <ros/ros.h>


namespace hiqp {
namespace tasks {

TDynRBFN::TDynRBFN(
  std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
  std::shared_ptr<Visualizer> visualizer)
  : TaskDynamics(geom_prim_map, visualizer) {

}


int TDynRBFN::init(const std::vector<std::string>& parameters,
                   RobotStatePtr robot_state,
                   const Eigen::VectorXd& e_initial,
                   const Eigen::VectorXd& e_final) {

  int size = parameters.size();
  if (size != 3) {
    printHiqpWarning("TDynRBFN requires 3 parameters, got "
                     + std::to_string(size) + "! Initialization failed!");

    return -1;
  }

  client_NN_ = nh_.serviceClient<grasp_learning::CallRBFN>("RBFNetwork/network_output", true);

  add_noise_clt_ = nh_.serviceClient<std_srvs::Empty>("/RBFNetwork/add_weight_noise");

  state_pub = nh_.advertise<grasp_learning::RobotState>("demo_learn_manifold/robot_state", 2000);

  lambda_ = std::stod(parameters.at(1));

  task_ = parameters.at(2);

  e_dot_star_.resize(e_initial.rows());
  performance_measures_.resize(e_initial.rows());

  fk_solver_pos_ =
    std::make_shared<KDL::TreeFkSolverPos_recursive>(robot_state->kdl_tree_);
  fk_solver_jac_ =
    std::make_shared<KDL::TreeJntToJacSolver>(robot_state->kdl_tree_);

  vec.resize(3);
  return 0;
}

int TDynRBFN::update(RobotStatePtr robot_state,
                     const Eigen::VectorXd& e,
                     const Eigen::MatrixXd& J) {

  // Calculating the taskspace dynamics

  // Calculating the nullspace dynamics
  gpm = this->getGeometricPrimitiveMap();
  point = gpm->getGeometricPrimitive<GeometricPoint>("point_eef");

  kin_q_list;
  kin_q.ee_J_.resize(robot_state->getNumJoints());
  kin_q.frame_id_ = point->getFrameId();
  pointForwardKinematics(kin_q_list, point, robot_state);

  jointarray = robot_state->kdl_jnt_array_vel_.qdot;
  qdot.clear();
  for (int i = 9; i <= 15; i++) {
    qdot.push_back(jointarray(i));
  }

  sampling = robot_state->sampling_time_;
  // grasp_learning::RobotState stateMsg;

  // std::vector<double> vec {kin_q_list.back().ee_p_[0],kin_q_list.back().ee_p_[1],kin_q_list.back().ee_p_[2]};
  for (int i = 0; i < 3; i++) {
    vec[i] = kin_q_list.back().ee_p_[i];
  }

  stateMsg.gripperPos = vec;
  stateMsg.jointVel = qdot;
  stateMsg.samplingTime = sampling;
  state_pub.publish(stateMsg);


  srv_.request.pos = vec;

  if (client_NN_.call(srv_)) {
    RBFNOutput = srv_.response.result;
  } else {
    // std::cout<<"Calling RBFN server failed"<<std::endl;
  }

  if (task_.compare("frame")==0) {


    e_dot_star_.resize(1);
    frame_ = gpm->getGeometricPrimitive<GeometricFrame>("point_frame");
    frame_pos_(0) = vec[0] + sampling * RBFNOutput[0];
    frame_pos_(1) = vec[1] + sampling * RBFNOutput[1];
    frame_pos_(2) = vec[2] + sampling * RBFNOutput[2];
    // std::cout<<sampling<<" "<<RBFNOutput[0]<<" "<<RBFNOutput[1]<<" "<<RBFNOutput[2]<<std::endl;
    frame_->setCenterOffsetKDL(frame_pos_);

    e_dot_star_(0) = -lambda_ * e(0);
  }

  else if (task_.compare("plane")==0){
    e_dot_star_.resize(3);
    e_dot_star_(0) = -lambda_ * e(0);
    e_dot_star_(1) = RBFNOutput[0];
    e_dot_star_(2) = RBFNOutput[1];
  }
  else if (task_.compare("manifold")==0){
    e_dot_star_.resize(2);

    if(RBFNOutput.size()>1){
      e_dot_star_(0) = RBFNOutput[1]-lambda_ * e(0);//RBFNOutput-lambda_ * e(0);
    }
    else{
      e_dot_star_(0) = -lambda_ * e(0);//RBFNOutput-lambda_ * e(0);
    }
    e_dot_star_(1) = RBFNOutput[0];//RBFNOutput;
    
  }
  else{
    ROS_ERROR("The task is not correct");
  }
  return 0;
}


int TDynRBFN::monitor() {
  return 0;
}

int TDynRBFN::pointForwardKinematics(
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

  // std::cout<<"["<<coord[0]<<", "<<coord[1]<<", "<<coord[2]<<"]"<<std::endl;
  // std::cout<<"["<<kin_q.ee_p_[0]<<", "<<kin_q.ee_p_[1]<<", "<<kin_q.ee_p_[2]<<"]"<<std::endl;

  return 0;
}

int TDynRBFN::forwardKinematics(
  KinematicQuantities& kin_q, RobotStatePtr const robot_state) const {

  if (fk_solver_pos_->JntToCart(robot_state->kdl_jnt_array_vel_.q,
                                kin_q.ee_frame_, kin_q.frame_id_) < 0) {
    printHiqpWarning(
      "TDefAvoidCollisionsSDF::forwardKinematics, end-effector FK for link "
      "'" +
      kin_q.frame_id_ + "' failed.");
    return -2;
  }

  if (fk_solver_jac_->JntToJac(robot_state->kdl_jnt_array_vel_.q, kin_q.ee_J_,
                               kin_q.frame_id_) < 0) {
    printHiqpWarning(
      "TDefAvoidCollisionsSDF::forwardKinematics, Jacobian computation for "
      "link '" +
      kin_q.frame_id_ + "' failed.");
    return -2;
  }

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

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDynRBFN,
                       hiqp::TaskDynamics)
