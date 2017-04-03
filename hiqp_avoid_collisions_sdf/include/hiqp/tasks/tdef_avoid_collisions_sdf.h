// The HiQP Control Framework, an optimal control framework targeted at robotics
// Copyright (C) 2017 Marcus A Johansson
// Copyright (C) 2017 Robert Krug
// Copyright (C) 2017 Chittaranjan Srinivas Swaminathan
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

#pragma once

#include <string>

#include <hiqp/task_definition.h>
#include <sdf_collision_check/collision_checker_base.h>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treejnttojacsolver.hpp>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pluginlib/class_loader.h>

namespace hiqp {
namespace tasks {
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
    SamplesVector;

/*! \brief A struct holding Jacobian and end-effector point - used for forward
 * kinematics.
 *  \author Robert Krug */
struct KinematicQuantities {
  std::string frame_id_;
  KDL::Jacobian ee_J_;
  KDL::Frame ee_frame_;
  KDL::Vector ee_p_;
};
//==============================================================================================
/*! \brief A task definition that allows avoidance of geometric primitives on
 * the manipulator with the environment given as a SDF map.
 *  \author Robert Krug */
class TDefAvoidCollisionsSDF : public TaskDefinition {
 public:
  
  inline TDefAvoidCollisionsSDF() : TaskDefinition() {}

  TDefAvoidCollisionsSDF(std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
                         std::shared_ptr<Visualizer> visualizer);
  ~TDefAvoidCollisionsSDF() noexcept;

  int init(const std::vector<std::string>& parameters,
           RobotStatePtr robot_state);

  int update(RobotStatePtr robot_state);

  int monitor();

 private:
  TDefAvoidCollisionsSDF(const TDefAvoidCollisionsSDF& other) = delete;
  TDefAvoidCollisionsSDF(TDefAvoidCollisionsSDF&& other) = delete;
  TDefAvoidCollisionsSDF& operator=(const TDefAvoidCollisionsSDF& other) =
      delete;
  TDefAvoidCollisionsSDF& operator=(TDefAvoidCollisionsSDF&& other) noexcept =
      delete;

  void reset();
  /*! This function computes the kinematic quantities for a primitive and clears
   * the kin_q vector before computing*/
  int pointForwardKinematics(
      std::vector<KinematicQuantities>& kin_q_list,
      const std::shared_ptr<geometric_primitives::GeometricPoint>& point,
      RobotStatePtr const robot_state) const;

  int sphereForwardKinematics(
      std::vector<KinematicQuantities>& kin_q_list,
      const std::shared_ptr<geometric_primitives::GeometricSphere>& sphere,
      RobotStatePtr const robot_state) const;

  /*! Helper function which computes ee pose and Jacobian w.r.t. a given frame*/
  int forwardKinematics(KinematicQuantities& kin_q,
                        RobotStatePtr const robot_state) const;

  void appendTaskJacobian(const std::vector<KinematicQuantities> kin_q_list,
                          const SamplesVector& gradients);

  void appendTaskFunction(
      const std::shared_ptr<geometric_primitives::GeometricPoint>& point,
      const std::vector<KinematicQuantities> kin_q_list,
      const SamplesVector& gradients);

  void appendTaskFunction(
      const std::shared_ptr<geometric_primitives::GeometricSphere>& sphere,
      const std::vector<KinematicQuantities> kin_q_list,
      const SamplesVector& gradients);

  std::shared_ptr<KDL::TreeFkSolverPos_recursive> fk_solver_pos_;
  std::shared_ptr<KDL::TreeJntToJacSolver> fk_solver_jac_;

  std::vector<std::shared_ptr<geometric_primitives::GeometricSphere> >
      sphere_primitives_;
  std::vector<std::shared_ptr<geometric_primitives::GeometricPoint> >
      point_primitives_;

  std::string root_frame_id_;
  /*! Interface to the SDF map*/
  std::shared_ptr<sdf_collision_check::CollisionCheckerBase> collision_checker_;

  /// \todo should change it to a proper task visualizer
  void publishGradientVisualization(const SamplesVector& gradients,
                                    const SamplesVector& test_pts);
  ros::Publisher grad_vis_pub_;
  visualization_msgs::MarkerArray grad_markers_;
  ros::NodeHandle nh_;
};

}  // namespace tasks

}  // namespace hiqp
