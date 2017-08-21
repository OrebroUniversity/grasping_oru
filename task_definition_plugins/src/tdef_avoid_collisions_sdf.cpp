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

#include <hiqp/tasks/tdef_avoid_collisions_sdf.h>
#include <pluginlib/class_list_macros.h>
#include <sdf_collision_check/sdf_collision_checker.h>
#include <chrono>
#include <iostream>

// debug only
#include <geometry_msgs/PoseArray.h>

// distance added to the gradient norm to act as a safety margin
#define SAFETY_DISTANCE 0.010

namespace hiqp {
namespace tasks {
//==================================================================================
TDefAvoidCollisionsSDF::TDefAvoidCollisionsSDF(
    std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
    std::shared_ptr<Visualizer> visualizer)
    : TaskDefinition(geom_prim_map, visualizer) {}
//==================================================================================
TDefAvoidCollisionsSDF::~TDefAvoidCollisionsSDF() noexcept {
  collision_checker_->deactivate();
}
//==================================================================================

int TDefAvoidCollisionsSDF::init(const std::vector<std::string>& parameters,
                                 RobotStatePtr robot_state) {
  ROS_INFO("Initializing collision checker");
  collision_checker_ =
      std::make_shared<sdf_collision_check::SDFCollisionChecker>();
  collision_checker_->init();
  collision_checker_->activate();

  ROS_INFO("Collision checker initialized.");

  int size = parameters.size();
  if (size < 3) {
    printHiqpWarning(
        "TDefAvoidCollisionsSDF requires at least 3 parameters, got " +
        std::to_string(size) + "! Initialization failed!");
    return -2;
  }

  if (size % 2 != 0) {
    ROS_ERROR(
        "TDefAvoidCollisionsSDF requires even number of parameters after the "
        "task type. First the primitive type, then the primitive name. Got: "
        "%s! Failed.",
        std::to_string(size).c_str());
    return -2;
  }

  reset();

  // Get the number of samples from parameters.
  sscanf(parameters[1].c_str(), "%lf", &resolution_);

  if (resolution_ > 0.0999999) {
    ROS_ERROR("Resolution for avoid_collisions_sdf task is too low.");
    return -2;
  }

  // loop through all the geometric primitives intended for the obstacle
  // avoidance and extract the pointers
  std::shared_ptr<GeometricPrimitiveMap> gpm = this->getGeometricPrimitiveMap();

  for (unsigned int i = 2; i < size; i += 2) {
    // Make sure the type is either a point or a sphere.
    if (parameters.at(i) != "sphere" && parameters.at(i) != "point" &&
        parameters.at(i) != "cylinder") {
      ROS_ERROR(
          "Primitive is not a sphere or point or cylinder. Only sphere, point "
          "and cylinder are supported. FAILED!");
      return -2;
    }

    if (parameters.at(i) == "sphere") {
      std::shared_ptr<GeometricSphere> sphere =
          gpm->getGeometricPrimitive<GeometricSphere>(parameters.at(i + 1));
      if (sphere == nullptr) {
        ROS_ERROR("Can't find a sphere called \'%s\'. FAILED.",
                  parameters.at(i + 1).c_str());
        return -2;
      }
      if (kdl_getQNrFromLinkName(robot_state->kdl_tree_,
                                 sphere->getFrameId()) == -1) {
        ROS_ERROR(
            "TDefAvoidCollisionsSDF::init, avoidance sphere %s isn't attached "
            "to manipulator.",
            parameters.at(i + 1).c_str());
        return -2;
      }
      ROS_INFO("Adding a sphere.");
      gpm->addDependencyToPrimitive(parameters.at(i + 1), this->getTaskName());
      sphere_primitives_.push_back(sphere);
      n_dimensions_++;
    }

    else if (parameters.at(i) == "point") {
      std::shared_ptr<GeometricPoint> point =
          gpm->getGeometricPrimitive<GeometricPoint>(parameters.at(i + 1));
      if (point == nullptr) {
        ROS_ERROR("Can't find a point called \'%s\'. FAILED.",
                  parameters.at(i + 1).c_str());
        return -2;
      }
      if (kdl_getQNrFromLinkName(robot_state->kdl_tree_, point->getFrameId()) ==
          -1) {
        ROS_ERROR(
            "TDefAvoidCollisionsSDF::init, avoidance point %s isn't attached "
            "to manipulator.",
            parameters.at(i + 1).c_str());
        return -2;
      }
      gpm->addDependencyToPrimitive(parameters.at(i + 1), this->getTaskName());
      point_primitives_.push_back(point);
      n_dimensions_++;
    }

    else {
      std::shared_ptr<GeometricCylinder> cylinder =
          gpm->getGeometricPrimitive<GeometricCylinder>(parameters.at(i + 1));
      if (cylinder == nullptr) {
        ROS_ERROR("Can't find a cylinder called \'%s\'. FAILED",
                  parameters.at(i + 1).c_str());
        return -2;
      }
      if (kdl_getQNrFromLinkName(robot_state->kdl_tree_,
                                 cylinder->getFrameId()) == -1) {
        ROS_ERROR(
            "TDefAvoidCollisionsSDF::init, avoidance cylinder %s isn't "
            "attached to manipulator.",
            parameters.at(i + 1).c_str());
        return -2;
      }
      gpm->addDependencyToPrimitive(parameters.at(i + 1), this->getTaskName());
      cylinder_primitives_.push_back(cylinder);
      // For cylinders we have many points being constrained on a single
      // cylinder.
      n_dimensions_ += static_cast<int>(cylinder->getHeight() / resolution_);
      // n_dimensions_++;
    }
  }

  performance_measures_.resize(1);
  task_types_.insert(task_types_.begin(), n_dimensions_, 1);
  // -1 leq, 0 eq, 1 geq

  // DEBUG ===========================================
  // std::cerr<<"TDefAvoidCollisionsSDF::init(.)"<<std::endl;
  // std::cerr<<"size e: "<<e_.size()<<std::endl;
  // std::cerr<<"size J: "<<J_.rows()<<" "<<J_.cols()<<std::endl;
  // std::cerr<<"size task_types: "<<task_types_.size()<<std::endl;
  // DEBUG END ===========================================

  fk_solver_pos_ =
      std::make_shared<KDL::TreeFkSolverPos_recursive>(robot_state->kdl_tree_);
  fk_solver_jac_ =
      std::make_shared<KDL::TreeJntToJacSolver>(robot_state->kdl_tree_);

  root_frame_id_ =
      robot_state->kdl_tree_.getRootSegment()->second.segment.getName();

  if (nh_.getParam("/publish_gradient_visualization",
                   publish_gradient_visualization_))
    if (publish_gradient_visualization_) {
      grad_vis_pub_ =
          nh_.advertise<visualization_msgs::MarkerArray>("gradient_marker", 1);
      ROS_INFO("GRADIENTS WILL BE PUBLISHED");
    }

  gradients_pub_ =
      nh_.advertise<sdf_collision_check::SDFGradients>("sdf_gradients", 1);
  return 0;
}

//==================================================================================
/// \bug Should rewrite to survive consistency check
int TDefAvoidCollisionsSDF::update(RobotStatePtr robot_state) {
  e_.resize(0);
  J_.resize(0, robot_state->getNumJoints());

  if (!collision_checker_->map_available_) {
    //  ROS_INFO_THROTTLE(1,"Map not available yet.");
    e_ = Eigen::VectorXd::Zero(n_dimensions_);
    J_ = Eigen::MatrixXd::Zero(n_dimensions_, robot_state->getNumJoints());
    return 0;
  }

  SamplesVector gradientsViz, testPointsViz;
  std::vector<geometry_msgs::Point> gradientsMsg, testPointsMsg;

  for (unsigned int i = 0; i < point_primitives_.size(); i++) {
    // compute forward kinematics for each primitive (yet unimplemented
    // primitives such as capsules could have more than one ee_/J associated
    // with them, hence the vector-valued argument
    std::vector<KinematicQuantities> kin_q_list;
    if (pointForwardKinematics(kin_q_list, point_primitives_[i], robot_state) <
        0) {
      printHiqpWarning(
          "TDefAvoidCollisionsSDF::update, point forward kinematics "
          "computation failed.");
      return -2;
    }

    // get the gradient vectors associated with the ee points of the current
    // primitive from the SDF map
    SamplesVector test_pts;
    for (unsigned int j = 0; j < kin_q_list.size(); j++) {
      Eigen::Vector3d p(kin_q_list[j].ee_p_.x(), kin_q_list[j].ee_p_.y(),
                        kin_q_list[j].ee_p_.z());
      test_pts.push_back(p);
      testPointsViz.push_back(p);
    }

    SamplesVector gradients;
    if (!collision_checker_->obstacleGradientBulk(test_pts, gradients,
                                                  root_frame_id_)) {
      printHiqpWarning(
          "TDefAvoidCollisionsSDF::update, collision checker failed.");
      return -2;
    }
    assert(gradients.size() > 0);  // make sure a gradient was found

    gradientsViz = gradients;

    // compute the task jacobian for the current geometric primitive
    appendTaskJacobian(kin_q_list, gradients);
    // compute the task function value vector for the current geometric
    // primitive
    appendTaskFunction(kin_q_list, gradients);
  }

  for (unsigned int i = 0; i < sphere_primitives_.size(); i++) {
    // compute forward kinematics for each primitive (yet unimplemented
    // primitives such as capsules could have more than one ee_/J associated
    // with them, hence the vector-valued argument
    std::vector<KinematicQuantities> kin_q_list;
    if (sphereForwardKinematics(kin_q_list, sphere_primitives_[i],
                                robot_state) < 0) {
      printHiqpWarning(
          "TDefAvoidCollisionsSDF::update, primitive forward kinematics "
          "computation failed.");
      return -2;
    }

    // get the gradient vectors associated with the ee points of the current
    // primitive from the SDF map
    SamplesVector test_pts;
    for (unsigned int j = 0; j < kin_q_list.size(); j++) {
      Eigen::Vector3d p(kin_q_list[j].ee_p_.x(), kin_q_list[j].ee_p_.y(),
                        kin_q_list[j].ee_p_.z());
      test_pts.push_back(p);
      testPointsViz.push_back(p);
    }

    SamplesVector gradients;
    if (!collision_checker_->obstacleGradientBulk(test_pts, gradients,
                                                  root_frame_id_)) {
      printHiqpWarning(
          "TDefAvoidCollisionsSDF::update, collision checker failed.");
      return -2;
    }
    assert(gradients.size() > 0);  // make sure a gradient was found
    gradientsViz.insert(gradientsViz.end(), gradients.begin(), gradients.end());

    // for(auto grad : gradients) {
    //   ROS_INFO_THROTTLE(3, "Gradient: %lf, %lf, %lf", grad(0), grad(1),
    //   grad(2));
    // }

    // compute the task jacobian for the current geometric primitive
    appendTaskJacobian(kin_q_list, gradients);
    // compute the task function value vector for the current geometric
    // primitive
    appendTaskFunction(kin_q_list, gradients,
                       sphere_primitives_[i]->getRadius());
  }

  for (unsigned int i = 0; i < cylinder_primitives_.size(); i++) {
    // compute forward kinematics for each primitive (yet unimplemented
    // primitives such as capsules could have more than one ee_/J associated
    // with them, hence the vector-valued argument
    std::vector<KinematicQuantities> kin_q_list;
    if (cylinderForwardKinematics(kin_q_list, cylinder_primitives_[i],
                                  robot_state) < 0) {
      printHiqpWarning(
          "TDefAvoidCollisionsSDF::update, primitive forward kinematics "
          "computation failed.");
      return -2;
    }

    // get the gradient vectors associated with the ee points of the current
    // primitive from the SDF map

    auto t_begin = std::chrono::high_resolution_clock::now();

    SamplesVector test_pts;
    for (unsigned int j = 0; j < kin_q_list.size(); j++) {
      Eigen::Vector3d p(kin_q_list[j].ee_p_.x(), kin_q_list[j].ee_p_.y(),
                        kin_q_list[j].ee_p_.z());
      test_pts.push_back(p);
      testPointsViz.push_back(p);
    }

    SamplesVector gradients;
    if (!collision_checker_->obstacleGradientBulk(test_pts, gradients,
                                                  root_frame_id_)) {
      printHiqpWarning(
          "TDefAvoidCollisionsSDF::update, collision checker failed.");
      return -2;
    }

    
    std::vector<SamplesVector::iterator> min_elements;
    int interval = 5;
    for(unsigned int ii = 0; ii < std::ceil(gradients.size() / float(interval)); ii++) {

      auto compareGradients = [] (const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) {
        return lhs.norm() < rhs.norm();
      };
      
      SamplesVector::iterator min_element_in_region;
      if(gradients.end() > gradients.begin() + (ii+1)*interval) {
        min_element_in_region = std::min_element(gradients.begin() + ii*interval, gradients.begin() + (ii+1)*interval, compareGradients);
      }
      else
        min_element_in_region = std::min_element(gradients.begin() + ii*interval, gradients.end(), compareGradients);

      min_elements.push_back(min_element_in_region);
    }

    // Set the others to zero.
    for(SamplesVector::iterator it = gradients.begin(); it != gradients.end(); it++) {
      // ROS_INFO("%lf, %lf, %lf", it->x(), it->y(), it->z());
      if(std::find(min_elements.begin(), min_elements.end(), it) == min_elements.end()) {
        it->setZero();
      }
    }

    //ROS_INFO("Min elements: %ld", min_elements.size());
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> grad_comp_time = t_end - t_begin;
    this->performance_measures_(0) =
        static_cast<double>(grad_comp_time.count());

    ROS_INFO_THROTTLE(1,
                      "*************************************************"
                      "\nGradient computation time:%lf",
                      grad_comp_time.count());

    assert(gradients.size() > 0);  // make sure a gradient was found
    gradientsViz.insert(gradientsViz.end(), gradients.begin(), gradients.end());

    // for(auto grad : gradients) {
    //   ROS_INFO_THROTTLE(3, "Gradient: %lf, %lf, %lf", grad(0), grad(1),
    //   grad(2));
    // }

    // compute the task jacobian for the current geometric primitive
    appendTaskJacobian(kin_q_list, gradients);
    // compute the task function value vector for the current geometric
    // primitive
    appendTaskFunction(kin_q_list, gradients,
                       cylinder_primitives_[i]->getRadius());
  }

  for (auto gViz : gradientsViz) {
    geometry_msgs::Point gMsg;
    gMsg.x = gViz.x();
    gMsg.y = gViz.y();
    gMsg.z = gViz.z();
    gradientsMsg.push_back(gMsg);
  }

  for (auto tViz : testPointsViz) {
    geometry_msgs::Point tMsg;
    tMsg.x = tViz.x();
    tMsg.y = tViz.y();
    tMsg.z = tViz.z();
    testPointsMsg.push_back(tMsg);
  }

  sdf_collision_check::SDFGradients msg;
  msg.start = testPointsMsg;
  msg.end = gradientsMsg;
  msg.stamp = ros::Time::now();
  gradients_pub_.publish(msg);

  if (publish_gradient_visualization_)
    publishGradientVisualization(gradientsViz, testPointsViz);
  return 0;
}

//==================================================================================
void TDefAvoidCollisionsSDF::appendTaskJacobian(
    const std::vector<KinematicQuantities>& kin_q_list,
    const SamplesVector& gradients) {
  assert(kin_q_list.size() == gradients.size());
  for (unsigned int i = 0; i < gradients.size(); i++) {
    Eigen::Vector3d gradient(gradients[i]);
    J_.conservativeResize(J_.rows() + 1, Eigen::NoChange);
    // to check if a gradient to an obstacle is valid
    if (!collision_checker_->isValid(gradient) || gradient.norm() == 0.0) {
      J_.row(J_.rows() - 1).setZero();  // insert a zero-row
      continue;
    }

    // project the Jacobian onto the normalized gradient
    Eigen::Vector3d gradient_before = gradient;
    gradient.normalize();
    Eigen::MatrixXd ee_J_vel =
        kin_q_list[i].ee_J_.data.topRows<3>();  // velocity Jacobian
    // DEBUG===============================
    // std::cerr<<"Velocity Jacobian: "<<std::endl<<ee_J_vel<<std::endl;
    // std::cerr<<"Normalized gradient transpose:
    // "<<std::endl<<gradient.transpose()<<std::endl;
    // DEBUG END===============================
    J_.row(J_.rows() - 1) = -gradient.transpose() * ee_J_vel;
  }
  // DEBUG===============================
  // std::cerr<<"Task Jacobian: "<<std::endl<<J_<<std::endl;
  // DEBUG END===============================
}
//==================================================================================
void TDefAvoidCollisionsSDF::appendTaskFunction(
    const std::vector<KinematicQuantities>& kin_q_list,
    const SamplesVector& gradients, const double& offset) {
  assert(kin_q_list.size() == gradients.size());

  for (size_t i = 0; i < gradients.size(); i++) {
    Eigen::Vector3d gradient(gradients[i]);
    e_.conservativeResize(e_.size() + 1);
    // check if a gradient to an obstacle is valid
    if (!collision_checker_->isValid(gradient) || gradient.norm() == 0) {
      /// \bug this implicitly assumes that e*(e=0) = 0!
      e_(e_.size() - 1) = 0.0;  // insert zero
      continue;
    }

    double d = gradient.norm() - SAFETY_DISTANCE - offset;
    e_(e_.size() - 1) = d;  // > activation_distance_ ? 0.0 : d;
  }
  // DEBUG===============================
  // std::cerr<<"Task function value vector: "<<e_.transpose()<<std::endl;
  // DEBUG END===============================
  // if(e_(0)<0.0)
  // 	  ROS_WARN("negative e: %f",e_(0));
}
//==================================================================================
int TDefAvoidCollisionsSDF::monitor() { return 0; }
//==================================================================================
void TDefAvoidCollisionsSDF::reset() {
  n_dimensions_ = 0;
  task_types_.clear();
  point_primitives_.clear();
  sphere_primitives_.clear();
}
//==================================================================================
int TDefAvoidCollisionsSDF::forwardKinematics(
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
//==================================================================================
int TDefAvoidCollisionsSDF::sphereForwardKinematics(
    std::vector<KinematicQuantities>& kin_q_list,
    const std::shared_ptr<geometric_primitives::GeometricSphere>& sphere,
    RobotStatePtr const robot_state) const {
  kin_q_list.clear();
  KDL::Vector coord(sphere->getX(), sphere->getY(), sphere->getZ());
  KinematicQuantities kin_q;
  kin_q.ee_J_.resize(robot_state->kdl_jnt_array_vel_.q.rows());
  kin_q.frame_id_ = sphere->getFrameId();
  if (forwardKinematics(kin_q, robot_state) < 0) {
    printHiqpWarning(
        "TDefAvoidCollisionsSDF::primitiveForwardKinematics, primitive "
        "forward kinematics for GeometricSphere primitive '" +
        sphere->getName() + "' failed.");
    return -2;
  }
  // shift the Jacobian reference point
  kin_q.ee_J_.changeRefPoint(kin_q.ee_frame_.M * coord);
  // compute the ee position in the base frame
  kin_q.ee_p_ = kin_q.ee_frame_.p + kin_q.ee_frame_.M * coord;

  kin_q_list.push_back(kin_q);

  return 0;
}

// For cylinder we are trying to move the point with the least gradient on the
// cylinder.
int TDefAvoidCollisionsSDF::cylinderForwardKinematics(
    std::vector<KinematicQuantities>& kin_q_list,
    const std::shared_ptr<geometric_primitives::GeometricCylinder>& cylinder,
    RobotStatePtr const robot_state) const {
  kin_q_list.clear();
  KinematicQuantities kin_q;
  kin_q.ee_J_.resize(robot_state->kdl_jnt_array_vel_.q.rows());
  kin_q.frame_id_ = cylinder->getFrameId();

  if (forwardKinematics(kin_q, robot_state) < 0) {
    printHiqpWarning(
        "TDefAvoidCollisionsSDF::primitiveForwardKinematics, primitive "
        "forward kinematics for GeometricCylinder primitive '" +
        cylinder->getName() + "' failed.");
    return -2;
  }

  int no_of_samples = static_cast<int>(cylinder->getHeight() / resolution_);

  SamplesVector test_pts;
  for (int i = 0; i < no_of_samples; i++) {
    KDL::Vector thisPointKDL =
        (kin_q.ee_frame_.p) +
        (kin_q.ee_frame_.M * (cylinder->getDirectionKDL() * i * resolution_ /
                              cylinder->getDirectionKDL().Norm()));
    Eigen::Vector3d thisPoint(thisPointKDL(0), thisPointKDL(1),
                              thisPointKDL(2));
    test_pts.push_back(thisPoint);
  }

  // Constrain all points.
  for (int i = 0; i < no_of_samples; i++) {
    KinematicQuantities kin_q_ = kin_q;
    // shift the Jacobian reference point
    kin_q_.ee_J_.changeRefPoint(kin_q.ee_frame_.M *
                                (cylinder->getDirectionKDL() * i * resolution_ /
                                 cylinder->getDirectionKDL().Norm()));
    kin_q_.ee_p_ = KDL::Vector(test_pts[i](0), test_pts[i](1), test_pts[i](2));
    kin_q_list.push_back(kin_q_);
  }

  return 0;
}

int TDefAvoidCollisionsSDF::pointForwardKinematics(
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
  std::cerr<<"Point coordinates: "<<coord<<std::endl;
  std::cerr<<"frame id: "<<kin_q.frame_id_<<std::endl;
  // std::cerr<<"After ref change - J:
  // "<<std::endl<<kin_q.ee_J_.data<<std::endl;
  std::cerr<<"After ref change - ee: "<<kin_q.ee_p_<<std::endl;
  // DEBUG END =========================================

  return 0;
}
//==================================================================================
void TDefAvoidCollisionsSDF::publishGradientVisualization(
    const SamplesVector& gradients, const SamplesVector& test_pts) {
  assert(gradients.size() == test_pts.size());

  grad_markers_.markers.clear();

  for (unsigned int i = 0; i < gradients.size(); i++) {
    if (!collision_checker_->isValid(gradients[i]) ||
        gradients[i].norm() == 0.0) {
      continue;
    }

    Eigen::Vector3d grad = gradients[i];
    Eigen::Vector3d test_pt = test_pts[i];
    visualization_msgs::Marker g_marker;
    g_marker.ns = "gradients";
    g_marker.header.frame_id = root_frame_id_;
    g_marker.header.stamp = ros::Time::now();
    g_marker.type = visualization_msgs::Marker::ARROW;
    g_marker.action = visualization_msgs::Marker::ADD;
    g_marker.lifetime = ros::Duration(0);
    g_marker.id = grad_markers_.markers.size();

    geometry_msgs::Point start, end;
    // grad = 10*grad;
    start.x = test_pt(0);
    start.y = test_pt(1);
    start.z = test_pt(2);
    end.x = test_pt(0) + grad(0);
    end.y = test_pt(1) + grad(1);
    end.z = test_pt(2) + grad(2);
    g_marker.points.push_back(start);
    g_marker.points.push_back(end);

    g_marker.scale.x = 0.003;
    g_marker.scale.y = 0.005;
    g_marker.scale.z = 0.005;
    g_marker.color.r = 1.0;
    g_marker.color.g = 0.0;
    g_marker.color.b = 1.0;
    g_marker.color.a = 1.0;
    grad_markers_.markers.push_back(g_marker);
  }

  // publish
  grad_vis_pub_.publish(grad_markers_);
}

//==================================================================================
}  // namespace tasks

}  // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDefAvoidCollisionsSDF,
                       hiqp::TaskDefinition)
