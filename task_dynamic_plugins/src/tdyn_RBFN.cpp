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
#include <hiqp/tasks/tdyn_RBFN.h>
#include <hiqp/utilities.h>

#include <pluginlib/class_list_macros.h>

#define REWARD_THRESHOLD 0.9

namespace hiqp
{
  namespace tasks
  {

    TDynRBFN::TDynRBFN(
      std::shared_ptr<GeometricPrimitiveMap> geom_prim_map,
      std::shared_ptr<Visualizer> visualizer)
    : TaskDynamics(geom_prim_map, visualizer) {
      nh_ = ros::NodeHandle("~");
    }


    int TDynRBFN::init(const std::vector<std::string>& parameters,
     RobotStatePtr robot_state,
     const Eigen::VectorXd& e_initial,
     const Eigen::VectorXd& e_final) {

      int size = parameters.size();
      if (size != 11) {
        printHiqpWarning("TDynRBFN requires 11 parameters, got " 
          + std::to_string(size) + "! Initialization failed!");

        return -1;
      }

      nh_.advertiseService("policy_Search",  &TDynRBFN::policySearch, this);
      nh_.advertiseService("add_Noise",  &TDynRBFN::addParamNoise, this);

      lambda_ = std::stod(parameters.at(1));

      kernels = std::stod(parameters.at(2));
      int num_rows = std::stod(parameters.at(3));
      double radius = std::stod(parameters.at(4));
      double height = std::stod(parameters.at(5));
      double global_pos_x = std::stod(parameters.at(6));
      double global_pos_y = std::stod(parameters.at(7));
      double global_pos_z = std::stod(parameters.at(8));

      std::vector<double> global_pos {global_pos_x, global_pos_y, global_pos_z};
      network.buildRBFNetwork(kernels,num_rows,radius,height,global_pos);;

      int intial_rollouts = std::stod(parameters.at(9));
      int max_num_samples = std::stod(parameters.at(10));
      // PoWER.setParams(kernels, intial_rollouts, max_num_samples);
    // ROS_INFO("Initializing normal distrtibution with mean %lf and variance %lf",std::stod(parameters.at(1)),std::stod(parameters.at(2)));

      std::normal_distribution<double> d2(0,1);
      this->dist.param(d2.param());
      e_dot_star_.resize(e_initial.rows());
      performance_measures_.resize(e_initial.rows());

      fk_solver_pos_ =
      std::make_shared<KDL::TreeFkSolverPos_recursive>(robot_state->kdl_tree_);
      fk_solver_jac_ =
      std::make_shared<KDL::TreeJntToJacSolver>(robot_state->kdl_tree_);
      return 0;
    }

    int TDynRBFN::update(RobotStatePtr robot_state,
     const Eigen::VectorXd& e,
     const Eigen::MatrixXd& J) {
      e_dot_star_.resize(e.size());
     
      // Calculating the task space dynamics
      Eigen::VectorXd taskSpaceDyn;
      taskSpaceDyn = -lambda_ * e;
      e_dot_star_ = taskSpaceDyn;
      ROS_INFO("ASD");
      // Calculating the null space dynamics
      std::shared_ptr<GeometricPrimitiveMap> gpm = this->getGeometricPrimitiveMap();
      std::shared_ptr<GeometricPoint> point = gpm->getGeometricPrimitive<GeometricPoint>("point_eef");
      ROS_INFO("ASD2");

      double nullSpaceDyn;

      std::vector<KinematicQuantities> kin_q_list;
      KinematicQuantities kin_q;
      kin_q.ee_J_.resize(robot_state->getNumJoints());
      kin_q.frame_id_ = point->getFrameId();
      pointForwardKinematics(kin_q_list, point, robot_state);
      Eigen::Vector3d gripperPoint;
      gripperPoint(0) = kin_q_list.back().ee_p_[0];
      gripperPoint(1) = kin_q_list.back().ee_p_[0];
      gripperPoint(2) = kin_q_list.back().ee_p_[0];
      ROS_INFO("ASD3");

      nullSpaceDyn = network.networkOutput(gripperPoint);
      ROS_INFO("ASD4");

      return 0;
    }

    int TDynRBFN::monitor() {
      return 0;
    }



    bool TDynRBFN::policySearch(grasp_learning::PolicySearch::Request& req, grasp_learning::PolicySearch::Response& res){
      double reward = calculateReward();
      std::vector<double> affectedKernels = network.getActiveKernels();
      std::vector<double> newParams = PoWER.policySearch(rollout_noise, reward, affectedKernels);
      network.updateWeights(newParams);
      if (reward > REWARD_THRESHOLD){
        res.converged = true;
        explore = false;
      }
      else{
        res.converged = false;
        explore = true;
      }
      return true;
    }

    double TDynRBFN::calculateReward(){
      std::vector<double> weights = network.getRunningWeights();
      double temp = 0;
      for (auto w: weights){
        temp += pow(w,2);
      }
      double reward = exp(-temp);

      ROS_INFO("The reward is %lf\n",reward);
      return reward;

    }



    std::vector<double> TDynRBFN::sampleNoise(){
      rollout_noise.clear();
      std::vector<double> noise;
      double sample = 0;
      for (int i =0;i<kernels;i++){
        sample = this->dist(this->generator);
        noise.push_back(sample);
      }
      return noise;
    }

    bool TDynRBFN::addParamNoise(grasp_learning::AddNoise::Request& req, grasp_learning::AddNoise::Response& res){

      network.resetRunningWeights();
      if (explore == true){
        rollout_noise = sampleNoise();
        network.addWeightNoise(rollout_noise);
      }

      res.sucess = true;
      return true;
    }


  } // namespace tasks

} // namespace hiqp

PLUGINLIB_EXPORT_CLASS(hiqp::tasks::TDynRBFN,
 hiqp::TaskDynamics)
