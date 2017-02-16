#pragma once

#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <mutex>
#include <vector>

namespace sdf_collision_check {

/**
  * @brief base class for all collision checkers.
  * Provides methods to initialize and start a background thread for model
  * maintenance.
  * Provides interfaces for answering collision querries
  */
class CollisionCheckerBase {
 public:
  typedef std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d> >
      SamplesVector;

  virtual bool obstacleGradient(const Eigen::Vector3d &x, Eigen::Vector3d &g,
                                std::string frame_id = "") = 0;
  virtual bool obstacleGradientBulk(const SamplesVector &x, SamplesVector &g,
                                    std::string frame_id = "") = 0;
  virtual void init() = 0;
  // to check if a gradient to an obstacle is valid
  virtual bool isValid(const Eigen::Vector3d &grad) = 0;

  inline void activate() {
    active_mutex.lock();
    isActive_ = true;
    active_mutex.unlock();
    ROS_INFO("Collision check activated");
  }
  inline void deactivate() {
    active_mutex.lock();
    isActive_ = false;
    active_mutex.unlock();
    ROS_INFO("Collision check deactivated");
  }

  inline bool isActive() {
    bool act;
    active_mutex.lock();
    act = isActive_;
    active_mutex.unlock();
    return act;
  }

 protected:
  bool isActive_;
  std::mutex active_mutex;
};
}
