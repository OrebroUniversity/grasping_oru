#include <sdf_collision_check/sdf_collision_checker.h>
#include <std_srvs/Empty.h>
#include <Eigen/Geometry>

namespace sdf_collision_check {

SDFCollisionChecker::SDFCollisionChecker() : raycast_steps(200) {
  nh_ = ros::NodeHandle("~");
  n_ = ros::NodeHandle();

  nh_.param<std::string>("sdf_map_topic", sdf_map_topic, "sdf_map");
  myGrid_ = new float***;
  validMap = false;

  ROS_INFO("Constructor done");
}

SDFCollisionChecker::~SDFCollisionChecker() {
  sdf_map_sub_.shutdown();
  ROS_INFO("Unsubscribed from map topic.");
  if (validMap && myGrid_ != NULL) {
    // clean up memory
    float*** grid;
    grid = *myGrid_;
    for (int x = 0; x < XSize; ++x) {
      for (int y = 0; y < YSize; ++y) {
        delete[] grid[x][y];
      }
      delete[] grid[x];
    }
    delete grid;
    delete myGrid_;
  }
}

void SDFCollisionChecker::init() {
  // subscribe to topic
  // sdf_map_sub_ = n_.subscribe(sdf_map_topic, 1,
  // &SDFCollisionChecker::mapCallback, this);
  map_available_ = false;
  // ROS_INFO("subscribed to topics");
  ros::ServiceClient client =
      n_.serviceClient<sdf_tracker_msgs::GetSDFMap>("/gplanner/map_to_edt");
  client.waitForExistence();
  sdf_tracker_msgs::GetSDFMap msg;
  if (client.call(msg)) {
    ROS_INFO("Asked GraspPlanner for map.");
    mapCallback(msg.response.map);
    map_available_ = true;
  } else {
    ROS_ERROR(
        "There was a problem calling the grasp planner for map. Collision "
        "avoidance may not work.");
  }
}

void SDFCollisionChecker::mapCallback(const sdf_tracker_msgs::SDFMap& msg) {
  if (!this->isActive()) {
    this->activate();
  }

  ROS_INFO("got a new map");
  // lock data duffer mutex. this should not make any difference as the que size
  // is 1, but just in case
  buffer_mutex.lock();

  // old sizes, needed for dealloc
  double xs, ys, zs;
  xs = XSize;
  ys = YSize;
  zs = ZSize;
  float ****grid, ***buffer;
  if (validMap) grid = myGrid_;

  int ctr = 0;
  // allocate and copy
  buffer = new float**[msg.XSize];
  for (int x = 0; x < msg.XSize; ++x) {
    buffer[x] = new float*[msg.YSize];
    for (int y = 0; y < msg.YSize; ++y) {
      buffer[x][y] = new float[msg.ZSize * 2];
      for (int z = 0; z < 2 * msg.ZSize; ++z) {
        buffer[x][y][z] = msg.grid[ctr];
        ctr++;
      }
    }
  }

  ROS_INFO("Copied out into buffer");

  data_mutex.lock();
  // data swap
  map_frame_id = msg.header.frame_id;
  resolution = msg.resolution;
  Wmax = msg.Wmax;
  Dmax = msg.Dmax;
  Dmin = msg.Dmin;
  XSize = msg.XSize;
  YSize = msg.YSize;
  ZSize = msg.ZSize;

  *myGrid_ = buffer;
  // SaveSDF("mymap.vti");
  data_mutex.unlock();

  ROS_INFO("Buffers switched");
  if (validMap) {
    // dealloc grid
    for (int x = 0; x < xs; ++x) {
      for (int y = 0; y < ys; ++y) {
        delete[](*grid)[x][y];
      }
      delete[](*grid)[x];
    }
    delete (*grid);
  }
  validMap = true;
  buffer_mutex.unlock();

  ROS_INFO("Cleaned up and done");
  map_available_ = true;
}

bool SDFCollisionChecker::obstacleGradient(const Eigen::Vector3d& x,
                                           Eigen::Vector3d& g,
                                           std::string frame_id) {
  // g<<Dmax,Dmax,Dmax;
  g = Eigen::Vector3d(1, 1, 1) * std::numeric_limits<double>::quiet_NaN();

  if (!this->isActive()) return false;
  if (!validMap) return false;

  // if a new frame_id, check on TF for a transformation to the correct frame
  // and buffer
  if (frame_id != "") {
    if (frame_id != request_frame_id) {
      // update transform
      tf::StampedTransform r2m;
      try {
        tl_.waitForTransform(map_frame_id, frame_id, ros::Time(0),
                             ros::Duration(0.80));
        tl_.lookupTransform(map_frame_id, frame_id, ros::Time(0), r2m);
        request_frame_id = frame_id;
      } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
        request2map.setIdentity();
      }
      tf::transformTFToEigen(r2m, request2map);
    }
  } else {
    request2map.setIdentity();
  }
  // transform x to map frame
  Eigen::Vector3d x_new;
  x_new = request2map * x;

  data_mutex.lock();
  if (ValidGradient(x_new)) {
    g(0) = SDFGradient2(x_new, 0);
    g(1) = SDFGradient2(x_new, 1);
    g(2) = SDFGradient2(x_new, 2);
    g.normalize();  // normal vector
    g = ShootSingleRay(x_new, g);
    // g = -g * SDF(x_new);  // scale by interpolated SDF value
    g = request2map.inverse().rotation() * g;
  }
  data_mutex.unlock();

  return true;
}

Eigen::Vector3d SDFCollisionChecker::ShootSingleRay(
    const Eigen::Vector3d& start, const Eigen::Vector3d& direction_) {
  Eigen::Vector4d camera;
  camera << start(0), start(1), start(2), 1;
  Eigen::Vector3d direction = direction_;
  direction.normalize();
  Eigen::Vector4d p;
  p << direction(0), direction(1), direction(2), 0;

  bool hit = false;

  //  double scaling = Dmax + Dmin;
  double scaling = 0.0;
  double scaling_prev = 0.0;
  int steps = 0;
  double D = resolution;

  while (steps < raycast_steps * 2 && !hit) {
    double D_prev = D;
    Eigen::Vector4d temp4d = camera + p * scaling;
    Eigen::Vector3d temp3d;
    temp3d << temp4d(0), temp4d(1), temp4d(2);

    double i, j, k;
    modf(temp3d(0) / resolution + XSize / 2, &i);
    modf(temp3d(1) / resolution + YSize / 2, &j);
    modf(temp3d(2) / resolution + ZSize / 2, &k);
    int I = static_cast<int>(i);
    int J = static_cast<int>(j);
    int K = static_cast<int>(k);

    if (!(I >= 0 && I < XSize && J >= 0 && J < YSize && K >= 0 && K < ZSize)) {
      // ROS_INFO("Test point outside map: %lf, %lf %lf", temp3d(0), temp3d(1),
      // temp3d(2));
      return Eigen::Vector3d(1, 1, 1) *
             std::numeric_limits<double>::quiet_NaN();
    }

    D = SDF(temp3d);
    // ROS_INFO("D: %lf", D);
    // ROS_INFO("Test point: %lf, %lf %lf", temp3d(0), temp3d(1), temp3d(2));
    // ROS_INFO("Scaling: %lf, P: %lf, %lf, %lf", scaling, p(0), p(1), p(2));

    if (D < 0.0)  // hit
    {
      scaling = scaling_prev + (scaling - scaling_prev) * D_prev / (D_prev - D);
      hit = true;
      Eigen::Vector4d currentPoint = camera + p * scaling;

      double i, j, k;
      modf(currentPoint(0) / resolution + XSize / 2, &i);
      modf(currentPoint(1) / resolution + YSize / 2, &j);
      modf(currentPoint(2) / resolution + ZSize / 2, &k);
      int I = static_cast<int>(i);
      int J = static_cast<int>(j);
      int K = static_cast<int>(k);

      // If raycast terminates within the reconstructed volume, keep the surface
      // point.
      // ROS_INFO("Steps: %d", steps);
      if (I >= 0 && I < XSize && J >= 0 && J < YSize && K >= 0 && K < ZSize) {
        return (p * scaling).head<3>();
        //        return currentPoint.head<3>();
      } else
        return Eigen::Vector3d(1, 1, 1) *
               std::numeric_limits<double>::quiet_NaN();
    }
    scaling_prev = scaling;
    //    scaling += std::max(resolution, D);
    scaling += D;
    ++steps;
  }
  //  ROS_INFO("[FAILED] Steps: %d", steps);
  return Eigen::Vector3d(1, 1, 1) * std::numeric_limits<double>::infinity();
}

bool SDFCollisionChecker::obstacleGradientBulk(
    const CollisionCheckerBase::SamplesVector& x,
    CollisionCheckerBase::SamplesVector& g, std::string frame_id) {
  // fill in g so we have a result always
  Eigen::Vector3d gd;
  gd = Eigen::Vector3d(1, 1, 1) * std::numeric_limits<double>::quiet_NaN();
  g.clear();
  for (int i = 0; i < x.size(); ++i) {
    g.push_back(gd);
  }

  if (!this->isActive()) return false;
  if (!validMap) return false;

  // if a new frame_id, check on TF for a transformation to the correct frame
  // and buffer
  if (frame_id != "") {
    if (frame_id != request_frame_id) {
      // update transform
      tf::StampedTransform r2m;
      try {
        tl_.waitForTransform(map_frame_id, frame_id, ros::Time(0),
                             ros::Duration(0.8));
        tl_.lookupTransform(map_frame_id, frame_id, ros::Time(0), r2m);
        request_frame_id = frame_id;
      } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
        request2map.setIdentity();
      }

      tf::transformTFToEigen(r2m, request2map);
    }
  } else {
    request2map.setIdentity();
  }

  data_mutex.lock();

  // std::cout<<"Transform is "<<request2map.matrix()<<std::endl;
  for (int i = 0; i < x.size(); ++i) {
    // transform x[i] to map frame
    Eigen::Vector3d x_new;
    x_new = request2map * x[i];

    if (ValidGradient(x_new)) {
      g[i](0) = SDFGradient2(x_new, 0);
      g[i](1) = SDFGradient2(x_new, 1);
      g[i](2) = SDFGradient2(x_new, 2);
      g[i].normalize();  // normal vector
      // g[i] = -g[i] * SDF(x_new);  // scale by interpolated SDF value
      g[i] = ShootSingleRay(x_new, -g[i]);
      g[i] = request2map.inverse().rotation() * g[i];
      // ROS_INFO_THROTTLE(3, "Grad: %lf, %lf, %lf", g[i](0), g[i](1), g[i](2));
    } else {
      // ROS_WARN_THROTTLE(5, "Gradient at this point seems invalid. %lf, %lf,
      // %lf", x_new(0), x_new(1), x_new(2));
    }
  }

  data_mutex.unlock();

  return true;
}

/// NOTE: not thread safe! lock data_mutex before calling
double SDFCollisionChecker::SDF(const Eigen::Vector3d& location) {
  if (!validMap) return Dmax;
  float*** grid = *myGrid_;

  double i, j, k;
  double x, y, z;

  if (std::isnan(location(0) + location(1) + location(2))) return Dmax;

  x = modf(location(0) / resolution + XSize / 2, &i);
  y = modf(location(1) / resolution + YSize / 2, &j);
  z = modf(location(2) / resolution + ZSize / 2, &k);

  if (i >= XSize - 1 || j >= YSize - 1 || k >= ZSize - 1 || i < 0 || j < 0 ||
      k < 0)
    return Dmax;

  int I = int(i);
  int J = int(j);
  int K = int(k);

  float* N1 = &grid[I][J][K * 2];
  float* N2 = &grid[I][J + 1][K * 2];
  float* N3 = &grid[I + 1][J][K * 2];
  float* N4 = &grid[I + 1][J + 1][K * 2];

  double a1, a2, b1, b2;
  a1 = double(N1[0] * (1 - z) + N1[2] * z);
  a2 = double(N2[0] * (1 - z) + N2[2] * z);
  b1 = double(N3[0] * (1 - z) + N3[2] * z);
  b2 = double(N4[0] * (1 - z) + N4[2] * z);

  return double((a1 * (1 - y) + a2 * y) * (1 - x) +
                (b1 * (1 - y) + b2 * y) * x);
}

/// NOTE: not thread safe! lock data_mutex before calling
bool SDFCollisionChecker::ValidGradient(const Eigen::Vector3d& location) {
  if (!validMap) return false;
  float*** grid = *myGrid_;

  float eps = 10e-9;

  double i, j, k;
  modf(location(0) / resolution + XSize / 2, &i);
  modf(location(1) / resolution + YSize / 2, &j);
  modf(location(2) / resolution + ZSize / 2, &k);

  if (std::isnan(i) || std::isnan(j) || std::isnan(k)) return false;

  int I = int(i) - 1;
  int J = int(j) - 1;
  int K = int(k) - 1;

  if (I >= XSize - 4 || J >= YSize - 3 || K >= ZSize - 3 || I <= 1 || J <= 1 ||
      K <= 1)
    return false;

  // 2*K because weights and distances are packed into the same array
  float* D10 = &grid[I + 1][J + 0][2 * (K + 1)];
  float* D20 = &grid[I + 2][J + 0][2 * (K + 1)];

  float* D01 = &grid[I + 0][J + 1][2 * (K + 1)];
  float* D11 = &grid[I + 1][J + 1][2 * (K + 0)];
  float* D21 = &grid[I + 2][J + 1][2 * (K + 0)];
  float* D31 = &grid[I + 3][J + 1][2 * (K + 1)];

  float* D02 = &grid[I + 0][J + 2][2 * (K + 1)];
  float* D12 = &grid[I + 1][J + 2][2 * (K + 0)];
  float* D22 = &grid[I + 2][J + 2][2 * (K + 0)];
  float* D32 = &grid[I + 3][J + 2][2 * (K + 1)];

  float* D13 = &grid[I + 1][J + 3][2 * (K + 1)];
  float* D23 = &grid[I + 2][J + 3][2 * (K + 1)];

  if (D10[0] > Dmax - eps || D10[2 * 1] > Dmax - eps || D20[0] > Dmax - eps ||
      D20[2 * 1] > Dmax - eps ||

      D01[0] > Dmax - eps || D01[2 * 1] > Dmax - eps || D11[0] > Dmax - eps ||
      D11[2 * 1] > Dmax - eps || D11[2 * 2] > Dmax - eps ||
      D11[2 * 3] > Dmax - eps || D21[0] > Dmax - eps ||
      D21[2 * 1] > Dmax - eps || D21[2 * 2] > Dmax - eps ||
      D21[2 * 3] > Dmax - eps || D31[0] > Dmax - eps ||
      D31[2 * 1] > Dmax - eps ||

      D02[0] > Dmax - eps || D02[2 * 1] > Dmax - eps || D12[0] > Dmax - eps ||
      D12[2 * 1] > Dmax - eps || D12[2 * 2] > Dmax - eps ||
      D12[2 * 3] > Dmax - eps || D22[0] > Dmax - eps ||
      D22[2 * 1] > Dmax - eps || D22[2 * 2] > Dmax - eps ||
      D22[2 * 3] > Dmax - eps || D32[0] > Dmax - eps ||
      D32[2 * 1] > Dmax - eps ||

      D13[0] > Dmax - eps || D13[2 * 1] > Dmax - eps || D23[0] > Dmax - eps ||
      D23[2 * 1] > Dmax - eps)
    return false;
  else
    return true;
}

double SDFCollisionChecker::SDFGradient2(const Eigen::Vector3d& location,
                                         int dim) {
  double delta = resolution;
  Eigen::Vector3d location_offset = Eigen::Vector3d(0, 0, 0);
  location_offset(dim) = delta;

  auto fx2 = SDF(location + (2 * location_offset));
  auto fx_2 = SDF(location - (2 * location_offset));
  auto fx = SDF(location + location_offset);
  auto fx_ = SDF(location - location_offset);

  return (fx2 - fx_2 + 8 * (fx - fx_)) / (12 * delta);
}

/// NOTE: not thread safe! lock data_mutex before calling
double SDFCollisionChecker::SDFGradient(const Eigen::Vector3d& location,
                                        int dim) {
  double delta = resolution;
  Eigen::Vector3d location_offset = Eigen::Vector3d(0, 0, 0);
  location_offset(dim) = delta;
  return ((SDF(location + location_offset)) -
          (SDF(location - location_offset))) /
         (2.0 * delta);
}

// void SDFCollisionChecker::SaveSDF(const std::string& filename) {
//   // http://www.vtk.org/Wiki/VTK/Examples/Cxx/IO/WriteVTI

//   vtkSmartPointer<vtkImageData> sdf_volume =
//       vtkSmartPointer<vtkImageData>::New();

//   sdf_volume->SetDimensions(XSize, YSize, ZSize);
//   sdf_volume->SetOrigin(resolution * XSize / 2, resolution * YSize / 2,
//                         resolution * ZSize / 2);

//   float spc = resolution;
//   sdf_volume->SetSpacing(spc, spc, spc);

//   vtkSmartPointer<vtkFloatArray> distance =
//       vtkSmartPointer<vtkFloatArray>::New();
//   vtkSmartPointer<vtkFloatArray> weight =
//   vtkSmartPointer<vtkFloatArray>::New();

//   int numCells = ZSize * YSize * XSize;

//   distance->SetNumberOfTuples(numCells);
//   weight->SetNumberOfTuples(numCells);

//   int i, j, k, offset_k, offset_j;
//   for (k = 0; k < ZSize; ++k) {
//     offset_k = k * XSize * YSize;
//     for (j = 0; j < YSize; ++j) {
//       offset_j = j * XSize;
//       for (i = 0; i < XSize; ++i) {
//         int offset = i + offset_j + offset_k;
//         distance->SetValue(offset, (*myGrid_)[i][j][k * 2]);
//         weight->SetValue(offset, (*myGrid_)[i][j][k * 2 + 1]);
//       }
//     }
//   }
//   sdf_volume->GetPointData()->AddArray(distance);
//   distance->SetName("Distance");

//   sdf_volume->GetPointData()->AddArray(weight);
//   weight->SetName("Weight");

//   vtkSmartPointer<vtkXMLImageDataWriter> writer =
//       vtkSmartPointer<vtkXMLImageDataWriter>::New();
//   writer->SetFileName(filename.c_str());
// #if VTK_MAJOR_VERSION <= 5
//   writer->SetInput(sdf_volume);
// #else
//   writer->SetInputData(sdf_volume);
// #endif
//   writer->Write();
// }
}
