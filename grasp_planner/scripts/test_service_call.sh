rosservice call /gplanner/plan_grasp \
"header:
    seq: 0
    stamp: {secs: 0, nsecs: 0}
    frame_id: 'yumi_body'
approach_frame: 'yumi_body'
approach_vector: [0.0, 0.0, 0.0]
approach_angle: -1.0
objectPose:
    position: {x: 0.46, y: 0.0, z: 0.02}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
object_radius: 0.03
object_height: 0.1" 

