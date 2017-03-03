rosservice call /yumi/hiqp_joint_velocity_controller/set_primitives "primitives:
  -
    name: 'sphere_eef_r2'
    type: 'sphere'
    frame_id: 'gripper_r_base'
    visible: True
    color: [0.0, 0.0, 1.0, 0.2]
    parameters: [0.0, 0.0, -0.02, 0.075]
  -
    name: 'sphere_eef_r1'
    type: 'sphere'
    frame_id: 'gripper_r_base'
    visible: True
    color: [0.0, 0.0, 1.0, 0.2]
    parameters: [0.0, 0.0, 0.08, 0.075]
  -
    name: 'sphere_link_6_r'
    type: 'sphere'
    frame_id: 'yumi_link_6_r'
    visible: True
    color: [0.0, 0.0, 1.0, 0.2]
    parameters: [-0.01, -0.01, 0.01, 0.075]"

echo 'adding task now...'
sleep 1

rosservice call /yumi/hiqp_joint_velocity_controller/set_tasks "tasks:
  -
    name: 'avoid_eef_right'
    priority: 1
    visible: true
    active: true
    monitored: true
    def_params: ['TDefAvoidCollisionsSDF', 'sphere', 'sphere_eef_r1', 'sphere', 'sphere_eef_r2', 'sphere', 'sphere_link_6_r']
    dyn_params: ['TDynLinear', '50.0']"
