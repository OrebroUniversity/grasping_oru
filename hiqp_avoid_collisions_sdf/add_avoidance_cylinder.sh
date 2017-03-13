rosservice call /yumi/hiqp_joint_velocity_controller/set_primitives "primitives:
  -
    name: 'cylinder_eef_r'
    type: 'cylinder'
    frame_id: 'gripper_r_base'
    visible: True
    color: [1.0, 1.0, 0.0, 0.7]
    parameters: [0.0, 0.0, 1.0, 0.0, 0.0, 0.05, 0.05, 0.1]"

echo 'adding task now...'
sleep 1

rosservice call /yumi/hiqp_joint_velocity_controller/set_tasks "tasks:
  -
    name: 'avoid_eef_right'
    priority: 1
    visible: true
    active: true
    monitored: true
    def_params: ['TDefAvoidCollisionsSDF', 'cylinder', 'cylinder_eef_r']
    dyn_params: ['TDynLinear', '5.0']"
