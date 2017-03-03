rosservice call /yumi/hiqp_joint_velocity_controller/set_primitives "primitives:
  - name: 'sphere_below_plane'
    type: 'sphere'
    frame_id: 'world'
    visible: True
    color: [0.0, 0.0, 1.0, 0.2]
    parameters: [0.3, 0.0, -0.2, 0.05]
  - name: 'sphere_eef_r'
    type: 'sphere'
    frame_id: 'gripper_r_base'
    visible: True
    color: [0.0, 0.0, 1.0, 0.2]
    parameters: [0.0, 0.0, 0.0, 0.05]"

sleep 1

rosservice call /yumi/hiqp_joint_velocity_controller/set_tasks "tasks:
  - name: 'move_below_table'
    priority: 2
    visible: true
    active: true
    monitored: true
    def_params: ['TDefGeomProj', 'sphere', 'sphere', 'sphere_below_plane = sphere_eef_r']
    dyn_params: ['TDynLinear', '1.0']"
