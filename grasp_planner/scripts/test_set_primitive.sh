rosservice call /yumi/hiqp_joint_velocity_controller/set_primitive "primitives:
  - 
    name: bottom
    type: plane
    frame_id: world
    visible: True
    color: [0.0, 0.0, 1.0, 1.0]
    parameters: [-0.0, -0.0, -1.0, -0.15333333611488342]
  - 
    name: top
    type: plane
    frame_id: world
    visible: True
    color: [0.0, 0.0, 1.0, 1.0]
    parameters: [0.0, 0.0, 1.0, 0.1733333319425583]
  - 
    name: left
    type: plane
    frame_id: world
    visible: True
    color: [0.0, 0.0, 1.0, 1.0]
    parameters: [-0.9048269987106323, 0.4257793724536896, -0.0, -0.38801392912864685]
  - 
    name: right
    type: plane
    frame_id: world
    visible: True
    color: [0.0, 0.0, 1.0, 1.0]
    parameters: [-0.7289685606956482, 0.6845471858978271, 0.0, -0.268203467130661]
  - 
    name: inner
    type: cylinder
    frame_id: world
    visible: True
    color: [0.0, 0.0, 1.0, 1.0]
    parameters: [0.0, 0.0, 1.0, 0.49000000953674316, 0.12999999523162842, 0.11999999731779099, 0.11428571492433548, 1.0]
  - 
    name: outer
    type: cylinder
    frame_id: world
    visible: True
    color: [0.0, 0.0, 1.0, 1.0]
    parameters: [0.0, 0.0, 1.0, 0.49000000953674316, 0.12999999523162842, 0.11999999731779099, 0.1314285695552826, 1.0]"
