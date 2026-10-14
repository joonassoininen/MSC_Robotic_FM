import numpy as np
import time
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float64MultiArray, Float32, Float64
from geometry_msgs.msg import PoseStamped
from crisp_py.gripper import make_gripper
from crisp_py.robot import make_robot
from rclpy.qos import qos_profile_sensor_data
import sys, select, termios, tty
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Slerp
from std_msgs.msg import Float64
from rclpy.action import ActionClient
from franka_msgs.action import Move, Grasp

# --------------------------------------
#
# Teleoperation script used to control the
# Robot without crisp_gym
#
# --------------------------------------


class HTCTeleop(Node):
    def __init__(self):
        super().__init__("HTC_teleop")
        
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.io_group = ReentrantCallbackGroup()


        # === Initialize robot ===
        self.robot = make_robot("fr3", namespace="franka")
        self.robot.wait_until_ready()
        
        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        self.robot.cartesian_controller_parameters_client.load_param_config(
            file_path="config/control/default_cartesian_impedance.yaml"
        )
        print("Robot initialized!")
        time.sleep(1.0)

        # Copy the robot pose for the teleoperation and initialize rotation
        self.current_target_pose = self.robot.end_effector_pose.copy()
        self.base_down = Rotation.from_quat([0.99999999, 0.00004908, 0.0, 0.0])
        self.target_rotation = self.base_down 

        # ---------------  Gripper --------------------------------
        self.gripper = make_gripper("gripper_franka", namespace="franka")
        self.gripper.config.max_delta = 0.15
        self.gripper.wait_until_ready()

        self.scale_factor = 1.0
        self.settings = termios.tcgetattr(sys.stdin)

        self._move_client = ActionClient(self, Move, '/franka/franka_gripper/move')
        self._grasp_client = ActionClient(self, Grasp, '/franka/franka_gripper/grasp')
        
        self.last_gripper_button = -1

        self.opened = 1.0
        self.gripper_pos = 0.0
        # ----------------------------------------------------------

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.from_frame = 'libsurvive_world'

        # Uncomment the used controller and comment the unused one
        #self.to_frame = 'LHR-F7F7BFC5'
        self.to_frame = 'LHR-F5279FC2'

        self.last_so_pos = None 
        self.last_rot = self.base_down

        # timer for the teleoperation loop
        self.vive_timer = self.create_timer(
            1.0 / 30.0, 
            self.vive_tf_loop, 
            callback_group=self.io_group
        )

        # Subscriber for controller button actions
        self.joy_sub = self.create_subscription(
            Joy, 
            "/libsurvive/joy",
            self.joy_callback,
            10
        )

        self.world_to_robot_matrix = np.eye(3)
        self.is_calibrated = False

        self.total_robot_rotation = None

        self.stopped = False
        self.first_data_received = False

        self.precise_step = 0.001
        self.precise_mode = False

        self.reference_rot = None

        self.rate = 1.0 / 30.0
        self.timer = self.create_timer(self.rate, self.update_robot, callback_group=self.control_group)

        self.last_print_time = time.time()
        self.last_scaled_robot_rotation = None

        self.allow_rotating = False
        print("ok3")

    def joy_callback(self, msg):
        
        # Calibration button
        if len(msg.buttons) > 6 and msg.buttons[6] == 1:
            self.calibrate_from_imu()
        # Dead man switch button
        if msg.buttons[17] == 1:
            self.stopped = False
        if msg.buttons[17] == 0:
            self.stopped = True
        
        # Gripper controlling
        current = msg.buttons[0]

        if current != self.last_gripper_button:
            if current == 0:
                self.open_gripper()
            if current == 1:
                self.close_gripper()
            self.last_gripper_button = current
        

    def open_gripper(self):
        if not self._move_client.wait_for_server(timeout_sec=1.0):
                return
        goal = Move.Goal()
        goal.width = 0.08
        goal.speed = 0.2
        self._move_client.send_goal_async(goal)
        self.opened = 1.0

    def close_gripper(self):
        if not self._grasp_client.wait_for_server(timeout_sec=1.0):
                return
        goal = Grasp.Goal()
        goal.width = 0.01
        goal.speed = 0.05
        goal.force = 20.0
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.040
        self._grasp_client.send_goal_async(goal)
        self.opened = 0.0       

        
    
    def calibrate_from_imu(self):

        self.get_logger().info("Calibration started!")
        t = self.tf_buffer.lookup_transform(self.from_frame, self.to_frame, rclpy.time.Time())
        
        current_rot = Rotation.from_quat([
            t.transform.rotation.x, t.transform.rotation.y,
            t.transform.rotation.z, t.transform.rotation.w
        ])
        
        #gravity_world = np.array([0,0,-1])

        controller_forward_local = np.array([0,-1,0])

        robot_x_in_world = current_rot.apply(controller_forward_local)

        robot_x_in_world[2] = 0
        robot_x_in_world /= np.linalg.norm(robot_x_in_world)

        robot_z_in_world = np.array([0,0,1])
        robot_y_in_world = np.cross(robot_z_in_world, robot_x_in_world)

        self.world_to_robot_matrix = np.array([
            robot_x_in_world,
            robot_y_in_world,
            robot_z_in_world
        ])

        self.is_calibrated = True
        self.target_rotation = self.base_down

        self.get_logger().info("Calibration ready")
        self.last_so_pos = None
    
    def vive_tf_loop(self):
        
        if not self.is_calibrated:
            
            if not hasattr(self, '_last_cal_warn') or time.time() - self._last_cal_warn > 3.0:
                self.get_logger().info("Calibrate the controller first.")
                self._last_cal_warn = time.time()
            return

        try:
            
            t = self.tf_buffer.lookup_transform(
                self.from_frame,
                self.to_frame,
                rclpy.time.Time()
            )
            
            
            current_so_pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])

            current_rot = Rotation.from_quat([
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w
            ])
            

            if self.precise_mode == True: 
                self.last_so_pos = current_so_pos

            if self.precise_mode == False:
                if self.last_so_pos is None:
                    self.last_so_pos = current_so_pos
                    self.reference_rot = current_rot
                    self.first_data_received = True
                    return

                
                delta_world = current_so_pos - self.last_so_pos

                
                if np.linalg.norm(delta_world) > 0.05:
                    self.get_logger().warn("Too big delta in Vive tracking!")
                    self.last_so_pos = current_so_pos
                    return
                
                delta_robot = np.dot(self.world_to_robot_matrix, delta_world)
                
                if self.stopped == True:
                    delta_robot[0] = 0.0
                    delta_robot[1] = 0.0
                    delta_robot[2] = 0.0
                    self.last_so_pos = current_so_pos
                    self.last_rot = current_rot
                
                self.current_target_pose.position[0] += delta_robot[0] * 1.2
                self.current_target_pose.position[1] += delta_robot[1] * 1.2
                self.current_target_pose.position[2] += delta_robot[2] * 1.5
                
                self.current_target_pose.position[2] = max(self.current_target_pose.position[2], 0.02)

                self.last_so_pos = current_so_pos

                self.last_rot = current_rot

                if self.reference_rot is not None:

                    total_world_rot = current_rot * self.reference_rot.inv()

                    r_matrix = Rotation.from_matrix(self.world_to_robot_matrix)
                    self.total_robot_rotation = r_matrix * total_world_rot * r_matrix.inv()

                    rot_scale_factor = 0.5

                    times = [0.0, 1.0]
                    key_rotations = Rotation.from_quat([
                        Rotation.identity().as_quat(),
                        self.total_robot_rotation.as_quat()
                    ])

                    slerp = Slerp(times, key_rotations)

                    scaled_robot_rotation = slerp(rot_scale_factor)

                    if self.last_scaled_robot_rotation == None:
                        self.last_scaled_robot_rotation = scaled_robot_rotation
                

                    if self.stopped == False:
                        self.last_scaled_robot_rotation = scaled_robot_rotation
                        self.target_rotation = scaled_robot_rotation * self.base_down
                    
                    if self.stopped == True:
                        self.reference_rot = current_rot
                        self.target_rotation = self.last_scaled_robot_rotation * self.base_down

             

        except TransformException as ex:
            pass
        
    def update_robot(self):
        if not self.first_data_received:
            return
        
        try:
            self.current_target_pose.orientation = self.target_rotation
            self.robot.set_target(pose=self.current_target_pose)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main():
    rclpy.init()
    teleop = HTCTeleop()
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        executor = MultiThreadedExecutor()
        rclpy.spin(teleop, executor=executor)
        
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        teleop.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()