import numpy as np
import time
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from pynput import keyboard
from crisp_py.gripper import make_gripper
from crisp_py.robot import make_robot
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64
from rclpy.action import ActionClient
from franka_msgs.action import Move, Grasp
import sys, select, termios, tty

class PS5Teleop(Node):
    def __init__(self):
        super().__init__("ps5_teleop")
        
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.io_group = ReentrantCallbackGroup()

        # === Initialize robot (same as sporadic_poses.py) ===
        self.robot = make_robot("fr3", namespace="franka")
        self.robot.wait_until_ready()

        #self.gripper = make_gripper("gripper_franka", node=self)
        #self.gripper = make_gripper("gripper_franka", node=self)
        #self.gripper.wait_until_ready()
        
        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        self.robot.cartesian_controller_parameters_client.load_param_config(
            file_path="config/control/default_cartesian_impedance.yaml"
        )
        time.sleep(1.0)
        print("ok1")
        #self.robot.wait_until_ready()
        #self.pose = self.robot.end_effector_pose.copy()
        self.current_target_pose = self.robot.end_effector_pose.copy()
        self.base_down = Rotation.from_quat([0.99999999, 0.00004908, 0.0, 0.0])
        print("ok2")
        

        # ---------------  Gripper --------------------------------
        self.gripper = make_gripper("gripper_franka", namespace="franka")
        self.gripper.config.max_delta = 0.15
        self.gripper.wait_until_ready()

        #self.gripper_pub = self.create_publisher(Float64, 'gripper_status', 10)

        self._move_client = ActionClient(self, Move, '/franka/franka_gripper/move')
        self._grasp_client = ActionClient(self, Grasp, '/franka/franka_gripper/grasp')

        self.subscription = self.create_subscription(
            JointState,
            '/so/joints',
            self.listener_callback,
            qos_profile_sensor_data
        )
        self.last_so_pos = None
        self.scale_factor = 2.5
        self.settings = termios.tcgetattr(sys.stdin)
        self.key_timer = self.create_timer(0.1, self.on_press)

        self.opened = 1.0
        #self.last_published_state = 1.0

        self.gripper_pos = 0.0

        # -----------------------------------------------------------

        self.stopped = False

        #self.gripper.enable_torque()

        self.first_data_received = False

        #self.listener = keyboard.Listener(on_press=self.on_press)
       #self.listener.start()
        self.precise_step = 0.001
        self.settings = termios.tcgetattr(sys.stdin)
        self.precise_mode = False
        self.create_timer(0.1, self.on_press)


        self.subscription = self.create_subscription(
            PoseStamped, 
            "/so/pose", 
            self.so_callback, 
            qos_profile_sensor_data,
            callback_group=self.io_group
        )
        self.rate = 1.0 / 30.0
        self.timer = self.create_timer(self.rate, self.update_robot, callback_group=self.control_group)

        self.last_print_time = time.time()
        print("ok2")

    def listener_callback(self, msg):
        
        self.gripper_pos = msg.position[5]
        #print(self.gripper_pos)

        if self.gripper_pos >= 0.5 and self.opened == 0.0:
            # Open
            if not self._move_client.wait_for_server(timeout_sec=1.0):
                return
            goal = Move.Goal()
            goal.width = 0.08
            goal.speed = 0.2
            self._move_client.send_goal_async(goal)
            self.opened = 1.0
        if self.gripper_pos < 0.5 and self.opened == 1.0: 
            # Grasp
            if not self._grasp_client.wait_for_server(timeout_sec=1.0):
                return
            goal = Grasp.Goal()
            goal.width = 0.01
            goal.speed = 0.05
            goal.force = 100.0
            goal.epsilon.inner = 0.005
            goal.epsilon.outer = 0.040
            self._grasp_client.send_goal_async(goal)
            self.opened = 0.0
    
    def publish_state_if_changed(self):
        if self.opened != self.last_published_state:
            msg = Float64()
            msg.data = float(self.opened)
            self.gripper_pub.publish(msg)
            self.get_logger().info(f"Gripper state: {msg.data}")

            self.last_published_state = self.opened
    
    #def open_gripper(self):
        #self.opened = 1.0
        #self.publish_state_if_changed()

    #def close_gripper(self):
        #self.opened = 0.0
        #self.publish_state_if_changed()

    def on_press(self):

        if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
            #tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            
            #finally: 
                #termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

            if key == ' ':
                if self.precise_mode == False:
                    self.precise_mode = True
                    self.get_logger().info("Precise Mode ON!")
                elif self.precise_mode == True:
                    self.precise_mode = False
                    self.get_logger().info("Precise Mode OFF!")

            if self.precise_mode == True:
                if key == "w":
                    self.current_target_pose.position[0] += self.precise_step
                if key == "s":
                    self.current_target_pose.position[0] -= self.precise_step
                if key == "a": 
                    self.current_target_pose.position[1] -= self.precise_step
                if key == "d":
                    self.current_target_pose.position[1] += self.precise_step
                if key == "q":
                    self.current_target_pose.position[2] -= self.precise_step
                if key == "e":
                    self.current_target_pose.position[2] += self.precise_step

                
    
    def so_callback(self, msg: PoseStamped):

        if self.stopped:
            return
        if self.precise_mode == True: 
            current_so_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            self.last_so_pos = current_so_pos

        if self.precise_mode == False:
            current_so_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])

            if self.last_so_pos is None:
                self.last_so_pos = current_so_pos
                self.first_data_received = True
                return

            delta_so = current_so_pos - self.last_so_pos

            if np.linalg.norm(delta_so) > 0.1:
                self.get_logger().warn("Too big delta")
                self.last_so_pos = current_so_pos
                return

            self.current_target_pose.position[0] += delta_so[0] * self.scale_factor
            self.current_target_pose.position[1] += delta_so[1] * self.scale_factor
            self.current_target_pose.position[2] += delta_so[2] * self.scale_factor

            self.current_target_pose.position[2] = max(self.current_target_pose.position[2], 0.02)

            self.last_so_pos = current_so_pos

            wrist_yaw = msg.pose.orientation.z * 1.2

            yaw_rot = Rotation.from_euler("z", wrist_yaw, degrees=False)

            self.target_rotation = self.base_down * yaw_rot
        
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
    teleop = PS5Teleop()
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        rclpy.spin(teleop)
    except KeyboardInterrupt:
        pass
    finally:
        
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        teleop.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
