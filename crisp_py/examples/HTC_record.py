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
from rclpy.action import ActionClient
from franka_msgs.action import Move, Grasp
import copy

# ----------------------------------------------
#
# Teleoperation used on data gathering
#
# ----------------------------------------------

class HTCTeleop(Node):
    def __init__(self):
        super().__init__("HTC_teleop")
        
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.io_group = ReentrantCallbackGroup()

        time.sleep(1.0)

        # Initialize the pose topics for the teleoperation
        self.pose = None
        self.current_target_pose = None

        self.base_down = Rotation.from_quat([0.99999999, 0.00004908, 0.0, 0.0])
        self.target_rotation = self.base_down 

        self.scale_factor = 1.0
        self.settings = termios.tcgetattr(sys.stdin)

        # Gripper control ROS2 Actions
        self._move_client = ActionClient(self, Move, '/franka/franka_gripper/move')
        self._grasp_client = ActionClient(self, Grasp, '/franka/franka_gripper/grasp')
        
        self.opened = 1.0
        self.gripper_pos = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Comment the unused controller and uncomment the used one
        self.from_frame = 'libsurvive_world'
        #self.to_frame = 'LHR-F7F7BFC5'
        self.to_frame = 'LHR-F5279FC2'

        self.last_HTC_pos = None 
        self.last_rot = self.base_down

        # Teleoperation loop timer
        self.vive_timer = self.create_timer(
            1.0 / 30.0, 
            self.vive_tf_loop, 
            callback_group=self.io_group
        )

        # Subscription for the wand buttons
        self.joy_sub = self.create_subscription(
            Joy, 
            "/libsurvive/joy",
            self.joy_callback,
            10
        )

        # Subscription for FR3 end effector position 
        self.subscription = self.create_subscription(
            PoseStamped,
            "/franka/franka_robot_state_broadcaster/current_pose",
            self.franka_callback,
            qos_profile_sensor_data
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

        # Gripper and pose action publishers for streamed teleop data
        self.pose_pub = self.create_publisher(PoseStamped, "/teleop_pose", 10)
        self.gripper_pub = self.create_publisher(Float32, "/teleop_gripper", 10)

        self.last_scaled_robot_rotation = None

        self.last_rot = None

        self.last_gripper_button = -1

        print("Teleoperation Initialized!")

    def franka_callback(self, msg): 
        
        # Stores the franka end effector pose for initialization

        self.pose = msg.pose

        if self.current_target_pose is None:
            self.current_target_pose = copy.deepcopy(msg.pose)
    
    def joy_callback(self, msg):
        
        # Calibration button
        if len(msg.buttons) > 6 and msg.buttons[6] == 1:
            self.calibrate_from_imu()
        # Dead man switch button
        if msg.buttons[17] == 1:
            self.stopped = False
        if msg.buttons[17] == 0:
            self.stopped = True

        current = msg.buttons[0]

        # Gripper controlling 
        if current != self.last_gripper_button:
            if current == 0:
                self.open_gripper()
            if current == 1:
                self.close_gripper()
            self.last_gripper_button = current

    def open_gripper(self):

        # ROS2 Action for opening the gripper

        goal = Move.Goal()
        goal.width = 0.08
        goal.speed = 0.2
        self._move_client.send_goal_async(goal)
        self.opened = 1.0

    def close_gripper(self):

        # ROS2 action for closing the gripper

        goal = Grasp.Goal()
        goal.width = 0.01
        goal.speed = 0.05
        goal.force = 20.0
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.040
        self._grasp_client.send_goal_async(goal)
        self.opened = 0.0       

        
    
    def calibrate_from_imu(self):

        # Controller calibration

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
        self.last_HTC_pos = None
    
    def vive_tf_loop(self):

        # Teleoperation loop        

        # If the robot end effector pose has not receivied, the loop doesn't continue
        if self.current_target_pose is None:
            return
        
        # Controller needs to be calibrated at least once before the loop begins
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
            
            # Stores the position of the wand controller
            current_HTC_pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])

            # Stores the rotation of the wand controller
            current_rot = Rotation.from_quat([
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w
            ])
            
 
            #if self.precise_mode == True: 
            #    self.last_HTC_pos = current_HTC_pos
 
            # The loop begins only when two consecutive controller poses are received
            # as the teleoperation utilizes relative position of the wand controller
            if self.last_HTC_pos is None:
                self.last_HTC_pos = current_HTC_pos
                self.last_rot = current_rot
                self.first_data_received = True
                return
 
            # difference between two consecutive wand controller poses
            delta_world = current_HTC_pos - self.last_HTC_pos
 
            # Safety functionality
            if np.linalg.norm(delta_world) > 0.05:

                # If the controller drops or pose calculations bug the robot does not move

                self.get_logger().warn("Too big delta in Vive tracking!")
                self.last_HTC_pos = current_HTC_pos
                self.last_rot = current_rot
                return
            
            delta_robot = np.dot(self.world_to_robot_matrix, delta_world)
            
            # If dead man switch button is not pressed, the robot doesnt move
            if self.stopped == True:
                    delta_robot[0] = 0.0
                    delta_robot[1] = 0.0
                    delta_robot[2] = 0.0
                    self.last_HTC_pos = current_HTC_pos
            
            # Stores the relative controller actions into puvlished variable
            self.current_target_pose.position.x += delta_robot[0] * 1.4
            self.current_target_pose.position.y += delta_robot[1] * 1.4
            self.current_target_pose.position.z += delta_robot[2] * 1.5
 
            self.last_HTC_pos = current_HTC_pos
 
            # Calculations for relative rotation controlling
            delta_rot_world = current_rot * self.last_rot.inv()
            r_matrix = Rotation.from_matrix(self.world_to_robot_matrix)
            delta_rot_robot = r_matrix * delta_rot_world * r_matrix.inv()
 
            times = [0.0, 1.0]
            key_rotations = Rotation.from_quat([
                Rotation.identity().as_quat(),
                delta_rot_robot.as_quat()
            ])

            # Controller rotations are slightly reduced because the wand controller 
            # rotations happened to be slightly aggressive without reducement
            slerp = Slerp(times, key_rotations)
            scaled_delta = slerp(0.5)

            # If dead man switch button is not pressed, the robot rotation doesn't change
            if self.stopped == False:
                self.target_rotation = scaled_delta * self.target_rotation
 
            self.last_rot = current_rot
 
        except TransformException as ex:
            pass
        
    def update_robot(self):
        """
            Function publishes the target end effector pose 
            to the streamed teleop topics
        """

        if not self.first_data_received:
            return
        
        try:
            quat = self.target_rotation.as_quat()

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "teleop"

            msg.pose.position.x = self.current_target_pose.position.x
            msg.pose.position.y = self.current_target_pose.position.y
            msg.pose.position.z = self.current_target_pose.position.z

            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])

            msg2 = Float32()
            msg2.data = self.opened

            self.pose_pub.publish(msg)
            self.gripper_pub.publish(msg2)
        
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