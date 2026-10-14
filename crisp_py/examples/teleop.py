import numpy as np
import time
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from crisp_py.robot import Pose, Robot
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped

from crisp_py.robot import make_robot

from crisp_py.gripper.gripper import Gripper, GripperConfig

class PS5Teleop(Node):
    def __init__(self):
        super().__init__("ps5_teleop")

        # === Initialize robot (same as sporadic_poses.py) ===
        self.robot = make_robot("fr3", namespace="franka")
        self.robot.wait_until_ready()

        #self.get_logger().info("Homing robot...")

        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

        self.robot.set_target_joint(self.robot.joint_values)

        self.subscription = self.create_subscription(Joy, "/joy", self.joy_callback, 10)

        self.pose = self.robot.end_effector_pose.copy()

        self.step = 0.005
        self.rota = 0.01

        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0

        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0

        self.timer = self.create_timer(1.0 / 20.0, self.update_robot)

        self.get_logger().info("PS5 teleop initialized. Use left stick + triggers to move.")

    def joy_callback(self, msg: Joy):

        self.dy = msg.axes[0] * self.step
        self.dx = msg.axes[1] * self.step
        self.dz = msg.axes[4] * self.step

        if msg.axes[2] != 1:
            self.rx = -msg.axes[2] * self.rota
        if msg.axes[5] != 1:
            self.rx = msg.axes[5] * self.rota
        self.ry = msg.axes[7] * self.rota
        self.rz = msg.axes[6] * self.rota

        if len(msg.buttons) > 1 and msg.buttons[1] == 1:
            self.get_logger().info("shutting down.")
            self.robot.shutdown()
            rclpy.shutdown()
        
    def update_robot(self):
        try:
            self.pose.position += np.array([self.dx, self.dy, self.dz])
            rot = Rotation.from_euler("xyz", [self.rx, self.ry, self.rz], degrees=False)
            self.pose.orientation = self.pose.orientation * rot
            self.robot.set_target(pose=self.pose)
            self.rx = 0.0
            self.ry = 0.0
            self.rz = 0.0
        except AttributeError:
            pass

def main():
    rclpy.init()
    teleop = PS5Teleop()
    rclpy.spin(teleop)
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
