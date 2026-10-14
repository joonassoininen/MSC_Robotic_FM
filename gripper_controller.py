import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from franka_msgs.action import Move, Grasp
from std_msgs.msg import Float32

class FrankaGripperController(Node):
    def __init__(self):

        super().__init__('franka_gripper_controller')

        self._move_client = ActionClient(self, Move, "/franka/franka_gripper/move")
        self._grasp_client = ActionClient(self, Grasp, '/franka/franka_gripper/grasp')

        self.subscription = self.create_subscription(
            Float32,
            "/gripper_command",
            self.gripper_callback,
            10
        )

        self.is_grasping = False
        self.current_goal = None



        print("Initialized!")

    def gripper_callback(self, msg):
        
        if msg.data >= 0.5 and self.is_grasping == False:
            self.cancel_current()
            self.open()
        if msg.data < 0.5 and self.is_grasping == True:
            self.cancel_current()
            self.grasp()

    def cancel_current(self):
        if self.current_goal is not None:
            self.current_goal.cancel_goal_async()
            self.current_goal = None

    def grasp(self):
        
        goal = Grasp.Goal()
        goal.width = 0.01
        goal.speed = 0.5
        goal.force = 20.0
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.040
        future = self._grasp_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
        self.is_grasping = True
    
    def open(self):

        goal = Move.Goal()
        goal.width = 0.08
        goal.speed = 0.5
        future = self._move_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
        self.is_grasping = False

    def goal_response_callback(self, future):
        self.current_goal = future.result()

def main():
    rclpy.init()
    gripper = FrankaGripperController()

    try: 
        rclpy.spin(gripper)
    except KeyboardInterrupt:
        pass
    finally: 
        gripper.destroy_node()
        rclpy.shutdown()
if __name__=="__main__":
    main()



