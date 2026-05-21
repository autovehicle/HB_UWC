import json
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuitFromJson(Node):
    def __init__(self):
        super().__init__("pure_pursuit_from_json")

        self.declare_parameter(
            "json_path",
            "/mnt/c/Users/autonav009/Desktop/HB_UWC/generated_paths/nav_path.json",
        )
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("wheel_cmd_topic", "/uwc/wheel_cmd")

        self.declare_parameter("lookahead_distance", 0.8)
        self.declare_parameter("linear_speed", 0.2)
        self.declare_parameter("wheel_base", 0.6)
        self.declare_parameter("command_scale", 15.0)
        self.declare_parameter("max_wheel_cmd", 80.0)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("angular_sign", 1.0)

        self.json_path = self.get_parameter("json_path").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.wheel_cmd_topic = self.get_parameter("wheel_cmd_topic").value

        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.wheel_base = float(self.get_parameter("wheel_base").value)
        self.command_scale = float(self.get_parameter("command_scale").value)
        self.max_wheel_cmd = float(self.get_parameter("max_wheel_cmd").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.angular_sign = float(self.get_parameter("angular_sign").value)

        self.waypoints = self.load_path_json(self.json_path)

        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.goal_reached = False

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            self.wheel_cmd_topic,
            10,
        )

        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f"Loaded path JSON: {self.json_path}")
        self.get_logger().info(f"Waypoint count: {len(self.waypoints)}")
        self.get_logger().info(f"Sub odom : {self.odom_topic}")
        self.get_logger().info(f"Pub wheel: {self.wheel_cmd_topic}")

    def load_path_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = data.get("points", [])

        if len(points) < 2:
            raise RuntimeError("nav_path.json 안에 waypoint가 2개 미만입니다.")

        waypoints = [
            (
                float(p["x"]),
                float(p["y"]),
                float(p.get("z", 0.0)),
            )
            for p in points
        ]

        return waypoints

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def publish_stop(self):
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0]
        self.cmd_pub.publish(msg)

    def distance_to(self, wp):
        return math.hypot(wp[0] - self.current_x, wp[1] - self.current_y)

    def find_lookahead_point(self):
        closest_idx = min(
            range(len(self.waypoints)),
            key=lambda i: self.distance_to(self.waypoints[i]),
        )

        for i in range(closest_idx, len(self.waypoints)):
            if self.distance_to(self.waypoints[i]) >= self.lookahead_distance:
                return self.waypoints[i]

        return self.waypoints[-1]

    def clip_cmd(self, value):
        return max(-self.max_wheel_cmd, min(self.max_wheel_cmd, value))

    def control_loop(self):
        if self.current_x is None or self.current_y is None or self.current_yaw is None:
            return

        if len(self.waypoints) < 2:
            return

        goal = self.waypoints[-1]
        goal_dist = self.distance_to(goal)

        if goal_dist <= self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info("Goal reached. Stop.")
                self.goal_reached = True
            self.publish_stop()
            return

        target = self.find_lookahead_point()

        dx = target[0] - self.current_x
        dy = target[1] - self.current_y

        # world 좌표 오차를 robot local 좌표로 변환
        local_x = math.cos(self.current_yaw) * dx + math.sin(self.current_yaw) * dy
        local_y = -math.sin(self.current_yaw) * dx + math.cos(self.current_yaw) * dy

        if local_x < 0.0:
            local_x = 0.001

        lookahead_sq = max(local_x * local_x + local_y * local_y, 1e-6)
        curvature = 2.0 * local_y / lookahead_sq

        v = self.linear_speed
        w = self.angular_sign * v * curvature

        left_vel = v - (w * self.wheel_base / 2.0)
        right_vel = v + (w * self.wheel_base / 2.0)

        left_cmd = self.clip_cmd(left_vel * self.command_scale)
        right_cmd = self.clip_cmd(right_vel * self.command_scale)

        msg = Float64MultiArray()

        # UWC 모델 기준:
        # 직진은 [+, +, -, -]
        msg.data = [
            left_cmd,
            left_cmd,
            -right_cmd,
            -right_cmd,
        ]

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitFromJson()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()