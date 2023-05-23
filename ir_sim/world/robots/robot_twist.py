import numpy as np
import modern_robotics as mr
import matplotlib as mpl
from .robot_base import RobotBase
from math import sin, cos, atan2, inf, pi
from ir_sim.util.util import WrapToPi

from geometric_utils import *
import matlab.engine
import time
import lcm
import select
from ir_sim.lcm_message.safelcm.state_t import state_t
from ir_sim.lcm_message.safelcm.result_t import result_t

class Polytope:
    def __init__(self, B=np.empty((0, 3)), b=np.empty((0, 1)) ) -> None:
        # initialize np array
        self.B = B
        self.b = b 
    
    def add_hyperplane(self, normal, point):
        self.B = np.vstack([self.B, [-normal[0], -normal[1], 0]])
        self.b = np.vstack([self.b, -normal.dot(point)])

    def clear(self):
        self.B = []
        self.b = []

    @classmethod
    def array_to_polytope(cls, B, b):
        assert B.shape[0] == b.shape[0]
        polytope = Polytope()
        
        for i in range(len(B)):
            polytope.B = np.vstack([polytope.B, B[i, None]])
            polytope.b = np.vstack([polytope.b, b[i, None]])
        return polytope
    
class RobotTwist(RobotBase):

    robot_type = 'twist'  # omni, acker
    appearance = 'polytope'  # shape list: ['circle', 'rectangle', 'polygon']
    state_dim = (3, 1) # the state dimension, x, y, theta(heading direction)
    vel_dim = (2, 1)  # the velocity dimension, linear and angular velocity
    goal_dim = (3, 1) # the goal dimension, x, y, theta
    position_dim=(2,1) # the position dimension, x, y 
    

    def __init__(self, id, points=[ [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]], 
                 state=np.zeros((3, 1)), vel=np.zeros((2, 1)), goal=np.zeros((3, 1)), 
                 radius=0.2, radius_exp=0.1, vel_min=[-2, -2], vel_max=[2, 2], step_time=0.1, acce=[inf, inf], alpha=[0.03, 0, 0, 0.03, 0, 0], **kwargs):

        # shape args
        # open matlab engine
        # eng_name = matlab.engine.find_matlab()
        # if len(eng_name) == 0:
        #     print('Please open matlab engine first!')
        # else:
        #     print('Matlab engine opened!')
        # self.eng = matlab.engine.connect_matlab()          
        # self.eng = matlab.engine.start_matlab()
        # self.eng=self.eng.result()
        # self.eng.addpath(self.eng.genpath('/home/ri-robot/PhD/poly_opt/src/polynomial_optimization_for_robot_control'))


        self.lc = lcm.LCM()
        self.subscription = self.lc.subscribe("SAFETY_FILTER", self.matlab_result_handle)
        # plot a ellipse and sample into N points
        # a = 1.0
        # b = 0.3
        # x = np.linspace(-a, a, 3)
        # y = np.sqrt(b*(1 - x**2/a))
        # x = np.hstack([x, -x])
        # y = np.hstack([y, -y])
        # points = np.vstack([x, y]).T.tolist()
        # mpl.pyplot.scatter(points[:,0], points[:,1])
        # mpl.pyplot.show()
        # exit()
        points = [[-0.7, 0.5], [-0.7, -0.5], [0.3, -0.5], [0.8, 0.0], [0.3, 0.5]]
        points.append(points[0])
        self.points = np.array(points)
        print(self.points.shape)
        self.init_vertex = self.points[:len(points)-1]
        self.poly = get_convex_hull(points)
        self.polytopes = Polytope()     # body's poly in its own frame
        hyperplanes = self.poly.hyperplanes()
        for hp in hyperplanes:
            print(hp.n_, hp.p_)
            self.polytopes.add_hyperplane(hp.n_, hp.p_)

        self.radius = radius
        self.radius_collision = radius + radius_exp
        self.shape = radius
        super(RobotTwist, self).__init__(id, state, vel, goal, step_time, vel_min=vel_min, vel_max=vel_max, acce=acce, **kwargs)
        
        self.vel_omni = np.zeros((2, 1))

        if self.noise:
            self.odometry = {'mean': self.state, 'std': np.array([[0.04], [0.01]])}   # odometry
        else:
            self.odometry = {'mean': self.state, 'std': np.array([[0], [0]])}   # odometry

        self.alpha = alpha
    
    def matlab_result_handle(self, channel, data):
        msg = result_t.decode(data)
        self.result = (msg.v_safe, msg.theta_safe, msg.success)

    def dynamics(self, state, vel,  **kwargs):
        # The differential-wheel robot dynamics
        # reference: Probability robotics, motion model
        # print(self.goal)
        # print(state)
        temp_state = self.stateToTransformationMatrix(state)
        temp_goal = self.stateToTransformationMatrix(self.goal)
        ref_ctrl, _, _  = self.cal_des_twist(temp_state, temp_goal)
        ref_ctrl = ref_ctrl.reshape(6,1)
        _, safe_poly = self.decomp_utils()

        safe_region_poly = Polytope()   # safe region's poly in world frame
        hyperplanes = safe_poly.hyperplanes()
    
        for hp in hyperplanes:
            safe_region_poly.add_hyperplane(-hp.n_, hp.p_)

        safe_region_poly = self.rep_robot_in_world(mr.TransInv(temp_state), safe_region_poly) # safe region's poly in body frame
        start_time = time.time()
        self.pub_matlab_request(self.polytopes, safe_region_poly, ref_ctrl,1)
        end_time = time.time()
        print('time ', end_time - start_time)
        # S_filtered, theta_safe, is_success = self.safety_filter(self.polytopes, safe_region_poly, ref_ctrl, 0)
        S_filtered, theta_safe, is_success = self.result
        S_filtered = np.array(S_filtered)

        # temp_vel = np.array([0, 0, vel[1][0], vel[0][0], 0, 0])
        if is_success == 0:
            print('No solution!')
            return state
        else:
            new_state = RobotTwist.motion_twist(self.stateToTransformationMatrix(state), S_filtered*theta_safe, self.step_time)
            new_state = self.transformationMatrixToState(new_state)
            self.vel = vel
            self.vel_omni = RobotTwist.diff_to_omni(state, self.vel)

            if self.noise:
                self.odometry['mean'] = RobotTwist.motion_diff(self.odometry['mean'], vel, self.step_time, False, self.alpha, **kwargs) 
            else:
                self.odometry['mean'] = self.state

            return new_state
    
    def stateToTransformationMatrix(self, state):
        # state: 3*1
        # return: 4*4 transformation matrix
        rotate_theta = state[2,0]
        # rotate rotate_theta along z axis
        rotate_matrix = np.array([[cos(rotate_theta), -sin(rotate_theta), 0],
                                  [sin(rotate_theta), cos(rotate_theta), 0],
                                  [0, 0, 1]])
           

        return mr.RpToTrans(rotate_matrix,np.array([state[0,0],state[1,0],0]))
    
    def transformationMatrixToState(self, matrix):
        # matrix: 4*4
        # return: 3*1 state
        rotate_matrix = matrix[0:3,0:3]
        # rotate matrix to rpy
        rotate_theta = atan2(rotate_matrix[1,0],rotate_matrix[0,0])
        return np.array([[matrix[0,3]],[matrix[1,3]],[rotate_theta]])

    def cal_des_twist(self, Tsb, Tsgoal):
        # calculate error twist between two 4*4 configuration Tsb and Tsgoal
        # return: 6*1 twist
        # Tsb: 4*4
        # Tsgoal: 4*4
        # Tsb = self.stateToTransformationMatrix(self.state)
        # Tsgoal = self.stateToTransformationMatrix(self.goal)

        T_err = mr.TransInv(Tsb) @ Tsgoal

        Vb = mr.se3ToVec(mr.MatrixLog6(T_err))
        [S_error, theta_error] = mr.AxisAng6(Vb)
        return Vb, S_error, theta_error
    
    def pub_matlab_request(self, robot, free_region, V_nominal, flag):
        print('pub robot info')
        msg = state_t()
        msg.num_hyper_robot = robot.B.shape[0]
        # flatten matrix and convert to list, then padding zero
        msg.robot_B = robot.B.flatten(order='F').tolist() + [0] * (30 - robot.B.shape[0] * 3)
        msg.robot_b = robot.b.flatten(order='F').tolist() + [0] * (10 - robot.b.shape[0])
        msg.num_hyper_convex = free_region.B.shape[0]
        msg.hyper_B = free_region.B.flatten(order='F').tolist() + [0] * (90 - free_region.B.shape[0] * 3)
        msg.hyper_b = free_region.b.flatten(order='F').tolist() + [0] * (30 - free_region.b.shape[0])
        msg.v = V_nominal.flatten(order='F').tolist()
        msg.flag = bool(flag)
        print("before publish",time.time())
        self.lc.publish("ENV_INFO", msg.encode())
        print("after publish",time.time())
        self.lc.handle()

    def safety_filter(self, robot, free_region, V_nominal, flag):
        #  Input: 
        #  robot:polytope H-representation for robot in its own frame
        #  free_region: polytope H-representation for current free_region in body frame
        #  V_nominal: nominal twist for current robot, [w;v];
        #  flag: 0 represent for pure translation optimization while 1 represent
        #  control with rotation.(For parallel computing)
        #  Output:
        #  S_filtered: Filtered screw axis
        #  theta_safe: safe distance travel along the S_filterd
        #  success: 0/1 represent for if the optimization problem is solved
        # B_array = np.array([-0.1066,-0.0685,0,0.1801,-0.2805,0,0.0533,0.0342,0,-0.090,0.140,0]).reshape(4,3)
        # b_array = np.array([0.4943,0.6764,0.2597,0.1618]).reshape(4,1)

        start_time = time.time()
        filtered = self.eng.planer_safety_filter(robot.B,robot.b,free_region.B,free_region.b,V_nominal, flag)
        print('matlab time: ', time.time() - start_time)
        # S_filtered, theta_safe, success = self.eng.planer_safety_filter(robot.B,robot.b,B_array,b_array,V_nominal, flag)
        success = int(filtered[0][-1])
        theta_safe = filtered[0][-2]
        S_filtered = np.array(filtered[0][0:6])[0]

        return S_filtered, theta_safe, success
    
    @classmethod
    def motion_twist(cls, Tsb, Twist, step_time):
        # Tsb: 4*4
        # Body Twist: 6*1
        # step_time: float
        # noise: bool
        # return: 4*4 next_Tsb represent body frame in world frame
        # print('Tsb ', Tsb)
        next_Tsb = Tsb @  mr.MatrixExp6(mr.VecTose3(Twist*step_time))
        # print(next_Tsb)
        return next_Tsb
        
    def rep_robot_in_world(self, Tsb, polytope_robot):
        # polytope_robot: polytope wrapping robot, expressed in body frame, always the same in its own frame.
        # this function is to represent the polytope_robot in world frame for visulization
        # Tsb is the current configuration of robot in world frame
        Tbs = mr.TransInv(Tsb)
        B = polytope_robot.B
        b = polytope_robot.b
        B_w = B @ Tbs[0:3,0:3]
        b_w = b - (B @ Tbs[0:3,3].reshape(3,1))

        # For plot the polytope in world frame
        return Polytope.array_to_polytope(B_w,b_w)
        

    def cal_des_vel(self, tolerance=0.12):
        # calculate desire velocity
        des_vel = np.zeros((2, 1))

        if self.arrive_mode == 'position':

            dis, radian = RobotTwist.relative_position(self.state, self.goal)      

            if dis < self.goal_threshold:
                return des_vel
            else:
                diff_radian = RobotTwist.wraptopi( radian - self.state[2, 0] )
                des_vel[0, 0] = np.clip(self.vel_acce_max[0, 0] * cos(diff_radian), 0, inf) 

                if abs(diff_radian) < tolerance:
                    des_vel[1, 0] = 0
                else:
                    des_vel[1, 0] = self.vel_acce_max[1, 0] * (diff_radian / abs(diff_radian))

        elif self.arrive_mode == 'state':
            pass
        
        return des_vel
    
    def gen_inequal_global(self):
        # generalized inequality, inside: Gx <=_k g, norm2 cone  at current position
        G = np.array([ [1, 0], [0, 1], [0, 0] ])
        h = np.row_stack((self.center, -self.radius * np.ones((1,1))))

        return G, h

    def plot_robot(self, ax, robot_color = 'g', goal_color='r', 
                   show_polygon=False,
                    show_goal=True, show_text=False, show_traj=False, 
                    show_uncertainty=False, traj_type='-g', fontsize=10, 
                    arrow_width=0.6, arrow_length=0.4, **kwargs):
        x = self.state[0, 0]
        y = self.state[1, 0]
        yaw = -self.state[2, 0]

        ego_vertices = np.array(self.points) @ np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        ego_vertices += np.array([x, y])
        
        goal_x = self.goal[0, 0]
        goal_y = self.goal[1, 0]

        robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = robot_color)
        robot_circle.set_zorder(3)
        robot_poly = mpl.patches.Polygon(ego_vertices, color='green', fill=False, alpha=0.5)
        robot_poly.set_zorder(4)

        ax.add_patch(robot_circle)
        ax.add_patch(robot_poly)
        if show_text: 
            r_text = ax.text(x - 0.5, y, 'r'+ str(self.id), fontsize = fontsize, color = 'r')
            self.plot_text_list.append(r_text)
        self.plot_patch_list.append(robot_poly)
        self.plot_patch_list.append(robot_circle)
        
        if show_polygon:
            vertices, _ = self.decomp_utils()
            convex_poly = mpl.patches.Polygon(vertices, color='orange', fill=False, alpha=0.5)
            convex_poly.set_zorder(2)

            ax.add_patch(convex_poly)
            self.plot_patch_list.append(convex_poly)

        # arrow
        theta = self.state[2][0]
        arrow = mpl.patches.Arrow(x, y, arrow_length*cos(theta), arrow_length*sin(theta), width = arrow_width)
        arrow.set_zorder(3)
        ax.add_patch(arrow)
        self.plot_patch_list.append(arrow)

        if show_goal:
            goal_circle = mpl.patches.Circle(xy=(goal_x, goal_y), radius = self.radius, color=goal_color, alpha=0.5)
            goal_circle.set_zorder(1)
        
            ax.add_patch(goal_circle)
            if show_text: 
                g_text = ax.text(goal_x + 0.3, goal_y, 'g'+ str(self.id), fontsize = fontsize, color = 'k')
                self.plot_text_list.append(g_text)

            self.plot_patch_list.append(goal_circle)

        if show_traj:
            x_list = [t[0, 0] for t in self.trajectory]
            y_list = [t[1, 0] for t in self.trajectory]
            self.plot_line_list.append(ax.plot(x_list, y_list, traj_type))

        if show_uncertainty and self.noise:
            scale = 20
           
            ex = self.odometry['mean'][0, 0]
            ey = self.odometry['mean'][1, 0]
            etheta = self.odometry['mean'][2, 0]
            std_x = self.odometry['std'][0, 0]
            std_y = self.odometry['std'][1, 0]

            angle = etheta * (180 / pi)

            ellipse = mpl.patches.Ellipse(xy=(ex, ey), width=scale*std_x, height=scale*std_y, angle=angle, facecolor='gray', alpha=0.8)
            ellipse.set_zorder(1)
            ax.add_patch(ellipse)
            self.plot_patch_list.append(ellipse)


    @staticmethod
    def diff_to_omni(state, vel_diff):
        vel_linear = vel_diff[0, 0]
        theta = state[2, 0]
        vx = vel_linear * cos(theta)
        vy = vel_linear * sin(theta)
        return np.array([[vx], [vy]])




        
    
