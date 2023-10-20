import math
import numpy as np
from gym import spaces
from gym.utils import seeding
import copy

import cv2 #for camera stuff
import time

import sys
from os.path import join, dirname, abspath

mypath = join(dirname(abspath(__file__)), "../../../")
sys.path.append(mypath)
# sys.path.append("/home/jack/Documents/pyrfuniverse/")
import pyrfuniverse.attributes as attr
from pyrfuniverse.envs import RFUniverseGymWrapper

from pyrfuniverse.utils.controller import RFUniverseController


class FrankaClothHangEnv(RFUniverseGymWrapper):
    metadata = {'render.modes': ['human', 'rgb_array']}
    scale = 8
    y_bound = 3.8
    franka_init_y = 4.8
    object2id = {
        'franka': 965874,
        'target': 1000,
        'cloth': 89212,
        'gripper': 9658740,
        'camera': 123456,
        'egocamera': 9658741,
        'agent': 1234
    }

    def __init__(
            self,
            with_force_zone=False,
            force_zone_intensity=1,
            force_zone_turbulence=10,
            max_steps=300,

            cloth_init_pos_min=(-4, 4, -3),
            cloth_init_pos_max=(0, 6, 0),
            executable_file=None,
            assets=['HangingClothSolver', 'Camera', "my_franka_prefab", "myfrankahand", "robotiqfranka"],
    ):
        super().__init__(
            executable_file=executable_file,
            assets=assets
        )
        self.with_force_zone = with_force_zone
        self.force_zone_intensity = force_zone_intensity
        self.force_zone_turbulence = force_zone_turbulence
        self.max_steps = max_steps
        self.init_agent_pos_max = np.array([0, 6, 1])
        self.init_agent_pos_min = np.array([-2, 5, -1])

        self.init_pos = [-1.85, 6, 0]#[0.15, 0.3, 0] * self.scale

        self.gripper = "franka" #"robotiq" or "franka"

        self.cloth_init_pos_min = np.array(cloth_init_pos_min)
        self.cloth_init_pos_max = np.array(cloth_init_pos_max)
        self.cloth_init_rot_min = np.array([0, -90, 0])
        self.cloth_init_rot_max = np.array([0, -70, 0])

        self.cloth_init_pos = (-5, 6, 0)
        self.reset_step_count = 300
        self.tolerance = 0.2
        self.resolution = 64

        self.last_pos = []
        self.seed()
        
        self.prev_joint_positions = np.array([0.0 for i in range(8)], dtype=float)
        if self.gripper == "robotiq":
            self.controller = RFUniverseController("franka", robot_urdf=join(dirname(abspath(__file__)), "../../../URDF/franka_robotiq/franka_robotiq.urdf"))
        else:
            self.controller = RFUniverseController("franka", robot_urdf="franka_panda/panda.urdf")

        self.eef_orn = self.controller.bullet_client.getQuaternionFromEuler(
            [0, math.pi/2, 0]) #-60,0,90
        
        self._load_cloth()
        self._load_agent()
        self.t = 0
        self.total_t = 0
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        obs = self._get_obs()
        self.init_eef_rot = obs['observation'][3:6]
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=obs['observation'].shape, dtype=np.float32),
            'image' : spaces.Box(0.0, 1.0, shape=(self.resolution, self.resolution, 3), dtype=np.float32),
            'ego' : spaces.Box(0.0, 1.0, shape=(self.resolution, self.resolution, 3), dtype=np.float32),

            # 'depth' : spaces.Box(0.0, 1.0, shape=(self.resolution, self.resolution), dtype=np.float32),
        })

        self.target_grabbed = False

        self.instance_channel.set_action(
            "SetTransform",
            id=self.object2id['camera'],
            position=list([-2.5,2.2,-8])
        )

    def step(self, action: np.ndarray):
        """
        Params:
            action: 4-d numpy array. -> move 3D and grab y/n
        """

        pos_ctrl = action[:3] * 0.2
        curr_pos = self._get_eef_position() * self.scale
        pos_ctrl = curr_pos + pos_ctrl
        #add bounds for actions
        pos_ctrl[0] = max(min(pos_ctrl[0], 0), -10)
        pos_ctrl[1] = max(min(pos_ctrl[1], 7), 3)
        pos_ctrl[2] = max(min(pos_ctrl[2], 2), -2)

        if self.gripper == "robotiq":
            joint_positions = self.controller.calculate_ik_recursive(pos_ctrl/self.scale, eef_orn=self.eef_orn, end_effector_id=8)
        else:
            joint_positions = self.controller.calculate_ik_recursive(pos_ctrl/self.scale, eef_orn=self.eef_orn)


        joint_positions.append(float(1))
        velocities = [0.25 for i in range(8)]
        a = np.array(joint_positions + velocities)
        
        self._set_franka_joints(a)
        self._wait_for_moving()
        self._update_joint_positions()

        grab_conf = action[-1]
        # if really confident, grab/release. else do nothing
        if(grab_conf < -0.5):
            self._set_gripper_width(0.4)
        elif(grab_conf > 0.5):
            self._set_gripper_width(0.01)

        self._step()

        self.t += 1
        self.total_t += 1

        done = False
        obs = self._get_obs()
        end_target = self._get_cloth_position()
        end_target[0] += 0.4
        achieved_goal = pos_ctrl / self.scale
        ccorner_target = self._get_cloth_corner_positions()[3:]
        ep_success = self._distance(achieved_goal, end_target) < (0.05) and self._distance(ccorner_target, end_target) < (0.05)

        reward = self._distance(achieved_goal, end_target) < (0.1) and self._distance(ccorner_target, end_target) < (0.1)
        success = self._check_success(obs['observation'])
        if self.t == self.max_steps:
            done = True
        info = {
            'is_success': success
        }

        return obs, reward, done, info



    def reset(self):
        self.SetTimeScale(5)
        self._destroy_cloth()
        self._destroy_agent()
     
        self.t = 0
        self.target_grabbed = False

        self._load_cloth()
        self._load_agent()
        self.SetTimeScale(1)
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def rerender(self):
        self.instance_channel.set_action(
            'GetRGB',
            id=self.object2id['camera'],
            width=256,
            height=256,
            fov=42
        )

        self._step() #requires a step to generate images
        image_np = np.frombuffer(self.instance_channel.data[self.object2id['camera']]['rgb'], dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np.copy()
    def render(self, mode='human'):
        self._step()

    def heuristic(self):
        reward = -self.t
        action = []
        delta = 0.2
        corner_target = self._get_cloth_corner_positions()[3:]
        joint_angles = self._get_joint_angles()
        target_position = []
        if(self.target_grabbed):
            target_position = self._get_cloth_position() * self.scale
            target_position[0] += (0.4 * self.scale)
            i = 0
        else:
            target_position = (corner_target) * self.scale
        curr_pos = self._get_eef_position() * self.scale

        pos_diff = (target_position - curr_pos)
        pos_diff = np.clip(pos_diff, -1, 1)
        pos_ctrl = curr_pos + (pos_diff * delta) #keep same scale as agent
        
        if self.gripper == "robotiq":
            joint_positions = self.controller.calculate_ik_recursive(pos_ctrl/self.scale, eef_orn=self.eef_orn, end_effector_id=8)
        else:
            joint_positions = self.controller.calculate_ik_recursive(pos_ctrl/self.scale, eef_orn=self.eef_orn)

        joint_positions.append(float(1))

        velocities = [0.25 for i in range(8)]
        a = np.array(joint_positions + velocities)
        self._set_franka_joints(a)
        self._wait_for_moving()
        self._update_joint_positions()

        grab_conf = -1#(2 * np.random.rand(1) - 1) #-1
        distance = self._distance(target_position, pos_ctrl)
        if(distance < (self.tolerance*0.75)):
            grab_conf = 1 # self.target_grabbed #do not regrab
        grab_conf += self.target_grabbed
        grab_conf = min(grab_conf, 1)
        # if really confident, grab/release. else do nothing
        if(grab_conf < -0.5):
            self._set_gripper_width(0.4)
        elif(grab_conf > 0.5):
            self._set_gripper_width(0.01)

        self._step()

        curr_joint_angles = self._get_joint_angles()
        joint_angle_diff = (curr_joint_angles - joint_angles)
        action = np.asarray(pos_diff)
        #action = np.interp(joint_angle_diff, (-3.141592e-3, 3.141592e-3), (-1, 1))#scale actions
        # action = self._get_joint_velocities()
        action = np.append(action, grab_conf)

        self.target_grabbed = self._get_grabbed()

        achieved_goal = pos_ctrl / self.scale

        reward = 0#self._compute_reward2() + (0.1*self.target_grabbed) + -self._distance(corner_target, achieved_goal)

        self.t += 1
        success = (distance < self.tolerance*self.scale)

        end_target = self._get_cloth_position()
        end_target[0] += 0.4
        achieved_goal = pos_ctrl / self.scale
        ep_success = self._distance(achieved_goal, end_target) < (0.05)
        done = self.t >= self.max_steps or ep_success

        reward = self._distance(achieved_goal, end_target) < 0.1 #inflate target area for reward

        obs = self._get_obs()
        image = obs['image']
        depth = obs['ego']
        return obs['observation'], depth, image, action, reward, done

    def _get_gripper_width(self) -> float:
        gripper_joint_positions = copy.deepcopy(self.instance_channel.data[self.object2id['gripper']]['joint_positions'])
        return [(gripper_joint_positions[0] + gripper_joint_positions[1]) / self.scale]

    def _set_gripper_width(self, w: float):
        if self.gripper == "robotiq":
            w = 0 if w > 0.04 else 50
            self.instance_channel.set_action(
                'SetJointPosition',
                id=self.object2id['gripper'],
                joint_positions=[w, w],
            )
        else:
            w = (w / 2) * self.scale
            self.instance_channel.set_action(
                'SetJointPosition',
                id=self.object2id['gripper'],
                joint_positions=[w, w],
            )

    def _set_franka_joints(self, a: np.ndarray):
        self.instance_channel.set_action(
            "SetJointPosition",
            id=self.object2id['franka'],
            joint_positions=list(a[0:7]),
            speed_scales=list(a[8:15]),
        )
        self._step()

    def _wait_for_moving(self):
        while not (
            self.instance_channel.data[self.object2id['franka']]["all_stable"]
            and self.instance_channel.data[self.object2id['gripper']]["all_stable"]
        ):
            self._step()

    def _update_joint_positions(self):
        data = self.instance_channel.data
        arm_joint_positions = data[self.object2id['franka']]["joint_positions"]
        gripper_joint_positions = data[self.object2id['gripper']]["joint_positions"]

        self.prev_joint_positions[0:7] = np.array(arm_joint_positions)
        self.prev_joint_positions[7] = abs(gripper_joint_positions[0]) + abs(
            gripper_joint_positions[1]
        )

    def _get_obs(self):
        catcher_position = self._get_eef_position()
        cloth_position = self._get_cloth_position()

        joint_angles = []#self._get_joint_angles()

        gripper_width = self._get_gripper_width()
        grabbed = []# np.asarray([self._get_grabbed()], dtype=np.float32)
        image = self.get_noised_image('camera')
        ego_image = self.get_noised_image('egocamera')

        obs = np.concatenate((catcher_position, joint_angles, gripper_width, cloth_position, grabbed))
        return { 
            'observation': obs.copy(),
            'image' : image.copy(),
            'ego': ego_image.copy()
        }
    
    def get_image(self, camera='camera', mode='human'):
        self.instance_channel.set_action(
            'GetRGB',
            id=self.object2id[camera],
            width=self.resolution,
            height=self.resolution,
            fov=54
        )

        self._step() #requires a step to generate images
        image_np = np.frombuffer(self.instance_channel.data[self.object2id[camera]]['rgb'], dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np
    
    def get_depth(self, camera='camera', mode='human'):
        self.instance_channel.set_action(
            'GetDepth',
            id=self.object2id[camera],
            zero_dis=0.3,#0.29*self.scale
            one_dis=10,#2*self.scale
            width=self.resolution,
            height=self.resolution,
            fov=54
        )

        self._step() #requires a step to generate images
        image_np = np.frombuffer(self.instance_channel.data[self.object2id[camera]]['depth'], dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_ANYDEPTH)
        return image_np
    
    def get_noised_image(self, camera):
        rgb = self.get_image(camera)
        depth = self.get_depth(camera)
        idx=(depth==0)
        chex = np.random.random(rgb.shape) * 255
        rgb[idx]=chex[idx]
        return rgb
    
    def _load_agent(self):               
        self.asset_channel.set_action(
            action='InstanceObject',
            name='my_franka_prefab' if self.gripper == "franka" else "robotiqfranka",
            id=self.object2id['franka']
        )

        if self.gripper == "robotiq":
            self.instance_channel.set_action(
                "SetIKTargetOffset",
                id=self.object2id['franka'],
                position=[0, 0.251, 0],#0.261 #0.212 franka
            )
        else:
            self.instance_channel.set_action(
                "SetIKTargetOffset",
                id=self.object2id['franka'],
                position=[0, 0.212, 0],#0.261 #0.212 franka
            )

        joint_velocities = [1 for i in range(8)]

        self.instance_channel.set_action(
            "SetJointVelocity",
            id=965874,
            joint_velocitys=joint_velocities
        )
             
        self.instance_channel.set_action(
            "IKTargetDoMove",
            id=965874,
            position=[self.init_pos[0], self.init_pos[1], self.init_pos[2]],
            duration=0,
            speed_based=False,
        )

        self.instance_channel.set_action(
            "IKTargetDoRotate",
            id=self.object2id['franka'],
            vector3=[-60, 0, 90],
            duration=0,
            speed_based=False,
        )

        self.controller.reset()
        self._step()
        self._set_gripper_width(0.04)

        for i in range(5):
            self._step()

    def _load_cloth(self):               
        self.asset_channel.set_action(
            action='InstanceObject',
            name='HangingClothSolver',
            id=self.object2id['cloth']
       )
        
        disp = (2 * np.random.rand(3) - 1) * 0.35
        rot_disp = (2 * np.random.rand(1) - 1) * 10

        self.instance_channel.set_action(
            'SetTransform',
            id=self.object2id['cloth'],
            position=[disp[0] -5, disp[1] +7.5, disp[2]+1],
            rotation=[0, 45+rot_disp, 0]
        )

        self.instance_channel.set_action(
            'SetSolverParameters',
            attr_name='hanging_cloth_attr',
            id=self.object2id['cloth'],
            gravity=[0, -9.8, 0],
        )
        if self.with_force_zone:
            self.instance_channel.set_action(
                'SetForceZoneParameters',
                attr_name='halling_cloth_attr',
                id=self.object2id['cloth'],
                orientation=self.np_random.uniform(-180, 180),
                intensity=self.force_zone_intensity,
                turbulence=self.force_zone_turbulence,
                turbulence_frequency=2,
            )
        for i in range(self.reset_step_count):
            self._step()

    def _destroy_cloth(self):
        self.instance_channel.set_action(
            action='Destroy',
            id=self.object2id['cloth']
        )
        self._step()

    def _destroy_agent(self):
        self.instance_channel.set_action(
            action='Destroy',
            id=self.object2id['franka']
        )
        self._step()   

    #get joint angles in rads
    def _get_joint_angles(self):
        return np.array(self.instance_channel.data[self.object2id['franka']]['joint_positions']) / (180 / 3.14159265)
    
    def _get_joint_velocities(self):
        return np.array(self.instance_channel.data[self.object2id['franka']]['joint_velocities'])# / (180 / 3.14159265)

    def _get_eef_position(self):
        if self.gripper == "robotiq":
            return np.array(self.instance_channel.data[self.object2id['gripper']]['positions'][7]) / self.scale #franka pos 3
        else:
            return np.array(self.instance_channel.data[self.object2id['gripper']]['positions'][3]) / self.scale #franka pos 3
    
    def _get_cloth_position(self):
        try:
            return np.array(self.instance_channel.data[self.object2id['cloth']]['avg_position']) / self.scale
        except:
            return np.zeros(3)

    def _get_cloth_velocity(self):
        return np.array(self.instance_channel.data[self.object2id['cloth']]['avg_velocity']) / self.scale
    
    def _get_cloth_corner_positions(self):
        return np.array(self.instance_channel.data[self.object2id['cloth']]['corner_positions']) / self.scale

    def _get_force_zone_parameters(self):
        return np.array([
            self.instance_channel.data[self.object2id['cloth']]['force_zone_orientation'] / 180 * math.pi,
            self.instance_channel.data[self.object2id['cloth']]['force_zone_intensity'],
            self.instance_channel.data[self.object2id['cloth']]['force_zone_turbulence'],
        ])


    def _check_success(self, obs):
        return (-self._compute_reward2() < 0.25)

    def _get_grabbed(self):
        try:
            return self.instance_channel.data[self.object2id['cloth']]['grabbed']
        except:
            return False

    def _compute_reward(self, obs=None):
        corner_positions = self._get_cloth_corner_positions()
        corner1 = corner_positions[:3]
        gripper_pos = self._get_eef_position()
        distance = self._distance(gripper_pos, corner1)
        return -distance
    
    def _compute_reward2(self, obs=None):
        corner_positions = self._get_cloth_corner_positions()
        corner1 = corner_positions[:3]
        cloth_pos = self._get_cloth_position()#obs[6:9]
        target_pos = cloth_pos
        target_pos[0] += 0.4
        distance = self._distance(corner1, target_pos)
        return -distance
    

    def _distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2, axis=-1)

def thread_function(name):
    env = FrankaClothHangEnv(
        executable_file="@editor",
        )
    
    

    step = 0
    total_steps = 0
    steps_to_collect = 100000/4
    while total_steps < steps_to_collect:
        observations = []
        images = []
        images256 = []
        depths = []
        actions = []
        rewards = []
        is_firsts = []
        is_lasts = []

        obs = env.reset()
        image = obs['image']
        image256 = env.rerender()
        depth = obs['depth']
        obs = obs['observation']
        action = np.zeros(env.action_space.shape)
        reward = 0
        is_first = True
        is_last = False
        while True:
            observations.append(obs)
            images.append(image)
            images256.append(image256)
            depths.append(depth)
            actions.append(action)
            rewards.append(reward)
            is_firsts.append(is_first)
            is_lasts.append(is_last)

            obs, depth, image, action, reward, done = env.heuristic()
            image256 = env.rerender()
            is_first = False
            is_last = done
            step += 1

            if done:
                observations.append(obs)
                images.append(image)
                images256.append(image256)
                depths.append(depth)
                actions.append(action)
                rewards.append(reward)
                is_firsts.append(is_first)
                is_lasts.append(is_last)
                break

        total_steps += step
        print(f'episode done, thread: {name} length: {step} sum of rewards: {np.sum(rewards)}, {total_steps}/{steps_to_collect} complete')
        
        #save file
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "episodes/normal_fixed_IK/faux_episode_editor_" + timestr + "-" + str(name) + "_" + str(step) + ".npz"
        step = 0
        np.savez_compressed(filename, observation=observations, image=images, largeimgs=images256, depth=depths, reward=rewards, is_first=is_firsts, is_last=is_lasts, is_terminal=is_lasts,action=actions, )

if __name__ == "__main__":
    import threading
    import logging
    threads = list()
    for index in range(4):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=thread_function, args=(index,))
        threads.append(x)
        time.sleep(0.5)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)
