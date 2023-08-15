import math
import numpy as np
from gym import spaces
from gym.utils import seeding
import copy

import cv2 #for camera stuff

import sys
sys.path.append("/home/jack/Documents/pyrfuniverse/")
import pyrfuniverse.attributes as attr
from pyrfuniverse.envs import RFUniverseGymWrapper

class FrankaClothHangEnv(RFUniverseGymWrapper):
    metadata = {'render.modes': ['human', 'rgb_array']}
    scale = 8
    y_bound = 3.8
    franka_init_y = 4.8
    object2id = {
        'franka': 1001,
        'target': 1000,
        'cloth': 89212,
        'gripper': 10010,
        'camera': 123456,
        'agent': 1234
    }

    def __init__(
            self,
            with_force_zone=False,
            force_zone_intensity=1,
            force_zone_turbulence=10,
            max_steps=500,

            cloth_init_pos_min=(-4, 4, -3),
            cloth_init_pos_max=(0, 6, 0),
            executable_file=None,
            assets=['HangingClothSolver', 'Camera', "my_franka_panda", "myfrankahand"],
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

        self.cloth_init_pos_min = np.array(cloth_init_pos_min)
        self.cloth_init_pos_max = np.array(cloth_init_pos_max)
        self.cloth_init_rot_min = np.array([0, -90, 0])
        self.cloth_init_rot_max = np.array([0, -70, 0])

        self.cloth_init_pos = (-5, 6, 0)
        self.reset_step_count = 500
        self.tolerance = 0.2
        self.resolution = 64

        self.seed()
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
            'depth' : spaces.Box(0.0, 1.0, shape=(self.resolution, self.resolution), dtype=np.float32),
        })

        self.target_grabbed = False

    def step(self, action: np.ndarray):
        """
        Params:
            action: 4-d numpy array. -> move 3D and grab y/n
        """

        pos_ctrl = np.array(action[:3]) * 0.005 * self.scale
        curr_pos = self._get_eef_position() * self.scale
        pos_ctrl = curr_pos + pos_ctrl

        self.instance_channel.set_action(
            'SetTransform',
            id=self.object2id['agent'],
            position=list(pos_ctrl),
        )

        #if really confident, grab/release. else do nothing
        grab_conf = action[3]
        if(grab_conf < -0.5):
            self._set_gripper_width(0.4)
        elif(grab_conf > 0.5):
            self._set_gripper_width(0.01)

        self._step()

        self.t += 1
        self.total_t += 1

        done = False
        obs = self._get_obs()
        reward = self._compute_reward(obs['observation']) + self._compute_reward2() + float(self._get_grabbed())
        success = self._check_success(obs['observation'])
        if success or self.t == self.max_steps:
            done = True
        info = {
            'is_success': success
        }

        return obs, reward, done, info


    def reset(self):
        self._destroy_cloth()
        self._destroy_agent()
     
        self.t = 0
        self.target_grabbed = False

        self._load_cloth()
        self._load_agent()
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        self._step()

    def heuristic(self):
        reward = -self.t
        action = []
        delta = 0.005 * self.scale
        corner_target = self._get_cloth_corner_positions()[:3]
        target_position = []
        if(self.target_grabbed):
            target_position = self._get_cloth_position() * self.scale
            target_position[0] -= (0.4 * self.scale)
        else:
            target_position = (corner_target+np.array([-0.1,0,0])) * self.scale
        curr_pos = self._get_eef_position() * self.scale
        pos_diff = (target_position - curr_pos)
        np.clip(pos_diff, -1, 1)
        pos_ctrl = curr_pos + (pos_diff * delta) #keep same scale as agent
        self.instance_channel.set_action(
            'SetTransform',
            id=self.object2id['agent'],
            position=list(pos_ctrl),
        )

        action.append(pos_diff)
        grab_conf = -1
        distance = self._distance(target_position, pos_ctrl)
        if(distance < self.tolerance):
            grab_conf = 1 # self.target_grabbed #do not regrab
        grab_conf += self.target_grabbed
        #if really confident, grab/release. else do nothing
        if(grab_conf < -0.5):
            self._set_gripper_width(0.4)
        elif(grab_conf > 0.5):
            self._set_gripper_width(0.01)
        action.append(grab_conf)
        self._step()
        self.target_grabbed = self._get_grabbed()

        achieved_goal = pos_ctrl / self.scale

        reward = self._compute_reward2() + (0.1*self.target_grabbed) + -self._distance(corner_target, achieved_goal)

        self.t += 1
        success = (distance < self.tolerance*self.scale)

        end_target = self._get_cloth_position()
        end_target[0] -= 0.4
        achieved_goal = pos_ctrl / self.scale
        ep_success = self._distance(achieved_goal, end_target) < (0.05)
        done = self.t >= self.max_steps or ep_success

        obs = self._get_obs()
        image = obs['image']
        depth = obs['depth']
        return obs['observation'], image, depth, action, reward, done

    def _get_gripper_width(self) -> float:
        gripper_joint_positions = copy.deepcopy(self.instance_channel.data[self.object2id['agent']]['joint_positions'])
        return [(gripper_joint_positions[0] + gripper_joint_positions[1]) / self.scale]

    def _set_gripper_width(self, w: float):
        w = (w / 2) * self.scale
        self.instance_channel.set_action(
            'SetJointPosition',
            id=self.object2id['agent'],
            joint_positions=[w, w],
        )

    def _get_obs(self):
        catcher_position = self._get_eef_position()
        cloth_position = self._get_cloth_position()

        gripper_width = self._get_gripper_width()
        grabbed = np.asarray([self._get_grabbed()], dtype=np.float32)
        image = self.get_image()
        depth = self.get_depth()
        obs = np.concatenate((catcher_position, gripper_width, cloth_position, grabbed))
        return { 
            'observation': obs.copy(),
            'image' : image.copy(),
            'depth' : depth.copy()
        }
    
    def get_image(self, mode='human'):
        self.instance_channel.set_action(
            'GetRGB',
            id=self.object2id['camera'],
            width=self.resolution,
            height=self.resolution,
            fov=42
        )

        self._step() #requires a step to generate images
        image_np = np.frombuffer(self.instance_channel.data[self.object2id['camera']]['rgb'], dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np
    
    def get_depth(self, mode='human'):
        self.instance_channel.set_action(
            'GetDepth',
            id=self.object2id['camera'],
            zero_dis=0.29*self.scale,
            one_dis=2*self.scale,
            width=self.resolution,
            height=self.resolution,
            fov=42
        )

        self._step() #requires a step to generate images
        image_np = np.frombuffer(self.instance_channel.data[self.object2id['camera']]['depth'], dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_ANYDEPTH)
        return image_np
    
    def _load_agent(self):               
        self.asset_channel.set_action(
            action='InstanceObject',
            name='myfrankahand',
            id=self.object2id['agent']
        )

        init_pos =self.np_random.uniform(low=self.init_agent_pos_min, high=self.init_agent_pos_max)
        self.instance_channel.set_action(
            'SetTransform',
            id=self.object2id['agent'],
            position=list(init_pos),
        )

        self._set_gripper_width(0.04)


        for i in range(5):
            self._step()

    def _load_cloth(self):               
        self.asset_channel.set_action(
            action='InstanceObject',
            name='HangingClothSolver',
            id=self.object2id['cloth']
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
            id=self.object2id['agent']
        )
        self._step()   

    def _get_eef_position(self):
        return np.array(self.instance_channel.data[self.object2id['agent']]['position']) / self.scale
    
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
        corner1 = corner_positions[3:]
        gripper_pos = self._get_eef_position()
        distance = self._distance(gripper_pos, corner1)
        return -distance
    
    def _compute_reward2(self, obs=None):
        corner_positions = self._get_cloth_corner_positions()
        corner1 = corner_positions[3:]
        cloth_pos = self._get_cloth_position()#obs[6:9]
        target_pos = cloth_pos
        target_pos[0] -= 0.4 
        distance = self._distance(corner1, target_pos)
        return -distance
    

    def _distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2, axis=-1)

if __name__ == "__main__":
    env = FrankaClothHangEnv(
        executable_file="@editor",
        )
        
    import time
    steps_to_collect = 50000
    step = 0
    total_steps = 0
    while total_steps < steps_to_collect:
        observations = []
        images = []
        depths = []
        actions = []
        rewards = []
        is_firsts = []
        is_lasts = []

        obs = env.reset()
        image = obs['image']
        depth = obs['depth']
        obs = obs['observation']
        action = np.zeros(env.action_space.shape)
        reward = 0
        is_first = True
        is_last = False
        while True:
            observations.append(obs)
            images.append(image)
            depths.append(depth)
            actions.append(action)
            rewards.append(reward)
            is_firsts.append(is_first)
            is_lasts.append(is_last)

            obs, image, depth, acts, reward, done = env.heuristic()
            is_first = False
            is_last = done
            step += 1

            if done:
                observations.append(obs)
                images.append(image)
                depths.append(depth)
                actions.append(action)
                rewards.append(reward)
                is_firsts.append(is_first)
                is_lasts.append(is_last)
                break
            
        total_steps += step
        print(f'episode done, length: {step} sum of rewards: {np.sum(rewards)}, {total_steps}/{steps_to_collect} complete')
        
        #save file
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "episodes/gripper_rgbd/faux_episode_" + timestr + "-" + str(step) + ".npz"
        step = 0
        np.savez_compressed(filename, observation=observations, image=images, depth=depths, reward=rewards, is_first=is_firsts, is_last=is_lasts, is_terminal=is_lasts,action=actions, )