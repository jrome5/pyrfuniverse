from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import pyrfuniverse.utils.rfuniverse_utility as utility
import os

env = RFUniverseBaseEnv()

ur5 = env.LoadURDF(path=os.path.abspath('../URDF/UR5/ur5_robot.urdf'), native_ik=True)
ur5.SetTransform(position=[1, 0, 0])
yumi = env.LoadURDF(path=os.path.abspath('../URDF/yumi_description/urdf/yumi.urdf'), native_ik=False)
yumi.SetTransform(position=[2, 0, 0])
kinova = env.LoadURDF(path=os.path.abspath('../URDF/kinova_gen3/GEN3_URDF_V12.urdf'), native_ik=False)
kinova.SetTransform(position=[3, 0, 0])
env.step()

ur5.IKTargetDoMove(position=[0, 0.5, 0], duration=0.1, relative=True)
env.step()
ur5.WaitDo()
ur5.IKTargetDoMove(position=[0, 0, -0.5], duration=0.1, relative=True)
env.step()
ur5.WaitDo()
ur5.IKTargetDoMove(position=[0, -0.2, 0.3], duration=0.1, relative=True)
ur5.IKTargetDoRotateQuaternion(quaternion=utility.UnityEularToQuaternion([0, 90, 0]), duration=30, relative=True)
env.step()
ur5.WaitDo()

env.Pend()
