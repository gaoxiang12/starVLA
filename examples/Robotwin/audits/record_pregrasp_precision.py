"""Add passive per-physics-step pose recording to the frozen grasp case runner."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'examples/Robotwin/eval_files')]
from examples.Robotwin.audits import run_grasp_lift_case as case

instances = []


class PrecisionScene(case.ObservedScene):
    def __init__(self, scene, callback):
        super().__init__(scene, callback)
        self.dt = float(scene.get_timestep())
        self.joints = {j.name:j for a in scene.get_all_articulations() for j in a.get_joints()}
        self.articulations = list(scene.get_all_articulations())
        self.blocks = sorted([e for e in scene.get_entities() if e.name == 'box'],
                             key=lambda e:int(e.per_scene_id))
        if len(self.blocks) != 3:
            raise RuntimeError('Expected exactly three box entities')
        self.block_ids = [int(e.per_scene_id) for e in self.blocks]
        self.ee = [self.joints[f'{prefix}_joint6'] for prefix in ('fl','fr')]
        self.finger_joints = [[self.joints[f'{prefix}_joint{i}'] for i in (7,8)] for prefix in ('fl','fr')]
        self.times, self.tcp, self.block_poses, self.finger_poses, self.drive_targets, self.qpos = [], [], [], [], [], []
        self.joint_names = [[j.name for j in a.get_active_joints()] for a in self.articulations]
        self.capture()
        instances.append(self)

    def capture(self):
        tcp=[]
        for joint in self.ee:
            pose=joint.global_pose
            matrix=pose.to_transformation_matrix()
            # Same calibrated Aloha TCP convention as robot._trans_endpose(True).
            matrix[:3,:3] = matrix[:3,:3] @ np.diag([1.,-1.,-1.])
            matrix[:3,3] += matrix[:3,:3] @ np.array([.12,0.,0.])
            tcp.append(matrix.copy())
        self.times.append(len(self.times)*self.dt)
        self.tcp.append(tcp)
        self.block_poses.append([e.pose.to_transformation_matrix() for e in self.blocks])
        self.finger_poses.append([[j.child_link.entity.pose.to_transformation_matrix() for j in group]
                                  for group in self.finger_joints])
        self.drive_targets.append([[float(j.get_drive_target()[0]) for j in group] for group in self.finger_joints])
        self.qpos.append([a.get_qpos().copy() for a in self.articulations])

    def step(self):
        try:
            return super().step()
        finally:
            self.capture()

    def save(self, output):
        np.savez_compressed(output, sim_s=np.asarray(self.times), tcp_world_matrix=np.asarray(self.tcp),
            block_world_matrix=np.asarray(self.block_poses), finger_world_matrix=np.asarray(self.finger_poses),
            finger_drive_targets=np.asarray(self.drive_targets), articulation_qpos=np.asarray(self.qpos),
            active_joint_names=np.asarray(self.joint_names),
            block_entity_ids=np.asarray(self.block_ids))


case.ObservedScene = PrecisionScene
if __name__ == '__main__':
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    try:
        case.main()
    finally:
        if len(instances) == 1:
            instances[0].save(output/'pregrasp_physics_poses.npz')
