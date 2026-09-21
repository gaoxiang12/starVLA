import unittest
import numpy as np
from starVLA.robotwin_feedback import measured_endpose, padded_command, feedback_keys
from examples.RobotwinEndPose.train_files.data_registry.data_config import ROBOT_TYPE_CONFIG_MAP


class FeedbackTest(unittest.TestCase):
    def observation(self):
        return dict(endpose=dict(left_endpose=[1,2,3,1,0,0,0], left_gripper=.25,
            right_endpose=[4,5,6,0,1,0,0], right_gripper=.75),
            joint_action={'vector': np.arange(14, dtype=np.float32)})

    def test_endpose_uses_current_native_pose_and_preserves_wxyz(self):
        obs = self.observation()
        expected = [1,2,3,1,0,0,0,.25,4,5,6,0,1,0,0,.75]
        np.testing.assert_array_equal(measured_endpose(obs), expected)
        obs['joint_action']['vector'][:] = np.nan
        np.testing.assert_array_equal(measured_endpose(obs), expected)

    def test_batched_training_and_single_observation_identical(self):
        obs = self.observation()
        batch = {'endpose': {k:np.stack([v,v]) for k,v in obs['endpose'].items()}}
        np.testing.assert_array_equal(measured_endpose(batch), np.stack([measured_endpose(obs)]*2))

    def test_invalid_pose_fails_instead_of_falling_back_to_commands(self):
        for invalid in ([1,2,3,0,0,0,0], [1,2,np.nan,1,0,0,0], [1,2,3]):
            obs = self.observation(); obs['endpose']['left_endpose'] = invalid
            with self.assertRaises(ValueError): measured_endpose(obs)
        with self.assertRaises(KeyError): measured_endpose({'joint_action': {'vector':np.zeros(14)}})

    def test_command_control_keeps_all_commands_and_ignores_endpose(self):
        obs = self.observation(); obs['endpose'] = None
        value = padded_command(obs)
        np.testing.assert_array_equal(value, [0,1,2,3,4,5,0,6,7,8,9,10,11,12,0,13])
        np.testing.assert_array_equal(value[[0,1,2,3,4,5,7,8,9,10,11,12,13,15]], np.arange(14))

    def test_action_contract_is_identical_for_both_state_inputs(self):
        command, pose = [ROBOT_TYPE_CONFIG_MAP[f'robotwin_feedback_{v}'] for v in ('command','endpose')]
        for attr in ('action_keys','action_key_dims','action_indices','gripper_indices','action_spec_id'):
            self.assertEqual(getattr(command,attr),getattr(pose,attr))
        for config, variant in ((command,'command'),(pose,'endpose')):
            self.assertEqual(config.state_keys,feedback_keys(variant))
            self.assertEqual(sum(config.state_key_dims.values()),16)
            self.assertEqual(config.action_indices,list(range(1,17)))
        self.assertNotEqual(command.state_spec_id,pose.state_spec_id)

    def test_eval_supplies_endpose_but_executes_joint_actions(self):
        from examples.RobotwinEndPose.interface import eval
        obs = self.observation()
        obs['observation'] = {name:{'rgb':np.zeros((2,2,3),np.uint8)} for name in
                             ('head_camera','left_camera','right_camera')}
        class Model:
            feedback_variant='endpose'
            task_language_mode='dataset_name'
            def step(self, example, step):
                self.example=example
                return np.arange(14,dtype=np.float32)
        class Env:
            task_name='blocks_ranking_rgb'
            take_action_cnt=0
            def get_instruction(self):return 'blocks ranking rgb'
            def take_action(self, action):self.action=action
        model,env=Model(),Env()
        eval(env,model,obs)
        np.testing.assert_array_equal(model.example['state'],measured_endpose(obs))
        np.testing.assert_array_equal(env.action,np.arange(14))


if __name__ == '__main__': unittest.main()
