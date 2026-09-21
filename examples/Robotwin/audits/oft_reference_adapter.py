"""Input contract for the released, state-free Qwen3-VL-OFT RoboTwin policy."""

UNNORM_KEY = 'new_embodiment'
INSTRUCTION = ('Start with the red block, followed by the green block and the blue block, '
               'placing them in order left to right.')
CONTRACT = dict(
    instruction=INSTRUCTION, state_input=False, image_order=['head', 'left', 'right'],
    image_size=[224, 224], image_color='RGB', unnorm_key=UNNORM_KEY,
    action_horizon=50, action_mode='abs', action_order='L6,R6,Lgrip,Rgrip',
    instruction_source='RoboTwin/description/task_instruction/blocks_ranking_rgb.json seen[3]; colors substituted',
    note='Red lift diagnostic with the original sorting instruction. Other-color holds are reported separately.')


def policy_observation(example):
    """Only images and the fixed task instruction; never simulator/state labels."""
    assert len(example['image']) == 3
    return dict(image=example['image'], lang=INSTRUCTION)


def adapt_client(model):
    assert model.unnorm_key == UNNORM_KEY and model.action_chunk_size == 50
    original_step = model.step

    def step(example, step=0):
        return original_step(policy_observation(example), step=step)

    model.step = step
    return model
