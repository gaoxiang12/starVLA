"""Scope the expert recorder hook so it cannot retain a previous scene."""
from contextlib import contextmanager


@contextmanager
def record_frame_phases(env, frame_phases, get_phase, on_capture):
    original = env._take_picture
    sentinel = object()
    previous_attribute = env.__dict__.get('_take_picture', sentinel)

    def capture():
        previous_frame = env.FRAME_IDX
        original()
        assert env.FRAME_IDX == previous_frame + 1
        frame_phases.append(get_phase())
        on_capture(len(frame_phases))

    env._take_picture = capture
    try:
        yield
    finally:
        if previous_attribute is sentinel:
            del env.__dict__['_take_picture']
        else:
            env._take_picture = previous_attribute
