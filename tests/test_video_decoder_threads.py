from unittest.mock import patch

import av
import numpy as np
import pytest
import torchvision

from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


@pytest.mark.parametrize("codec", ["libx264", "libaom-av1"])
def test_pyav_threads_are_bounded_before_decode_and_frames_preserved(tmp_path, codec):
    path = tmp_path / "frames.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for index in range(12):
            rgb = np.full((32, 32, 3), index * 16, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        expected = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    real_reader = torchvision.io.VideoReader
    readers = []

    def reader_with_assertion(*args, **kwargs):
        reader = real_reader(*args, **kwargs)
        original_seek = reader.seek

        def checked_seek(*args, **kwargs):
            assert reader.container.streams.video[0].codec_context.thread_count == 1
            return original_seek(*args, **kwargs)

        reader.seek = checked_seek
        readers.append(reader)
        return reader

    with patch.object(torchvision.io, "VideoReader", side_effect=reader_with_assertion):
        for _ in range(8):
            frames = get_frames_by_timestamps(str(path), [0.1, 0.3, 0.3, 0.5], "torchvision_av")
            np.testing.assert_array_equal(frames, np.stack([expected[i] for i in [2, 6, 6, 10]]))
    assert all(reader.container is None and reader._c is None for reader in readers)
