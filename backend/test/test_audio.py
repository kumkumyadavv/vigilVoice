import numpy as np

from audio.fragmenter import split_into_chunks


def test_three_second_chunking():

    sample_rate = 16_000

    # Simulate 18 seconds of audio
    waveform = np.zeros(sample_rate * 18)

    chunks = split_into_chunks(
        waveform,
        sample_rate,
    )

    assert len(chunks) == 6

    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 3

    assert chunks[1]["start"] == 3
    assert chunks[1]["end"] == 6

    assert chunks[2]["start"] == 6
    assert chunks[2]["end"] == 9

    assert chunks[3]["start"] == 9
    assert chunks[3]["end"] == 12

    assert chunks[4]["start"] == 12
    assert chunks[4]["end"] == 15

    assert chunks[5]["start"] == 15
    assert chunks[5]["end"] == 18