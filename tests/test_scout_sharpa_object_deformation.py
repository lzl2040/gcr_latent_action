import pytest

from scripts.scout_sharpa_object_deformation import evenly_spaced_episodes, sample_frames


def test_stratified_episode_selection_covers_early_and_late_recordings_without_duplicates():
    episodes = [{"episode_index": index} for index in range(31)]
    selected = evenly_spaced_episodes(episodes, 7)
    assert selected[0] == episodes[0]
    assert selected[-1] == episodes[-1]
    assert len({episode["episode_index"] for episode in selected}) == 7
    assert evenly_spaced_episodes(episodes, 40) == episodes
    with pytest.raises(ValueError, match="positive"):
        evenly_spaced_episodes(episodes, 0)


def test_sampling_removes_rounded_duplicates_and_never_crosses_episode_boundaries():
    assert sample_frames(10, [0.9, 0.2, 0, 0.21, 0.99]) == [0, 2, 8, 9]
    assert sample_frames(1, [0, 0.1, 0.9]) == [0]
    with pytest.raises(ValueError, match="fractions"):
        sample_frames(10, [-0.1])
    with pytest.raises(ValueError, match="fractions"):
        sample_frames(10, [1])
