import torch

from mrrate_report_training.train import exact_rank_indices


def test_distributed_epoch_has_full_coverage_without_replacement():
    shards = [
        exact_rank_indices(11, epoch=2, seed=4, world=4, rank=rank, shuffle=True)
        for rank in range(4)
    ]
    real = [value for shard in shards for value in shard if value >= 0]
    assert sorted(real) == list(range(11))
    assert len(real) == len(set(real))
    assert {len(shard) for shard in shards} == {3}
    assert sum(value == -1 for shard in shards for value in shard) == 1

