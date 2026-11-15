"""Integration of saved stages, including target-coupling gating and reuse."""
import json
import numpy as np
import pytest

from parquet_superconductivity.config import default_config
from parquet_superconductivity.workflow import initialize, stage1_screening, stage2_vertex, stage3_pairing


def test_three_stages_and_target_resume(tmp_path):
    config = default_config()
    config['model'].update(radius=2, spacing=.06, temperature_eV=.005)
    config['solver'].update(maximum_memory_mb=512, maximum_iterations=30)
    config['continuation']['stages'] = [.05, .1, .2, .4, .6, .8, 1.]
    system, crpa, output = stage1_screening(config, tmp_path, progress=False)
    assert crpa.metadata['hf_used'] is False
    system, problem, result, gamma, output = stage2_vertex(config, output, progress=False)
    assert result.converged and gamma.metadata['target_reached']
    pair, superconductivity, output = stage3_pairing(config, output, progress=False)
    assert superconductivity.linearized.converged
    assert superconductivity.nonlinear.converged
    assert np.allclose(pair.interaction, .5 * pair.raw_gamma)
    summary = json.loads((output / '03_superconductivity' / 'summary.json').read_text())
    assert summary['target_scale'] == 1.
    assert summary['hf_used'] is False
    assert gamma.metadata['schema'] != 'abca_crpa_full_gamma_v1'
    gamma.flat._mmap.close()
    # Release the old mapping before stage 2 atomically replaces its archive
    # on Windows, then explicitly close the replacement as well.
    _, _, _, resumed_gamma, _ = stage2_vertex(config, output, progress=False)
    resumed_gamma.flat._mmap.close()
    continuation = json.loads((output / '02_vertex' / 'continuation.json').read_text())
    assert continuation['resume_from_scale'] == 1.
    assert continuation['attempts'] == []


def test_changed_config_cannot_overwrite_run(tmp_path):
    config = default_config()
    initialize(config, tmp_path)
    config['model']['temperature_eV'] *= 2
    with pytest.raises(ValueError, match='another configuration'):
        initialize(config, tmp_path)
