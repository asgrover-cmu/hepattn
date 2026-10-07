"""Node ordering for the CLIC dataset (``sort_nodes_by``).

The Linformer projection is indexed by sequence position, so the order of the node
sequence matters for it. These tests pin down (i) the permutation itself and (ii) that
``load_event`` permutes features and incidence columns with the *same* permutation.
"""

import numpy as np
import pytest
import torch

from hepattn.experiments.clic.pflow_data import NODE_SORT_MODES, CLICDataset, node_sort_order
from hepattn.utils.scaling import FeatureScaler

SCALE_DICT = "src/hepattn/experiments/clic/configs/clic_var_transform.yaml"

# One hand-built event: 2 tracks then 3 topoclusters, phi chosen so that no ordering
# coincides with file order and the phi-sorted and eta-sorted orders differ.
TRACK_PHI = [0.5, -2.0]
TOPO_PHI = [1.0, -1.0, 2.5]
TRACK_ETA = [0.2, 0.9]
TOPO_ETA = [-0.5, 1.5, 0.1]
N_TRACKS, N_TOPOS, N_PARTICLES = 2, 3, 3


def _is_track():
    return torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])


def _phi():
    return torch.tensor(TRACK_PHI + TOPO_PHI)


def _eta():
    return torch.tensor(TRACK_ETA + TOPO_ETA)


class TestNodeSortOrder:
    @pytest.mark.parametrize("mode", sorted(NODE_SORT_MODES))
    def test_is_a_permutation_and_not_identity(self, mode):
        order = node_sort_order(mode, _is_track(), _phi(), _eta())
        assert sorted(order.tolist()) == list(range(5))
        assert order.tolist() != list(range(5)), "event was built so that every mode reorders it"

    def test_phi_sorts_all_nodes_together(self):
        order = node_sort_order("phi", _is_track(), _phi(), _eta())
        assert order.tolist() == [1, 3, 0, 2, 4]
        assert torch.all(torch.diff(_phi()[order]) > 0)

    def test_eta_sorts_all_nodes_together(self):
        order = node_sort_order("eta", _is_track(), _phi(), _eta())
        assert order.tolist() == [2, 4, 0, 1, 3]

    def test_type_phi_keeps_tracks_first(self):
        order = node_sort_order("type_phi", _is_track(), _phi(), _eta())
        assert order.tolist() == [1, 0, 3, 2, 4]
        is_track = _is_track()[order]
        assert is_track.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
        assert torch.all(torch.diff(_phi()[order][:2]) > 0)
        assert torch.all(torch.diff(_phi()[order][2:]) > 0)

    def test_stable_on_ties(self):
        phi = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3])
        order = node_sort_order("phi", _is_track(), phi, _eta())
        assert order.tolist() == [0, 1, 2, 3, 4]

    def test_rejects_unknown_mode_at_construction(self):
        with pytest.raises(ValueError, match="sort_nodes_by"):
            CLICDataset(filepath="", inputs={}, targets={"particle": []}, scale_dict_path=SCALE_DICT, dummy_data=True, sort_nodes_by="pt")


def _per_event(arr):
    out = np.empty(1, dtype=object)
    out[0] = arr
    return out


def _synthetic_dataset(sort_nodes_by):
    """A CLICDataset with one hand-built event, bypassing the ROOT loader."""
    ds = CLICDataset.__new__(CLICDataset)
    ds.init_label_dicts()
    ds.init_variables_list()
    ds.scaler = FeatureScaler(SCALE_DICT)
    ds.max_nodes = 8
    ds.num_objects = 6
    ds.is_inference = False
    ds.sort_nodes_by = sort_nodes_by
    ds.n_tracks = np.array([N_TRACKS])
    ds.n_topos = np.array([N_TOPOS])
    ds.n_particles = np.array([N_PARTICLES])
    ds.track_cumsum = np.array([0, N_TRACKS])
    ds.topo_cumsum = np.array([0, N_TOPOS])
    ds.particle_cumsum = np.array([0, N_PARTICLES])

    g = torch.Generator().manual_seed(0)
    fda = {}
    for var in ds.track_variables:
        fda[var] = torch.rand(N_TRACKS, generator=g) + 0.5
    for var in ds.topo_variables:
        fda[var] = torch.rand(N_TOPOS, generator=g) + 0.5
    for var in ds.particle_variables:
        fda[var] = torch.rand(N_PARTICLES, generator=g) + 0.5
    fda["track_phi"] = torch.tensor(TRACK_PHI)
    fda["topo_phi"] = torch.tensor(TOPO_PHI)
    fda["track_eta"] = torch.tensor(TRACK_ETA)
    fda["topo_eta"] = torch.tensor(TOPO_ETA)
    for var in ("track_phi", "track_phi_int", "topo_phi"):
        fda[var.replace("phi", "sinphi")] = torch.sin(fda[var])
        fda[var.replace("phi", "cosphi")] = torch.cos(fda[var])
    ds.full_data_array = fda
    ds.particle_class = torch.tensor([0, 0, 4])
    # uproot(library="np") yields a 1-D object array holding one ndarray per event.
    ds.aux_data_array = {
        "track_particle_idx": _per_event(np.array([0, 1])),
        "topo2particle_topo_idx": _per_event(np.array([0, 1, 2])),
        "topo2particle_particle_idx": _per_event(np.array([0, 1, 2])),
        "topo2particle_energy": _per_event(np.array([1.0, 1.0, 1.0])),
    }
    return ds


class TestLoadEventSorting:
    def test_unsorted_is_file_order(self):
        ev = _synthetic_dataset(None).load_event(0)
        assert ev["node_raw_features"]["raw_phi"][:5].tolist() == pytest.approx(TRACK_PHI + TOPO_PHI)
        assert ev["node_raw_features"]["is_track"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]

    @pytest.mark.parametrize("mode", sorted(NODE_SORT_MODES))
    def test_features_and_incidence_share_one_permutation(self, mode):
        ref = _synthetic_dataset(None).load_event(0)
        ev = _synthetic_dataset(mode).load_event(0)
        n = N_TRACKS + N_TOPOS

        # Recover the permutation from raw phi (unique in this event) and check it is the
        # one node_sort_order defines.
        expected = node_sort_order(mode, _is_track(), _phi(), _eta())
        got = torch.tensor([(TRACK_PHI + TOPO_PHI).index(p) for p in ev["node_raw_features"]["raw_phi"][:n].tolist()])
        assert got.tolist() == expected.tolist()

        # Every per-node tensor moved with that permutation ...
        torch.testing.assert_close(ev["node_inp_features"][:n], ref["node_inp_features"][expected])
        for key in ref["node_raw_features"]:
            torch.testing.assert_close(ev["node_raw_features"][key][:n], ref["node_raw_features"][key][expected])
        # ... and so did the incidence columns, so each node keeps its own truth.
        torch.testing.assert_close(ev["incidence_truth"][:, :n], ref["incidence_truth"][:, expected])
        assert ev["incidence_truth"][:, :n].sum() > 0

        # Padding, particle side and mask are untouched.
        assert torch.all(ev["node_inp_features"][n:] == 0)
        assert torch.all(ev["incidence_truth"][:, n:] == 0)
        torch.testing.assert_close(ev["node_q_mask"], ref["node_q_mask"])
        for key in ref["particle_data"]:
            torch.testing.assert_close(ev["particle_data"][key], ref["particle_data"][key])
        torch.testing.assert_close(ev["indicator_truth"], ref["indicator_truth"])

    def test_phi_is_monotone_after_sort(self):
        ev = _synthetic_dataset("phi").load_event(0)
        phi = ev["node_raw_features"]["raw_phi"][: N_TRACKS + N_TOPOS]
        assert torch.all(torch.diff(phi) > 0)
