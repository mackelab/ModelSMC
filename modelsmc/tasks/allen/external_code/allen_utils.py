########################################################################################
#
# This file is copied from
#
# https://github.com/mackelab/IdentifyMechanisticModels_2020/blob/b93c90ec6156ae5f8afee6aaac7317373e9caf5e/6_allen/model/utils.py
#
# (the base simulator this file supports is from
# https://github.com/mackelab/IdentifyMechanisticModels_2020/blob/b93c90ec6156ae5f8afee6aaac7317373e9caf5e/6_allen/model/HodgkinHuxleyBioPhys.py)
#
# which was released under the MIT license:
#
# Copyright 2020 Pedro J. Goncalves, Jan-Matthis Lueckmann, Michael Deistler,
# Marcel Nonnenmacher, Kaan Ocal, Giacomo Bassetto, Chaitanya Chintaluri,
# William F. Podlaski, Sara A. Haddad, Tim P. Vogels, David S. Greenberg,
# Jakob H. Macke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The electrophysiology recordings loaded by `allen_obs_data` (the
# `ephys_cell_*.pkl` files, copied from that repo's `6_allen/support_files/`)
# originate from the Allen Cell Types Database (Allen Institute for Brain
# Science, 2015), available from celltypes.brain-map.org.
#
########################################################################################

import os
import pickle

import numpy as np
import torch
from sbi.utils import BoxUniform


def get_allen_task_parameters(task_idx):
    assert task_idx in range(1, 11), "task idx must be between 1 and 10"

    list_cells_AllenDB = [
        (518290966, 57, 0.0234 / 126),
        (509881736, 39, 0.0153 / 184),
        (566517779, 46, 0.0195 / 198),
        (567399060, 38, 0.0259 / 161),
        (569469018, 44, 0.033 / 403),
        (532571720, 42, 0.0139 / 127),
        (555060623, 34, 0.0294 / 320),
        (534524026, 29, 0.027 / 209),
        (532355382, 33, 0.0199 / 230),
        (526950199, 37, 0.0186 / 218),
    ]
    ephys_cell, sweep_number, A_soma = list_cells_AllenDB[task_idx - 1]
    return ephys_cell, sweep_number, A_soma


def allen_obs_data(ephys_cell, sweep_number, A_soma, dir_data="."):
    """Data for x_o. Cell from AllenDB
    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """
    # TODO only supports local loading of files via path
    real_data_path = os.path.join(
        dir_data,
        f"ephys_cell_{ephys_cell}_sweep_number_{sweep_number}.pkl",
    )

    # not sure whats up with this pickle encoding
    def pickle_load(file):
        """Loads data from file."""
        f = open(file, "rb")
        data = pickle.load(f, encoding="latin1")
        f.close()
        return data

    real_data_obs, I_real_data, dt, t_on, t_off = pickle_load(real_data_path)

    duration = 1450.0
    t = np.arange(0, duration, dt)

    # external current
    I = I_real_data / A_soma  # muA/cm2

    # return real_data_obs, I_obs
    return {
        "data": real_data_obs.reshape(-1),
        "time": t,
        "dt": dt,
        "I": I.reshape(-1),
        "t_on": t_on,
        "t_off": t_off,
    }

def synth_obs_data(idx: int, dir_cache="."):
    assert idx in range(1, 11), "task idx must be between 1 and 10"

    synth_data_path = os.path.join(
        dir_cache,
        f"allen_support_files/synthetic_obs_{idx}.pkl",
    )

    with open(synth_data_path, "rb") as f:
        synth_data = pickle.load(f)

    return synth_data

def prior_original(prior_log=False):
    """Prior"""
    range_lower = param_transform(
        prior_log,
        np.array([
            0.5,    # gbar_Na: minimum sodium channel conductance
            1e-4,   # gbar_K: minimum potassium channel conductance
            1e-4,   # g_leak: minimum leak channel conductance
            1e-4,   # gbar: minimum additional conductance
            50.0,   # tau_max: minimum max time constant
            40.0,   # Vt: minimum threshold voltage
            1e-4,   # nois_fact: minimum noise factor
            35.0,   # E_leak: minimum leak reversal potential
        ])
    )
    range_upper = param_transform(
        prior_log,
        np.array([
            80.0,   # gbar_Na: maximum sodium channel conductance
            15.0,   # gbar_K: maximum potassium channel conductance
            0.6,    # g_leak: maximum leak channel conductance
            0.6,    # gbar: maximum additional conductance
            3000.0, # tau_max: maximum max time constant
            90.0,   # Vt: maximum threshold voltage
            0.15,   # nois_fact: maximum noise factor
            100.0,  # E_leak: maximum leak reversal potential
        ])
    )

    prior_min = torch.tensor(range_lower.astype(np.float32))
    prior_max = torch.tensor(range_upper.astype(np.float32))
    return BoxUniform(prior_min, prior_max)


def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x


def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return x