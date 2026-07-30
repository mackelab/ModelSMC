########################################################################################
#
# This file is copied from
#
# https://github.com/mackelab/IdentifyMechanisticModels_2020/blob/b93c90ec6156ae5f8afee6aaac7317373e9caf5e/6_allen/model/HodgkinHuxleyStatsMoments.py
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
########################################################################################

import numpy as np
from scipy import stats as spstats


# NOTE this function is somewhat weird, but works. Wouldn't touch it further for now.
class HodgkinHuxleyStatsMoments:
    """Moment based SummaryStats class for the Hodgkin-Huxley model
    Calculates summary statistics
    """

    def __init__(self, t_on, t_off, n_xcorr=5, n_mom=5, n_summary=13):
        self.t_on = t_on
        self.t_off = t_off
        self.n_xcorr = n_xcorr
        self.n_mom = n_mom
        self.n_summary = np.minimum(n_summary, n_xcorr + n_mom + 3)

    def calc(self, repetition_list):
        """Calculate summary statistics
        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance
        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        stats = []
        for r in range(len(repetition_list)):
            x = repetition_list[r]

            N = x["data"].shape[0]
            t = x["time"]
            dt = x["dt"]
            t_on = self.t_on
            t_off = self.t_off

            # initialise array of spike counts
            v = np.array(x["data"])

            # put everything to -10 that is below -10 or has negative slope
            ind = np.where(v < -10)
            v[ind] = -10
            ind = np.where(np.diff(v) < 0)
            v[ind] = -10

            # remaining negative slopes are at spike peaks
            ind = np.where(np.diff(v) < 0)
            spike_times = np.array(t)[ind]
            spike_times_stim = spike_times[(spike_times > t_on) & (spike_times < t_off)]

            # number of spikes
            if spike_times_stim.shape[0] > 0:
                spike_times_stim = spike_times_stim[
                    np.append(1, np.diff(spike_times_stim)) > 0.5
                ]

            # ISI
            # ISI = np.diff(spike_times_stim).astype(float)
            # ind = [0,1,-1]
            # ISI1 = np.array([1000.]*3)
            # ISI1[0:np.maximum(0,spike_times_stim.shape[0]-1)] = ISI[ind[0:np.maximum(0,spike_times_stim.shape[0]-1)]]
            # if spike_times_stim.shape[0] > 1:
            #    ISImom = np.array([np.mean(ISI),np.std(ISI)])
            # else:
            #    ISImom = np.array([t_off,0.])
            # ISI1 = np.array([t_off-t_on,0.])
            # ISI1[0:np.maximum(0,spike_times_stim.shape[0]-1)] = ISImom[0:np.maximum(0,spike_times_stim.shape[0]-1)]

            ## accommodation index
            # if spike_times_stim.shape[0] < 3:
            #    A_ind = 1000
            # else:
            #    ISI = np.diff(spike_times_stim)
            #    A_ind = np.mean( [ (ISI[i_min+1]-ISI[i_min])/(ISI[i_min+1]+ISI[i_min]) for i_min in range (0,ISI.shape[0]-1)] )

            # resting potential and std
            rest_pot = np.mean(x["data"][t < t_on])
            rest_pot_std = np.std(x["data"][int(0.9 * t_on / dt) : int(t_on / dt)])

            # auto-correlations
            x_on_off = x["data"][(t > t_on) & (t < t_off)] - np.mean(
                x["data"][(t > t_on) & (t < t_off)]
            )
            x_corr_val = np.dot(x_on_off, x_on_off)

            xcorr_steps = np.linspace(
                1.0 / dt, self.n_xcorr * 1.0 / dt, self.n_xcorr
            ).astype(int)
            x_corr_full = np.zeros(self.n_xcorr)
            for ii in range(self.n_xcorr):
                x_on_off_part = np.concatenate(
                    (x_on_off[xcorr_steps[ii] :], np.zeros(xcorr_steps[ii]))
                )
                x_corr_full[ii] = np.dot(x_on_off, x_on_off_part)

            x_corr1 = x_corr_full / x_corr_val

            std_pw = np.power(
                np.std(x["data"][(t > t_on) & (t < t_off)]),
                np.linspace(3, self.n_mom, self.n_mom - 2),
            )
            std_pw = np.concatenate((np.ones(1), std_pw))
            moments = (
                spstats.moment(
                    x["data"][(t > t_on) & (t < t_off)],
                    np.linspace(2, self.n_mom, self.n_mom - 1),
                )
                / std_pw
            )

            # concatenation of summary statistics
            try:
                sum_stats_vec = np.concatenate(
                    (
                        np.array([spike_times_stim.shape[0]]),
                        x_corr1,
                        np.array(
                            [
                                rest_pot,
                                rest_pot_std,
                                np.mean(x["data"][(t > t_on) & (t < t_off)]),
                            ]
                        ),
                        moments,
                    )
                )
                sum_stats_vec = sum_stats_vec[0 : self.n_summary]
            except:
                return None

            stats.append(sum_stats_vec)

        return np.asarray(stats)