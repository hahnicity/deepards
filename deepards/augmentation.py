import math

import numpy as np
from scipy.signal import resample
from ventmap.SAM import find_x0s_multi_algorithms, x0_heuristic


class NaiveWindowWarping(object):
    def __init__(self, rate_lower_bound, rate_upper_bound, probability):
        """
        Window Warmping as covered by Le Guennec 2016. No special actions done otherwise.

        :param rate_lower_bound: Minimum rate to warp either insp or expiratory time
        :param rate_upper_bound: Maximum rate to warp either insp or expiratory time
        :param probability: probability of this transform being applied to a batch
        """
        self.rate_lower_bound = rate_lower_bound
        self.rate_upper_bound = rate_upper_bound
        self.probability = probability
        if self.probability < 0 or self.probability > 1:
            raise Exception('Probability bounding needs to be between 0 and 1.')
        self.min_size = 10
        # warped sequence size shouldn't take up more than half the total waveform
        self.max_size = 224 / 2 / self.rate_upper_bound

    def __call__(self, sub_batch):
        if np.random.rand() > self.probability:
            return sub_batch

        breaths_insts, chans, seq_len = sub_batch.shape
        dt = 0.02
        # modify each breath instance in the sub batch
        for b_idx, inst in enumerate(sub_batch):

            warping_ratio = np.random.uniform(self.rate_lower_bound, self.rate_upper_bound)
            # +1 because randint is [min_size, max_size)
            slice_len = np.random.randint(self.min_size, self.max_size+1)
            start = np.random.randint(0, seq_len-slice_len)
            end = start + slice_len
            chunk = inst[0][start:end]
            new_size = int(math.floor(slice_len*warping_ratio))
            new_chunk = resample(chunk, new_size)
            new_inst = np.concatenate((inst[0][:start], new_chunk, inst[0][end:]))
            if len(new_inst) >= seq_len:
                sub_batch[b_idx] = new_inst[:seq_len].reshape((1, seq_len))
            elif len(new_inst) < seq_len:
                sub_batch[b_idx] = resample(new_inst, seq_len).reshape((1, seq_len))

        return sub_batch


class IEWindowWarpingBase(object):
    def __init__(self, rate_lower_bound, rate_upper_bound, probability):
        """
        Window Warmping as covered by Le Guennec 2016 and applied to either inspiratory or
        expiratory lims.

        :param rate_lower_bound: Minimum rate to warp either insp or expiratory time
        :param rate_upper_bound: Maximum rate to warp either insp or expiratory time
        :param probability: probability of this transform being applied to a batch
        """
        self.rate_lower_bound = rate_lower_bound
        self.rate_upper_bound = rate_upper_bound
        self.probability = probability
        if self.probability < 0 or self.probability > 1:
            raise Exception('Probability bounding needs to be between 0 and 1.')

    def warp(self, sub_batch, i_or_e_choices):
        """
        :param i_or_e_choices: array of 1/0s or True/False vals on whether to use insp or exp lim.
                               True (1) for insp, False (0) for exp.
        """
        if np.random.rand() > self.probability:
            return sub_batch

        breaths_insts, chans, seq_len = sub_batch.shape
        dt = 0.02
        # modify each breath instance in the sub batch
        for b_idx, inst in enumerate(sub_batch):
            rel_time_array = list(np.arange(dt, dt+dt*seq_len, dt))
            x0s = find_x0s_multi_algorithms(list(inst[0]), rel_time_array, rel_time_array[-1], dt=dt)
            i_time, x0_idx = x0_heuristic(x0s, None, rel_time_array)

            # warp the i or e time.
            warping_ratio = np.random.uniform(self.rate_lower_bound, self.rate_upper_bound)
            i_or_e = i_or_e_choices[b_idx]

            # Always stretch sequences without an x0 because otherwise we wont have a complementary
            # part if we perform shrinking.
            if x0_idx >= seq_len - 1:
                start = 0
                end = seq_len
                warping_ratio = np.random.uniform(1.0, self.rate_upper_bound)
                n_new_pts = int(math.floor(end*warping_ratio))
                new_inst = resample(inst[0], n_new_pts)[:seq_len].reshape((1, seq_len))
            elif i_or_e:  # use inspiratory lim
                end = x0_idx
                n_new_pts = int(math.floor(end*warping_ratio))
                if n_new_pts <= 1:
                    n_new_pts = end
                new_chunk = resample(inst[0][:end], n_new_pts)
                n_remaining_pts = seq_len - n_new_pts
                if n_remaining_pts <= 0:
                    new_inst = new_chunk[:seq_len].reshape((1, seq_len))
                elif n_remaining_pts == 1:
                    new_inst = np.append(new_chunk, inst[0][end:])[:seq_len].reshape((1, seq_len))
                else:
                    new_remainder = resample(inst[0][end:], n_remaining_pts)
                    new_inst = np.append(new_chunk, new_remainder).reshape((1, seq_len))
            else:  # use expiratory lim
                start = x0_idx
                n_new_pts = int(math.floor((seq_len-start)*warping_ratio))
                if n_new_pts <= 1:
                    n_new_pts = seq_len-x0_idx
                new_chunk = resample(inst[0][start:], n_new_pts)
                n_remaining_pts = seq_len - n_new_pts
                if n_remaining_pts <= 0:  # breaths always should start at inspiration
                    new_inst = np.append(inst[0][0:start], new_chunk)[:seq_len].reshape((1, seq_len))
                elif n_remaining_pts == 1:
                    new_inst = np.append(inst[0][:start], new_chunk)[:seq_len].reshape((1, seq_len))
                else:
                    new_remainder = resample(inst[0][:start], n_remaining_pts)
                    new_inst = np.append(new_remainder, new_chunk).reshape((1, seq_len))

            sub_batch[b_idx] = new_inst
        return sub_batch


class IEWindowWarpingIEProgrammable(IEWindowWarpingBase):
    def __init__(self, rate_lower_bound, rate_upper_bound, probability, use_i):
        """
        Window Warmping as covered by Le Guennec 2016 and applied to either inspiratory or
        expiratory lims. Use of insp or exp. lim is programmable here tho.

        :param rate_lower_bound: Minimum rate to warp either insp or expiratory time
        :param rate_upper_bound: Maximum rate to warp either insp or expiratory time
        :param probability: probability of this transform being applied to a batch
        :param use_i: True if we want to use only insp lim. False if we want exp. lim
        """
        super(IEWindowWarpingIEProgrammable, self).__init__(rate_lower_bound, rate_upper_bound, probability)
        self.use_i = use_i

    def __call__(self, sub_batch):
        b_insts, _, __ = sub_batch.shape
        i_or_e_choices = [self.use_i for _ in range(b_insts)]
        return self.warp(sub_batch, i_or_e_choices)


class IEWindowWarping(IEWindowWarpingBase):
    def __init__(self, rate_lower_bound, rate_upper_bound, probability):
        """
        Window Warmping as covered by Le Guennec 2016 and applied to either inspiratory or
        expiratory lims.

        :param rate_lower_bound: Minimum rate to warp either insp or expiratory time
        :param rate_upper_bound: Maximum rate to warp either insp or expiratory time
        :param probability: probability of this transform being applied to a batch
        """
        super(IEWindowWarping, self).__init__(rate_lower_bound, rate_upper_bound, probability)

    def __call__(self, sub_batch):
        b_insts, _, __ = sub_batch.shape
        i_or_e_choices = np.random.choice([True, False], size=b_insts)
        return self.warp(sub_batch, i_or_e_choices)
