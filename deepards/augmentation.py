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
            slice_len = np.random.randint(self.min_size, self.max_size)
            start = np.random.randint(0, seq_len-1-slice_len)
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


class IEWindowWarping(object):
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

    def __call__(self, sub_batch):
        if np.random.rand() > self.probability:
            return sub_batch

        # its not clear how I can handle case of down-warp, because this causes
        # issue that you don't have enuf samples in sequence. I might be able to extend
        # time of everything after that to compensate.
        #
        # Well, we can do
        # * fill of 0's
        # * fill with last known val.
        # * impute curve.
        # * extend first/last piece. depending on which one we originally modify
        # * redo everything to keep some data in place so that you can choose real data (not guaranteed to work all the time)
        #
        # + at least for now I think the extend complement is best other option
        #
        #
        # I also wonder if the resample function is the right one to use because it uses
        # a Fourier transform.
        breaths_insts, chans, seq_len = sub_batch.shape
        dt = 0.02
        # modify each breath instance in the sub batch
        for b_idx, inst in enumerate(sub_batch):
            rel_time_array = list(np.arange(dt, dt+dt*seq_len, dt))
            x0s = find_x0s_multi_algorithms(list(inst[0]), rel_time_array, rel_time_array[-1], dt=dt)
            i_time, x0_idx = x0_heuristic(x0s, None, rel_time_array)

            # warp the i or e time.
            i_or_e = True if np.random.rand() > .5 else False
            warping_ratio = np.random.uniform(self.rate_lower_bound, self.rate_upper_bound)

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
