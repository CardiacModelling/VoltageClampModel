import numpy as np
import heka_reader

def load(f, i, vccc=False):
    # f: file path
    # i: index list, [group_ind, series_ind, sweep_ind]
    # vccc: bool, if True, read 4 traces, else 2.
    #
    # return
    # ====
    # out: Recorded traces
    # times: Recorded traces' times
    b = heka_reader.Bundle(f)
    if vccc:
        nt = 4
    else:
        nt = 2
    out = []
    for t in range(nt):
        out.append(b.data[i + [t]])
    info = b.pul[i[0]][i[1]][i[2]][0]
    times = np.linspace(info.XStart,
            info.XStart + info.XInterval * (len(out[0])-1),
            len(out[0]))
    return out, times


mc4_experiment_alphas = { # exp#: (alpha_r, alpha_p)
    0: (0, 0),
    1: (0.2, 0.2),
    2: (0.4, 0.4),
    3: (0.6, 0.6),
    4: (0.8, 0.8),
    5: (0.95, 0.95),
    6: (0.2, 0),
    7: (0.4, 0),
    8: (0.6, 0),
    9: (0.8, 0),
    10: (0.95, 0),
    11: (0.95, 0.2),
    12: (0.95, 0.4),
    13: (0.95, 0.6),
    14: (0.95, 0.8),
    15: (0.95, 0.95),
}
