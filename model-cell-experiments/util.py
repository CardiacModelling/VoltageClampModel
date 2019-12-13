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
