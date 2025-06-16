import numpy as np
from scipy.io import loadmat
import scipy.sparse as sp


def read_sdpa_sparse_format(file_path):
    with open(file_path, 'r') as file:
        cnt = 0
        for line in file:
            if line.startswith(('"', '*')):
                continue
            cnt += 1
            if cnt == 1:
                mdim = int(line)
            elif cnt == 2:
                # block = int(line)  # unused
                pass
            elif cnt == 3:
                dim = list(map(int, line.split()))
            elif cnt == 4:
                b = np.array([list(map(float, line.split()))]).T
                break

        CA_data = [
            [{'i': [], 'j': [], 'data': []} for q in range(len(dim))] for k in range(mdim + 1)
        ]
        for line in file:
            if line.startswith(('"', '*')):
                continue
            (k, q, i, j, data) = tuple(line.split())
            kq_data = CA_data[int(k)][int(q) - 1]
            kq_data['i'].append(int(i) - 1)
            kq_data['j'].append(int(j) - 1)
            kq_data['data'].append(float(data))
            if i != j:
                kq_data['i'].append(int(j) - 1)
                kq_data['j'].append(int(i) - 1)
                kq_data['data'].append(float(data))

    CA = [
        sp.block_diag([
            sp.coo_array(
                (kq_data['data'], (kq_data['i'], kq_data['j'])),
                shape=(dim[q], dim[q])
            )
            for q, kq_data in enumerate(k_data)
        ]).tocsr()
        for k_data in CA_data
    ]
    C = [-CA[0]]
    A = [CA[1:]]

    blk = [('s', dim)]

    return blk, A, b, C


def read_sedumi_mat(file_path: str):
    data = loadmat(file_path)
    ldim = data['K']['l'].item().item()
    qdim = data['K']['q'].item().flatten().tolist()
    sdim = data['K']['s'].item().flatten().tolist()

    blk = [('l', ldim),
           ('q', qdim),
           ('s', sdim)]

    A = sp.csc_array(data['A'])
    m = A.shape[0]
    Al = A[:, :ldim]
    Aq = A[:, ldim:ldim + sum(qdim)]
    As = A[:, ldim + sum(qdim):]

    idx = (np.array([0] + sdim) ** 2).cumsum()
    AA = [
        [Al[[i], :].T for i in range(m)],
        [Aq[[i], :].T for i in range(m)],
        [sp.block_diag([As[[i], s:t].reshape(n, n) for s, t, n in zip(idx[:-1], idx[1:], sdim)], format='csr')
         for i in range(m)]
    ]

    b = data['b'].toarray()

    c = sp.csr_array(data['c'])
    cl = c[:ldim]
    cq = c[ldim:ldim + sum(qdim)]
    cs = c[ldim + sum(qdim):]
    css = sp.block_diag([cs[s:t].reshape(n, n) for s, t, n in zip(idx[:-1], idx[1:], sdim)], format='csr')
    C = [cl, cq, css]

    return blk, AA, b, C


if __name__ == '__main__':
    blk, A, b, C = read_sdpa_sparse_format('res/SDPLIB/data/hinf1.dat-s')
