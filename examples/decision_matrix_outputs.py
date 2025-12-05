import numpy as np
import aspect


def pairwise_decision_results(M, labels, dict_comps):

    """
    Loop through all pairs of options and print the winner.

    Convention: M[i,j] > M[j,i] means i beats j.

    """

    n = len(labels)
    for idx_categ in labels:
        print(f'\n Component {dict_comps[idx_categ]}:')
        for idj_categ in labels:
            in_comb = (idx_categ, idj_categ)
            rev_comb = (idj_categ, idx_categ)
            out_idx = M[in_comb[0], in_comb[1]]
            if idx_categ == idj_categ:
                diag = out_idx == 2
                msg = '' if diag else '(BAD DIAG: Central diagonal not 2)'
                out_comp = dict_comps[idj_categ]
            else:
                out_comp = dict_comps[in_comb[out_idx]]
                diag = out_comp == dict_comps[rev_comb[M[rev_comb[0], rev_comb[1]]]]
                msg = '' if diag else '(Missmatch order result)'

            print(f'{dict_comps[idx_categ]} vs {dict_comps[idj_categ]} = {out_comp} {msg}')

        # a, b = labels[i], labels[j]
        # if M[i, j] > M[j, i]:
        #     winner = a
        # elif M[i, j] < M[j, i]:
        #     winner = b
        # else:
        #     winner = "Tie"
        # print(f"Input: {a} vs {b} -> Result: {winner}")

    return


comps = list(aspect.cfg['shape_number'].values())
dm_choice = np.array(aspect.cfg['decision_matrices']['time'])
pairwise_decision_results(dm_choice, comps, aspect.cfg['number_shape'])
