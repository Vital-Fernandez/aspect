import numpy as np


def comp_counter(arr_mask: np.ndarray) -> int:
    return np.sum(~arr_mask[:-1] & arr_mask[1:]) + arr_mask[0]


def test_comp_counter():

    arr_int = np.array([0, 1, 1, 0, 2, 2, 0])
    arr_bool = np.array([False, True, True, False, True, True, False])
    arr_float = np.array([0., 0., 1., 1., 0., 0., 2., 2., 0.])
    left_corner = np.array([True, False, False, True, True, False, False, True, True, False])
    right_corner = np.array([True, False, True, True, False, False, True, True, False, True, True])

    assert comp_counter(~(arr_int < 0.001)) == 2
    assert comp_counter(~(arr_bool < 0.001)) == 2
    assert comp_counter(~(arr_float < 0.001)) == 2
    assert comp_counter(~(left_corner < 0.001)) == 2
    assert comp_counter(~(right_corner < 0.001)) == 3

    arr1 = np.array([0., 1., 0., 2., 0.])        # 0 | 0 | 0   ? 3 zero regions
    arr2 = np.array([0., 0., 1., 2., 0., 3.])    # 00 | 0      ? 2 zero regions
    arr3 = np.array([1., 2., 3.])                # no zeros    ? 0 regions

    assert comp_counter(~(arr1 < 0.001)) == 2
    assert comp_counter(~(arr2 < 0.001)) == 2
    assert comp_counter(~(arr3 < 0.001)) == 3

    return