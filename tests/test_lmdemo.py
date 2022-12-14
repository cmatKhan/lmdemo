import pytest
import logging
import cProfile
import pstats
import io
from lmdemo import linmodest
from .conftest import *

import numpy as np

def test_linmodest(lm_data,caplog):

    caplog.set_level(logging.DEBUG)
    
    # this tests that an error is actually raised
    with pytest.raises(ValueError):
        linmodest(lm_data['predictors'], lm_data['response'][:-2])

    assert ['shape of x: (144, 2); shape of y: (142,)'] == [rec.message for rec in caplog.records]

    # cite: https://stackoverflow.com/a/51541290/9708266
    pr = cProfile.Profile()
    pr.enable()

    actual = linmodest(lm_data['predictors'], lm_data['response']) #pylint: disable=E1102

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    with open('temp/linmodest_time.log', 'a') as f:
        f.write(s.getvalue())

    assert np.allclose(actual['coef'], lm_data['expected']['coef'])
    assert np.allclose(actual['vcov'], lm_data['expected']['vcov'])
    assert np.allclose(actual['sigma'], lm_data['expected']['sigma'])
    assert np.allclose(actual['df'], lm_data['expected']['df'])
