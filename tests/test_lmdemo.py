import pytest
import logging


from .conftest import *
from lmdemo import linmodest

import numpy as np

def test_lmdemo(lm_data,caplog):

    caplog.set_level(logging.DEBUG)

    with pytest.raises(ValueError):
        linmodest(lm_data['predictors'], lm_data['response'][:-2])

    assert ['shape of A: (144, 2); shape of y: (142,)'] == \
        [rec.message for rec in caplog.records]

    
    actual = linmodest(lm_data['predictors'], lm_data['response']) 
    
    assert np.allclose(actual['coef'], lm_data['expected']['coef'])
    assert np.allclose(actual['vcov'], lm_data['expected']['vcov'])
    assert np.allclose(actual['sigma'], lm_data['expected']['sigma'])
    assert np.allclose(actual['df'], lm_data['expected']['df'])
