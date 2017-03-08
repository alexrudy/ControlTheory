# -*- coding: utf-8 -*-
import pytest

import numpy as np

from .. import periodogram

def test_cosine_window():
    """Test the cosine window."""
    cw = periodogram.cosine_window(1024)
    assert isinstance(cw, np.ndarray)
    assert cw.shape == (1024,)
    
def test_extend_axes():
    """Test extend axes function."""
    data = np.ones((5,))
    edata = periodogram.extend_axes(data, 3, 1)
    assert edata.shape == (1, 5, 1)
    
    edata = periodogram.extend_axes(data, 4, -2)
    assert edata.shape == (1, 1, 5, 1)
    