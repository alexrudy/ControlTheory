# -*- coding: utf-8 -*-
import pytest

import numpy as np

from .. import periodogram

def test_cosine_window():
    """Test the cosine window."""
    cw = periodogram.cosine_window(1024)
    assert isinstance(cw, np.ndarray)
    assert cw.shape == (1024,)
    
