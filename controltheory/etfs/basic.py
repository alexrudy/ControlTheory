# -*- coding: utf-8 -*-
"""
Basic model implementation
"""
import numpy as np
import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter

class TransferFunction(Fittable1DModel):
    """Model of a basic transfer function, 
    accounting for a WFS and DM in a standard single frame stare configuration."""
    
    inputs = ('frequency',)
    outputs = ('y',)
    
    tau = Parameter(min=0.0, doc="The delay, in seconds.")
    gain = Parameter(min=0.0, max=1.0, 
                     doc="System gain")
    ln_c = Parameter(min=-1e2, max=0.0, default=np.log(1.0 - 0.9),
                     doc="Natural logarithm of the integrator constant")
    rate = Parameter(fixed=True, default=1000,
                     doc="System sampling frequency, in Hz")
    
    def __init__(self, *args, **kwargs):
        integrator = kwargs.pop('integrator', None)
        super(TransferFunction, self).__init__(*args, **kwargs)
        if integrator is not None:
            self.integrator = integrator
    
    @property
    def integrator(self):
        """Return the integrator value."""
        return 1.0 - np.exp(self.ln_c.value)
        
    @integrator.setter
    def integrator(self, value):
        """Set the intergrator."""
        self.ln_c.value = np.log(1.0 - value)
    
    @classmethod
    def _T(cls, rate):
        return 1.0 / rate
    
    @classmethod
    def _s(cls, frequency):
        """Compute s"""
        return 1j * 2.0 * np.pi * frequency
    
    @classmethod
    def evaluate(cls, frequency, tau, gain, ln_c, rate):
        """Evaluate a transfer function."""
        integrator = 1.0 - np.exp(ln_c)
        
        s = cls._s(frequency)
        T = cls._T(rate)
        s_zeros = (s == 0)
        denom_term = T * s
        zinv = np.exp(-1.0 * T * s)
        
        
        # Anticipate divide by zero problems.
        denom_term[s_zeros] = 1.0
        
        hdw_cont = (1.0 - zinv) / denom_term
        hdw_cont[sz] = 1.0
        
        delay_cont = np.exp(-1.0 * tau * s)
        
        # Delay + WFS Stare + DM Stare
        delay_term = delay_cont * hdw_cont * hdw_cont
        
        # C(z) due to integrator.
        cofz = gain / (1.0 - integrator * zinv)
        
        # All together.
        return np.abs(1.0 / (1.0 + delay_term * cofz) ** 2.0)
