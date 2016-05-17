import lmfit
import ttvfast.models
import pandas as pd
import numpy as np
import copy
model_basis = "mu per e inc longnode w lambda"
supported_bases = [
    model_basis,
    "mu per secosw sesinw inc longnode tc",
]

class Parameters(lmfit.Parameters):
    """
    Create a parameters dictionary

    Args:
        nplanet (int): number of planets
        basis (str): one of:
            - "mu per e inc longnode w lambda"
            - "mu per secosw sesinw inc longnode tc"
    """
    def __init__(self, nplanet, basis):
        super(Parameters, self).__init__()
        self.nplanet = nplanet
        self.basis = basis
        self.param_model0 = lmfit.Parameters()
        
        assert supported_bases.count(basis), "basis not supported"
        for i in range(1, self.nplanet + 1):
            for variable in basis.split():
                self.add("{}{}".format(variable,i))

            for variable in model_basis.split():
                self.param_model0.add("{}{}".format(variable,i))
            
    def to_model_basis(self):
        """
        Convert parameters to fitting basis
        """
        if self.basis==model_basis:
            return self

        param_model = self.param_model0
        for i in range(1, self.nplanet + 1):
            # convert into fitting basis
            if self.basis=="mu per secosw sesinw inc longnode tc":
                mu = self['mu%i' % i]._val
                per = self['per%i' % i]._val
                secosw = self['secosw%i' % i]._val
                sesinw = self['sesinw%i' % i]._val
                inc = self['inc%i' % i]._val
                longnode = self['longnode%i' % i]._val

                e = np.sqrt(secosw**2 + sesinw**2)
                w = np.arctan2(sesinw,secosw)

            param_model['mu%i' % i]._val= mu
            param_model['per%i' % i]._val = per
            param_model['e%i' % i]._val = e
            param_model['inc%i' % i]._val = inc
            param_model['longnode%i' % i]._val = longnode
            param_model['w%i' % i]._val = w

            ## To do: actually compute from time of first transit.
            param_model['lambda%i' % i]._val = 100.0
 

        return param_model

def omc(df):
    pfit = np.polyfit(df.i_epoch,df.times,1)
    _omc = np.array(df['times'] - np.polyval(pfit,df.i_epoch))
    return pfit[0],pfit[1],_omc

def parse_ttvfast(results, nplanet):
    columns = 'i_planet i_epoch times rsky vsky'.split()
    df = pd.DataFrame(zip(*results['positions']),columns=columns)
    df = df[~((df.index >= nplanet) & (df.i_epoch==0))]

    dfout = []
    for i in range(nplanet):
        dftemp = df[df.i_planet==i] 
        per,t0, dftemp['omc'] = omc(dftemp)
        dfout.append(dftemp)

    return dfout        

class Model(object):
    def __init__(self, tstart, dt, tstop, rv_times=None):
        self.tstart = tstart
        self.dt = dt
        self.tstop = tstop
        self.rv_times = rv_times
        self.stellar_mass = 1.0

    def __call__(self, params):
        planets = []
        params_model = params.to_model_basis()
        for i in range(1, params.nplanet+1):
            planet_params = [
                params_model['mu%i' % i].value,
                params_model['per%i' % i].value,
                params_model['e%i' % i].value,
                params_model['inc%i' % i].value,
                params_model['longnode%i' % i].value,
                params_model['w%i' % i].value,
                params_model['lambda%i' % i].value,
            ]

            planet = ttvfast.models.Planet(*planet_params)
            planets.append( planet )

        results = ttvfast.ttvfast(
            planets, self.stellar_mass, self.tstart, self.dt, self.tstop
        )
        dfout = parse_ttvfast(results, params.nplanet)
        return dfout

class Likelihood(object):
    """
    Knows how to construct the call to TTVFast
    """
    def __init__(self):
        pass
    
