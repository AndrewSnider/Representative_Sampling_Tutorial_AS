import numpy as np
import pickle as pkl
import torch


class AtomCenteredDescriptor(object):
    def __init__(self, ind_map, cuttype, dtype=torch.float64):
        self.ind_map = ind_map
        self.nelems = len(ind_map)
        self.dtype = dtype

        if cuttype == 'cos':
            self.cutfunc = self.cutfunc_cos
        elif cuttype == 'tanh':
            self.cutfunc = self.cutfunc_tanh

        self.dstats = None
        
            
    def cutfunc_cos(self, r, Rc):
        r = torch.as_tensor(r, dtype=self.dtype)
        v = 0.5*(torch.cos(np.pi/Rc * r) + 1.0)
        v[r>Rc] = 0.0
        return v


    def cutfunc_tanh(self, r, Rc):
        r = torch.as_tensor(r, dtype=self.dtype)
        v = torch.tanh(1 - r/Rc)
        v = torch.pow(v, 3)
        v[r>Rc] = 0.0
        return v
    
    
    def calc_atom_descriptor(self, i, x, lat, weights=None):
        raise NotImplementedError


    def calc_descriptor(self, x, elems, lat):
        natoms = len(x)
        descriptors = []
        for i in range(natoms):
            idescrs = self.calc_atom_descriptor(i, x, elems, lat)
            descriptors += [idescrs]

        return torch.stack(descriptors)


    @torch.no_grad()
    def get_dstats(self, configs, cache=False):
        nconfigs = len(configs)
        nelems = self.nelems

        if self.dstats is not None: return self.dstats
        
        dstats = [{'ndescrs': 0, 'natoms': 0,
                   'avg': 0.0, 'std': 0.0,
                   'min': None, 'max': None} for i in range(nelems)]

        for i in range(nconfigs):
            x = configs[i]['pos']
            lat = configs[i]['lat']
            elems = configs[i]['elems']

            x = torch.as_tensor(x, dtype=self.dtype)
            if lat is not None:
                lat = torch.as_tensor(lat, dtype=self.dtype)

            G = self.calc_descriptor(x, elems, lat)
            

            elems = [self.ind_map[ielem] for ielem in configs[i]['elems']]
            elems = torch.as_tensor(elems, dtype=torch.int32)
            for j in range(nelems):
                jG = G[elems == j]
                dstats[j]['natoms'] += len(jG)
                dstats[j]['avg'] = dstats[j]['avg'] + torch.sum(jG, dim=0)
                dstats[j]['std'] = dstats[j]['std'] + torch.sum(torch.pow(jG, 2), dim=0)
                if dstats[j]['min'] is None:
                    dstats[j]['min'] = torch.min(jG, dim=0).values
                    dstats[j]['max'] = torch.max(jG, dim=0).values
                else:
                    jG_min = torch.min(jG, dim=0).values
                    jG_max = torch.max(jG, dim=0).values
                    minmask = jG_min < dstats[j]['min']
                    maxmask = jG_max > dstats[j]['max']
                    dstats[j]['min'][minmask] = jG_min[minmask]
                    dstats[j]['max'][maxmask] = jG_max[maxmask]

        for j in range(nelems):
            dstats[j]['avg'] /= dstats[j]['natoms']
            dstats[j]['std'] /= dstats[j]['natoms']
            dstats[j]['std'] = dstats[j]['std'] - torch.pow(dstats[j]['avg'], 2)
            dstats[j]['std'] = torch.sqrt(dstats[j]['std'])
            dstats[j]['ndescrs'] = len(dstats[j]['avg'])
            
        self.dstats = dstats
        return dstats


class ChebyshevDescriptor(AtomCenteredDescriptor):
    def __init__(self, rad_Rc, ang_Rc, rad_N, ang_N, wt_map,
                 ind_map, cuttype, dtype=torch.float64):
        self.rad_Rc = rad_Rc
        self.ang_Rc = ang_Rc
        self.rad_N = rad_N
        self.ang_N = ang_N
        self.wt_map = wt_map
        super(ChebyshevDescriptor, self).__init__(ind_map, cuttype, dtype=dtype)

    def chebpolynom(self, x, deg):
        x = torch.as_tensor(x, dtype=torch.float64)
        if deg == 0:
            return torch.ones_like(x, dtype=torch.float64)
        elif deg == 1:
            return x

        v = [x * 0.0 + 1.0]
        x2 = 2.0 * x
        v += [x]

        for i in range(2, deg+1):
            v += [v[i-1] * x2 - v[i-2]]
        return torch.stack(v)

    def calc_atom_rads(self, rmag, weights=None):
        fij = self.cutfunc(rmag, self.rad_Rc)
        T = self.chebpolynom(2.0*rmag[:, None]/self.rad_Rc - 1.0, self.rad_N)
        T = torch.squeeze(T)

        radial = T * fij
        if weights is not None:
            radial = torch.cat((radial, radial * weights), dim=0)

        return torch.sum(radial, dim=1)

    def calc_atom_angs(self, rij, rmag, weights=None):        
        n = len(rmag)
        fij = self.cutfunc(rmag, self.ang_Rc)

        prod_fij_fik = torch.matmul(fij[:, None], fij[None, :])
        prod_fij_fik = prod_fij_fik - torch.eye(n, n)*prod_fij_fik

        numer = torch.matmul(rij, rij.T)
        denom = torch.matmul(rmag[:, None], rmag[None, :])

        costheta = numer / denom
        T = self.chebpolynom(costheta, self.ang_N)
        angular = T * prod_fij_fik

        if weights is not None:
            wjk = torch.matmul(weights[:, None], weights[None, :])
            angular = torch.cat((angular, angular * wjk), dim=0)

        return torch.sum(angular, dim=(1, 2)) / 2.0


    def calc_atom_descriptor(self, i, x, elems, lat):
        weights = [self.wt_map[i] for i in elems]
        weights = torch.as_tensor(weights, dtype=self.dtype)
        
        rij = torch.cat((x[:i], x[i+1:]), dim=0) - x[i]
        if lat is not None:
            rij = rij - torch.round(rij / lat[None,:])*lat[None,:] 
        rmag = torch.sqrt(torch.sum(rij**2, dim=1))

        rad_inds = rmag < self.rad_Rc
        ang_inds = rmag < self.ang_Rc

        if weights is not None:
            iweights = torch.cat((weights[:i], weights[i+1:]))
            w_rad = iweights[rad_inds]
            w_ang = iweights[ang_inds]
        else:
            w_rad=None
            w_ang=None

        rads = self.calc_atom_rads(rmag[rad_inds], weights=w_rad)
        angs = self.calc_atom_angs(rij[ang_inds, :], rmag[ang_inds], weights=w_ang)

        rhalf = self.rad_N + 1
        ahalf = self.ang_N + 1
        return torch.cat((rads[:rhalf], angs[:ahalf],
                          rads[rhalf:], angs[ahalf:]), dim=0)


    
######### Example usage below ##############

# Setup Chebyshev descriptor parameters for pCT-
wt_map = {'H': -2.0, 'C': -1.0, 'O': 1.0, 'S': 2.0}
ind_map = {'H': 0, 'C': 1, 'O': 2, 'S': 3}
ang_Rc, rad_Rc = 3.00, 6.00
ang_N, rad_N = 5, 15
descrs = ChebyshevDescriptor(rad_Rc, ang_Rc, rad_N, ang_N, wt_map, ind_map, 'cos')

# Load pCT- in vacuum dataset
configs = pkl.load(open('configs.p', 'rb'))

# Create a pytorch tensor of the atomic positions for the first configuration
x = torch.as_tensor(configs[0]['pos'], dtype=torch.float64)

# Calculate Chebyshev descriptors for the first configuration in the dataset
# Note 'lat' is None to indicate that we are note doing periodic boundary conditions
# G is (natoms x ndescriptors)
G = descrs.calc_descriptor(x, configs[0]['elems'], configs[0]['lat'])
