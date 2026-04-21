"""
SAQ: Stabilizer-Aware Quantum Error Correction Decoder
"""

import numpy as np
import torch
from scipy.sparse import hstack, kron, eye, csr_matrix, block_diag
import itertools
import scipy.linalg
import galois


class Code():
    pass

class ToricCode:
    '''
    From https://github.com/Krastanov/neural-decoder/
        Lattice:
        X00--Q00--X01--Q01--X02...
         |         |         |
        Q10  Z00  Q11  Z01  Q12
         |         |         |
        X10--Q20--X11--Q21--X12...
         .         .         .
    '''
    def __init__(self, L):
        '''Toric code of ``2 L**2`` physical qubits and distance ``L``.'''
        self.L = L
        self.Xflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where an X error occured
        self.Zflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where a  Z error occured
        self._Xstab = np.empty((L,L), dtype=np.dtype('b'))
        self._Zstab = np.empty((L,L), dtype=np.dtype('b'))

    @property
    def flatXflips2Zstab(self):
        L = self.L
        _flatXflips2Zstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatXflips2Zstab[i*L+j, (2*i  )%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j+1)%L] = 1
        return _flatXflips2Zstab

    @property
    def flatZflips2Xstab(self):
        L = self.L
        _flatZflips2Xstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+1)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+3)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j+1)%L] = 1
        return _flatZflips2Xstab

    @property
    def flatXflips2Zerr(self):
        L = self.L
        _flatXflips2Zerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatXflips2Zerr[0, (2*k+1)%(2*L)*L+(0  )%L] = 1
            _flatXflips2Zerr[1, (2*0  )%(2*L)*L+(k  )%L] = 1
        return _flatXflips2Zerr

    @property
    def flatZflips2Xerr(self):
        L = self.L
        _flatZflips2Xerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatZflips2Xerr[0, (2*0+1)%(2*L)*L+(k  )%L] = 1
            _flatZflips2Xerr[1, (2*k  )%(2*L)*L+(0  )%L] = 1
        return _flatZflips2Xerr

    def H(self, Z=True, X=False):
        H = []
        if Z:
            H.append(self.flatXflips2Zstab) #[:-1,:]
        if X:
            H.append(self.flatZflips2Xstab) #[:-1,:]
        H = scipy.linalg.block_diag(*H)
        return H

    def E(self, Z=True, X=False):
        E = []
        if Z:
            E.append(self.flatXflips2Zerr) # Original: flatXflips2Zerr
        if X:
            E.append(self.flatZflips2Xerr) # Original: flatZflips2Xerr
        E = scipy.linalg.block_diag(*E)
        return E
##########################################################################################
class RotatedSurfaceCode:
    def __init__(self, L):
        '''Rotated surface code of `` L**2`` physical qubits and distance ``L``.'''
        self.L = L
        self.num_stabilizers = (L - 1) * (L + 1) // 2
        self.num_qubits = L * L
        self.H_X = np.zeros((self.num_stabilizers, self.num_qubits), dtype=int)
        self.H_Z = np.zeros((self.num_stabilizers, self.num_qubits), dtype=int)
        self.L_X = np.zeros((1, self.num_qubits), dtype=int)
        self.L_Z = np.zeros((1, self.num_qubits), dtype=int)
        self.generate_parity_check_matrices()
        self.generate_logical_operators()
    def generate_parity_check_matrices(self):
        stabilizerX_idx = stabilizerZ_idx = 0
        L = self.L
        # Loop for each l and k to create stabilizers
        for l in range(L - 1):
            z_stab = [
                l * L + (L - 1) * (l % 2),
                (l + 1) * L + (L - 1) * (l % 2)
            ]
            self.add_stabilizer(self.H_Z, stabilizerZ_idx, z_stab)
            stabilizerZ_idx += 1
            for k in range((L - 1) // 2):
                if l == 0 or l == L - 2:
                    x_stab = [
                        (l + 1) * (l + 1 + l % 2) + 2 * k,
                        (l + 1) * (l + 1 + l % 2) + 1 + 2 * k
                    ]
                    self.add_stabilizer(self.H_X, stabilizerX_idx, x_stab)
                    stabilizerX_idx += 1
                intercep_x_k = l % 2 + 2 * k
                intercep_z_k = (l + 1) % 2 + 2 * k
                # X-stabilizer (l*L + intercep_x_k, l*L +1+ intercep_x_k, L*(l+1) + intercep_x_k, L*(l+1) +1+ intercep_x_k)
                x_stab = [
                    l * L + intercep_x_k,
                    l * L + 1 + intercep_x_k,
                    (l + 1) * L + intercep_x_k,
                    (l + 1) * L + 1 + intercep_x_k
                ]
                self.add_stabilizer(self.H_X, stabilizerX_idx, x_stab)
                stabilizerX_idx += 1
                # Z-stabilizer (l*L + intercep_z_k, l*L +1+ intercep_z_k, L*(l+1) + intercep_z_k, L*(l+1) +1+ intercep_z_k)
                z_stab = [
                    l * L + intercep_z_k,
                    l * L + 1 + intercep_z_k,
                    (l + 1) * L + intercep_z_k,
                    (l + 1) * L + 1 + intercep_z_k
                ]
                self.add_stabilizer(self.H_Z, stabilizerZ_idx, z_stab)
                stabilizerZ_idx += 1
    def add_stabilizer(self, H, stabilizer_idx, qubit_indices):
        """ Helper function to add a stabilizer row to the parity check matrix """
        for qubit_idx in qubit_indices:
            H[stabilizer_idx, qubit_idx] = 1
    def generate_logical_operators(self):
        for l in range(self.L):
            self.L_Z[:, l] = 1
            self.L_X[:, l * self.L] = 1
    def H(self, Z=True, X=False):
        H = []
        if Z:
            H.append(self.H_X) #Orig: H_Z
        if X:
            H.append(self.H_Z)  #Orig: H_X
        H = scipy.linalg.block_diag(*H)
        return H
    def E(self, Z=True, X=False):
        E = []
        if Z:
            E.append(self.L_X) #Orig: L_X
        if X:
            E.append(self.L_Z) #Orig: L_Z
        E = scipy.linalg.block_diag(*E)
        return E
##########################################################################################

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def SCR(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

#############################################
def Get_toric_Code(L,full_H=False):
    toric = ToricCode(L)
    Hx = toric.H(Z=full_H,X=True)
    logX = toric.E(Z=full_H,X=True)
    return Hx, logX

def Get_rotated_surface_Code(L,full_H=False):
    rot_Surface = RotatedSurfaceCode(L)
    Hx = rot_Surface.H(Z=full_H,X=True)
    logX = rot_Surface.E(Z=full_H,X=True)
    return Hx, logX
#############################################
def gf2_rank(A):
    """
    Step 3 – compute rank over GF(2) with Gaussian elimination.
    Works on a *copy*; original A is unchanged.

    Parameters
    ----------
    A : ndarray (m , n)  -- entries 0/1

    Returns
    -------
    r : int              -- rank over GF(2)
    """
    A = A.copy() & 1  # make sure values are 0/1
    m, n = A.shape
    pivot_row = 0

    for col in range(n):
        # find a row with a '1' in this column
        rows_with_1 = np.where(A[pivot_row:, col] == 1)[0]
        if rows_with_1.size == 0:
            continue  # no pivot in this column
        row = pivot_row + rows_with_1[0]  # absolute index

        # swap pivot row upward
        if row != pivot_row:
            A[[pivot_row, row]] = A[[row, pivot_row]]

        # eliminate '1's in this column for all other rows
        for r in range(m):
            if r != pivot_row and A[r, col] == 1:
                A[r] ^= A[pivot_row]  # XOR row operation

        pivot_row += 1
        if pivot_row == m:  # full rank achieved
            break

    return pivot_row

#############################################
if __name__ == "__main__":
    Get_toric_Code(4)
    Get_rotated_surface_Code(3)
    class Code:
        pass

