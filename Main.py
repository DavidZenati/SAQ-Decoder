"""
Implementation of "SAQ: Stabilizer-Aware Quantum Error Correction Decoder"
"""

from __future__ import print_function
import argparse
import logging
import random
import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime

from Codes import *
from CPND import *
from Model import SAQ_Transformer
import time

import galois
GF2 = galois.GF(2)
##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


##################################################################
class QECC_Dataset(data.Dataset):
    def __init__(self, code, ps, len, args):
        self.code = code
        self.ps = ps
        self.len = len
        self.logic_matrix = code.logic_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1).clone().cpu()
        self.zero_cw = torch.zeros((self.pc_matrix.shape[0])).long()
        self.noise_method = self.independent_noise if args.noise_type == 'independent' else self.depolarization_noise
        self.args = args

    def independent_noise(self, pp=None):
        pp = random.choice(self.ps) if pp is None else pp
        return np.random.binomial(1, pp, self.pc_matrix.shape[0])

    def depolarization_noise(self, pp=None):
        ## See original noise definition in https://github.com/Krastanov/neural-decoder/
        pp = random.choice(self.ps) if pp is None else pp
        out_dimZ = out_dimX = self.pc_matrix.shape[0] // 2

        def makeflips(q):
            q = q / 3.
            flips = np.zeros((out_dimZ + out_dimX,), dtype=np.dtype('b'))
            rand = np.random.rand(out_dimZ or out_dimX)
            both_flips = (2 * q <= rand) & (rand < 3 * q)
            ###
            x_flips = rand < q
            flips[:out_dimZ] ^= x_flips
            flips[:out_dimZ] ^= both_flips
            ###
            z_flips = (q <= rand) & (rand < 2 * q)
            flips[out_dimZ:out_dimZ + out_dimX] ^= z_flips
            flips[out_dimZ:out_dimZ + out_dimX] ^= both_flips
            return flips

        flips = makeflips(pp)
        while not np.any(flips):
            flips = makeflips(pp)
        return flips * 1.

    def __getitem__(self, index):
        x = self.zero_cw
        pp = random.choice(self.ps)

        z = torch.from_numpy(self.noise_method(pp))
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(z.long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return x.float(), z.float(), y.float(), (magnitude * 0 + 1).float(), syndrome.float()

    def __len__(self):
        return self.len


##################################################################
##################################################################
class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return grad_output * (torch.abs(x[0]) <= 1)


def binarization(y):
    return sign_to_bin(Binarization.apply(y))

def logical_flipped(L, x):
    return torch.matmul(x.float(), L.float()) % 2

def diff_GF2_mul(H, x):
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x

    tmp = bin_to_sign(H_bin.unsqueeze(0) * x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp, 1)
    tmp = sign_to_bin(tmp)
    return tmp

def _bits_to_index(bits: torch.Tensor) -> torch.Tensor:
    """bits [B,k] → integer class index [B]   (LSB = bits[:,0])."""
    weights = 2 ** torch.arange(bits.size(1), device=bits.device)
    return (bits * weights).sum(dim=1).long()
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ler_L = cum_ler_S = cum_samples = 0
    cum_loss1 = cum_loss2 = cum_loss3 = 0
    t = time.time()
    for batch_idx, (x, z, _, _, syndrome) in enumerate(train_loader):
        syndrome = syndrome.to(device)
        out_S, out_L, out_LP = model(syndrome)
        loss1, loss2, loss3 = model.module.loss(out_S, out_L, out_LP, z.to(device))


        ###########
        loss = args.lambda_loss_ent * loss1 + args.lambda_loss_lp * loss2 + args.lambda_loss_lc * loss3
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        out_S = sign_to_bin(torch.sign(out_S))
        pred_class = out_L.argmax(dim=1)
        true_class = _bits_to_index(logical_flipped(train_loader.dataset.logic_matrix, z.to(device)))
        ler_L = (pred_class != true_class).float().mean().item()
        ler_S = FER(logical_flipped(train_loader.dataset.logic_matrix, out_S), logical_flipped(train_loader.dataset.logic_matrix, z.to(device)))
        #################################################################################

        cum_loss += loss.item() * z.shape[0]
        #
        cum_loss1 += loss1.item() * z.shape[0]
        cum_loss2 += loss2.item() * z.shape[0]
        cum_loss3 += loss3.item() * z.shape[0]

        #
        cum_ler_L += ler_L * z.shape[0]
        cum_ler_S += ler_S * z.shape[0]
        cum_samples += z.shape[0]
        #
        if (batch_idx + 1) % (len(train_loader) // 2) == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e} LER_L ={cum_ler_L / cum_samples:.3e} LER_S={cum_ler_S / cum_samples:.3e}')
            logging.info(
                f'***Loss={cum_loss / cum_samples:.5e} Loss LC={cum_loss1 / cum_samples:.5e} Loss MinEnt={cum_loss3 / cum_samples:.5e} Loss LP={cum_loss2 / cum_samples:.5e} ')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ler_L / cum_samples, cum_ler_S / cum_samples


##################################################################

def test(model, device, test_loader_list, ps_range_test, args, cum_count_lim=100000):
    model.eval()
    test_loss_ler_logical_list, test_loss_ler_list, cum_samples_all = [], [], []
    t = time.time()
    #################
    if args.code_type=='toric':
        H_hat =GF2(np.vstack([args.code.pc_matrix.to('cpu').numpy()[:-1, :], args.code.logic_matrix.to('cpu').numpy()]))
        H_test = args.code.pc_matrix[:-1, :]
    elif args.code_type=='rotated_surface':
        H_hat =GF2(np.vstack([args.code.pc_matrix.to('cpu').numpy(), args.code.logic_matrix.to('cpu').numpy()]))
        H_test = args.code.pc_matrix
    else:
        raise ValueError("Unrecognized Code Type")
    B = exact_left_inverse(H_hat)
    assert (H_hat @ B == np.identity(H_hat.shape[0])).all(), "Left-inverse check failed"
    N = kernel_basis(H_hat)
    #################
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ler_logical= test_ler = cum_count = 0.
            while True:
                (x, z, _, _, syndrome) = next(iter(test_loader))
                out_S, out_L, out_LP = model(syndrome)
                _ = model.module.loss(out_S, out_L, out_LP, z.to(device))
                out_S = sign_to_bin(torch.sign(out_S))
                pred_class = out_L.argmax(dim=1)
                true_class = _bits_to_index(logical_flipped(test_loader.dataset.logic_matrix, z.to(device)))
                test_ler_logical += (pred_class != true_class).float().mean().item()* z.shape[0]

                ## CPND ##
                p = out_S.sigmoid()  # shape (n,)  – probabilities in [0,1]
                w = -torch.log(p / (1 - p))
                w_np = w.detach().cpu().numpy()
                e_hat = logical_flipped(torch.tensor(H_hat).T.to(device), out_S)
                if args.code_type == 'toric':
                    s = sign_to_bin(syndrome[:, :-1]).to(device)
                elif args.code_type == 'rotated_surface':
                    s = sign_to_bin(syndrome).to(device)
                logical = logits_to_logical_bits(out_L, test_loader.dataset.logic_matrix.size(1))

                # projection
                y = torch.cat([s, logical], dim=1) + e_hat % 2
                e0 = (logical_flipped(torch.tensor(B).T.to(device), y) + out_S) % 2
                ok_s = torch.all(logical_flipped(H_test.T.to(device), e0) == s)
                ok_l = torch.all(logical_flipped(test_loader.dataset.logic_matrix, e0) == logical)
                if not (ok_s and ok_l):
                    raise RuntimeError("decoding produced an inconsistent operator")
                # nullspace descent
                e_final = greedy_nullspace_refine(e0, N, w_np)
                ok_s_final = torch.all(logical_flipped(H_test.T.to(device), e_final) == s)
                ok_l_final = torch.all(logical_flipped(test_loader.dataset.logic_matrix, e_final) == logical)
                if not (ok_s_final and ok_l_final):
                    raise RuntimeError("decoding produced an inconsistent operator")

                ##########
                test_ler += FER(logical_flipped(test_loader.dataset.logic_matrix, out_S), logical_flipped(test_loader.dataset.logic_matrix, z.to(device))) * z.shape[0]
                cum_count += z.shape[0]
                if cum_count > cum_count_lim:
                    break
            cum_samples_all.append(cum_count)
            test_loss_ler_logical_list.append(test_ler_logical / cum_count)
            test_loss_ler_list.append(test_ler / cum_count)
            print(f'Test p={ps_range_test[ii]:.3e}, LER_L={test_loss_ler_logical_list[-1]:.3e}, LER_S={test_loss_ler_list[-1]:.3e}')
        ###
        logging.info('Test LER_S  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_list, ps_range_test))]))
        logging.info('Test LER_L ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_logical_list, ps_range_test))]))
        logging.info(f'Mean LER_S = {np.mean(test_loss_ler_list):.3e}, Mean Ler_L = {np.mean(test_loss_ler_logical_list):.3e}')
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ler_logical_list, test_loss_ler_list


##################################################################
##################################################################
##################################################################


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.code.logic_matrix = args.code.logic_matrix.to(device)
    args.code.pc_matrix = args.code.pc_matrix.to(device)
    code = args.code

    #################################
    model = SAQ_Transformer(args, dropout=0).to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    ps_test = np.linspace(args.lower_phy_err, args.upper_phy_err, 9)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, args.upper_phy_err, 9)

    ps_train = ps_test

    train_dataloader = DataLoader(QECC_Dataset(code, ps_train, len=args.batch_size * args.batch_num, args=args),
                                  batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(QECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size), args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers)
                            for ii in range(len(ps_test))]
    #################################
    best_ler_log = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ler_log, ler = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()
        torch.save(model, os.path.join(args.path, 'last_model'))
        if ler_log < best_ler_log:
            best_ler_log = ler_log
            torch.save(model, os.path.join(args.path, 'best_model'))
            logging.info('Model Saved')
        if epoch % 60 == 0 or epoch in [1, args.epochs]:
            test(model, device, test_dataloader_list, ps_test, args)
    model = torch.load(
        os.path.join(args.path, 'best_model'),
        weights_only=False  # revert to the ≤2.5 default
    ).to(device)
    logging.info('Best model loaded')
    ps_test = np.linspace(0.01, 0.2, 18)
    if args.noise_type == 'depolarization':
        ps_test = np.linspace(0.05, 0.2, 18)
    ###
    test_dataloader_list = [DataLoader(QECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size), args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers)
                            for ii in range(len(ps_test))]
    test(model, device, test_dataloader_list, ps_test, args)


##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SAQ Decoder')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_num', type=int, default=5000)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=5e-8)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='toric', choices=['toric', 'rotated_surface'])
    parser.add_argument('--code_L', type=int, default=4, help='Lattice length')
    parser.add_argument('--noise_type', type=str, default='independent', choices=['independent', 'depolarization'],
                        help='Noise model')
    parser.add_argument('--upper_phy_err', type=float, default=0.2, help='Upper physical error rate boundary')
    parser.add_argument('--lower_phy_err', type=float, default=0.01, help='Lower physical error rate boundary')

    # Model args
    parser.add_argument('--N_dec', type=int, default=6, help='Number of SAQ-Decoder self-attention modules')
    parser.add_argument('--d_model', type=int, default=128, help='SAQ-Decoder dimension')
    parser.add_argument('--h', type=int, default=16, help='Number of heads')

    # Loss args
    parser.add_argument('--lambda_loss_ent', type=float, default=1.0, help='Minimum entropy loss regularization')
    parser.add_argument('--lambda_loss_lc', type=float, default=1.0, help='Logical class loss regularization')
    parser.add_argument('--lambda_loss_lp', type=float, default=0.2, help='Logical priors loss regularization')

    # ablation args
    parser.add_argument('--no_mask', type=int, default=0)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    code = Code()
    H, L = eval(f'Get_{args.code_type}_Code')(args.code_L, full_H=args.noise_type == 'depolarization')
    code.logic_matrix = torch.from_numpy(L).long()
    code.pc_matrix = torch.from_numpy(H).long()
    code.n = code.pc_matrix.shape[1]
    code.m = code.pc_matrix.shape[0]
    code.k = code.logic_matrix.shape[0]
    code.code_type = args.code_type
    args.code = code
    ####################################################################
    model_dir = os.path.join('Final_Results_SAQ_Decoder', args.code_type,
                             'Code_L_' + str(args.code_L),
                             f'noise_model_{args.noise_type}',
                             datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    main(args)
