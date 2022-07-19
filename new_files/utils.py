import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import Normflow
#import normflow as nf

# For plotting
import matplotlib.pyplot as plt


dtype = torch.FloatTensor
enable_cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

pdbfile = 'water.pdb' #'alanine-dipeptide.pdb'
num_atoms = 3 #22
forcefile = 'amber14/tip3pfb.xml' #'amber14-all.xml'

pklfile = 'water_img1024_dim25_nse1_blob0.001_q0.2.pkl' #'aldip_img1024_dim51_nse1_scl0.001_q1.pkl'
num_imgs = 1024
n_cells = 25 #51
noise_var = 1
blob_var = 0.001
Q = 0.2

num_flows = 25
beta = 10.0
num_epochs = 1000

kB = 1.3807e-26
Avo = 6.0221408e+23
T = 300
kBxAvoxT = kB*Avo*T

count = 1


# Utility functions for putting gaussian blobs and Cryo-EM image plotting

def gaussian_pdf(target, candidate):
    return torch.exp(-0.5*(torch.linalg.norm(target - candidate))/noise_var_powk)/torch.sqrt(2*torch.pi)

def create_posmatrix(Q=Q, n_cells=n_cells):
    # Grids of x and y points
    x = torch.linspace(-Q, Q, n_cells, device=device)
    y = torch.linspace(-Q, Q, n_cells, device=device)
    # Create a 2-D meshgrid of (x, y) coordinates.
    x, y = torch.meshgrid(x, y, indexing='xy')

    # Creating mixture of gaussians
    pos = torch.empty(x.shape + (2,), device=device).double()
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    return pos

pos = create_posmatrix().type(dtype).to(device)

def torch_createblobs(coords, Q=Q, n_cells=n_cells, blob_var=blob_var, noise_var=0, rot_mat=torch.eye(3, device=device), pos=pos):
    new_coords = coords @ rot_mat.T
    noise_matrix = 0 if noise_var == 0 else np.sqrt(noise_var)*torch.randn((n_cells, n_cells))
    normals = torch.zeros((n_cells, n_cells), device=device)

    for row in new_coords:
        normals += torch.exp(-(torch.norm(pos - row[0:2], dim=2)**2)/(2*(blob_var)))

    return normals/(2*np.pi*blob_var*len(new_coords)) + noise_matrix


def plot_cryoem_imgs(img_list, skip_by=1, Q=Q):
    for idx, img in enumerate(img_list):
        if idx % skip_by == 0:
            c = plt.imshow(img.cpu(), cmap ='gray', vmin = 0, vmax = np.abs(img.cpu()).max(),\
                            extent = [-Q, Q, -Q, Q], interpolation = 'nearest', origin = 'lower')
            plt.colorbar(c)

            plt.title(f'Cryo-EM Image {idx+1}', fontweight = "bold")
            plt.show()


def create_fastestpos(Q=Q, n_cells=n_cells, num_atoms=num_atoms, n_imgs=count):
    # Grids of x and y points
    x = torch.linspace(-Q, Q, n_cells, device=device)
    y = torch.linspace(-Q, Q, n_cells, device=device)
    # Create a 2-D meshgrid of (x, y) coordinates.
    x, y = torch.meshgrid(x, y, indexing='xy')

    # Creating mixture of gaussians
    pos = torch.empty(x.shape + (2,), device=device).double()
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    intermediate = torch.stack([pos]*num_atoms, dim = 3)
    return torch.stack([intermediate]*n_imgs, dim = 0)

fstpos = create_fastestpos().type(dtype).to(device)

def torch_fastestblobs(coord_list, Q=Q, n_cells=n_cells, blob_var=blob_var, noise_var=0, rot_list=None, fstpos=fstpos):
    new_coords = coord_list if rot_list==None else coord_list @ torch.transpose(rot_list, 1, 2)
    noise_matrix = 0 if noise_var == 0 else np.sqrt(noise_var)*torch.randn((n_cells, n_cells))
    row_broadcast = torch.transpose(new_coords[:, :, 0:2], 1, 2)[:, None, None, :, :]

    normals = torch.exp(-(torch.norm(fstpos - row_broadcast, dim=3)**2)/(2*(blob_var)))

    return torch.sum(normals, 3)/(np.sqrt(2*np.pi*blob_var)*len(new_coords)) + noise_matrix


class pot_energy(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        reshaped_input = input.reshape(num_atoms, 3)
        ctx.save_for_backward(reshaped_input)

        simulation.context.setPositions(reshaped_input.cpu().numpy())
        state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
        return torch.tensor(state.getPotentialEnergy()._value).to(device)


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        reshaped_input, = ctx.saved_tensors

        simulation.context.setPositions(reshaped_input.cpu().numpy())
        state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
        return -grad_output * torch.tensor(state.getForces(asNumpy=True)._value).reshape(-1, num_atoms*3).to(device)


def potential(vec):
    eq_length, k_length = 0.101181082494, 462750.4
    eq_angle, k_angle = 1.88754640288, 836.8

    bond1 = vec[0:3] - vec[3:6]
    bond2 = vec[0:3] - vec[6:9]

    bond_pot = k_length*((eq_length - torch.norm(bond1))**2 + (eq_length - torch.norm(bond2))**2)
    angle_pot = k_angle*(eq_angle - torch.acos(torch.dot(bond1, bond2)/(torch.norm(bond1)*torch.norm(bond2))))**2

    return angle_pot + bond_pot


def potential_fast(coord_list):
    eq_length, k_length = 0.101181082494, 462750.4
    eq_angle, k_angle = 1.88754640288, 836.8

    bond1 = coord_list[:, 0, :] - coord_list[:, 1, :] #vec[0:3] - vec[3:6]
    bond2 = coord_list[:, 0, :] - coord_list[:, 2, :] #vec[0:3] - vec[6:9]

    bond_pot = k_length*((eq_length - torch.norm(bond1, dim = 1))**2 + (eq_length - torch.norm(bond2, dim = 1))**2)
    angle_pot = k_angle*(eq_angle - torch.acos(torch.bmm(bond1[:, None, :], bond2[:, :, None])[:,0,0]/(torch.norm(bond1, dim = 1)*torch.norm(bond2, dim = 1))))**2

    return angle_pot + bond_pot