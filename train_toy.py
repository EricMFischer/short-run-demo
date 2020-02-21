#############################################
# ## TRAIN EBM USING 2D TOY DISTRIBUTION ## #
#############################################

import torch as t
import json
import os
from nets import ToyNet
from utils import plot_diagnostics, ToyDataset
from pytorch_adam import Adam

# directory for experiment results
EXP_DIR = './out_toy/uniform_sgd_lr=5e-3_annealed_eps=1.5e-1_wt_decay=0.15/'
# json file with experiment config
CONFIG_FILE = './config_locker/toy_config.json'


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)

# make directory for saving results
if os.path.exists(EXP_DIR):
    # prevents overwriting old experiment folders by accident
    raise RuntimeError('Experiment folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
else:
    os.makedirs(EXP_DIR)
    for folder in ['checkpoints', 'landscape', 'plots', 'code', 'mcmc_chains']:
        os.mkdir(EXP_DIR + folder)

# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open(file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['train_toy.py', 'nets.py', 'utils.py', CONFIG_FILE]:
        save_file(file)
save_code()

# set seed for cpu and CUDA, get device
t.manual_seed(config['seed'])
if t.cuda.is_available():
    print('t.cuda is available')
    t.cuda.manual_seed_all(config['seed'])
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


########################
# ## TRAINING SETUP # ##
########################

print('Setting up network and optimizer...')
# set up network
net_bank = {'toy': ToyNet}
f = net_bank[config['net_type']]().to(device)
# set up optimizer
optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD}
# optim_bank = {'adam': Adam, 'sgd': t.optim.SGD}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    # scale learning rate according to langevin noise for invariant tuning
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2
if config['toy_type'] == 'gmm_2':
    optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'], weight_decay=0.15)
else:
    optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])

print('Processing data...')
# q.means: e.g. (3,2,1,1)
# q.xy_plot: e.g. (200), max=plot_val_max, min=-plot_val_max
# q.z_true_density: e.g. (200,200)
q = ToyDataset(config['toy_type'], config['toy_groups'], config['toy_sd'],
               config['toy_radius'], config['viz_res'], config['kde_bw'])


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# initialize persistent states from noise
if config['toy_type'] == 'gmm_2':
    s_t_0 = 2 * t.rand([config['s_t_0_size'], 2, 1, 1]).to(device) + 3
else:
    s_t_0 = 2 * t.rand([config['s_t_0_size'], 2, 1, 1]).to(device) - 1 # e.g. (10000,2,1,1)

# sample batch from given array of states
def sample_state_set(state_set, batch_size=config['batch_size']):
    rand_inds = t.randperm(state_set.shape[0])[0:batch_size]
    return state_set[rand_inds], rand_inds

# sample positive states from 2D toy distribution q
def sample_q(batch_size=config['batch_size']): return t.Tensor(q.sample_toy_data(batch_size)).to(device)

# visualize movement of MCMC chains during a training iteration
def plot_mcmc_chains(x_s_t, train_iter, mcmc_step):
    if train_iter+1 in [5000] and (mcmc_step+1) % 20 == 0:
        train_iter_dir = EXP_DIR+'mcmc_chains/'+'train_iter_{:>06d}/'.format(train_iter+1)
        if not os.path.exists(train_iter_dir):
            os.mkdir(train_iter_dir)
        q.plot_toy_density(True, f, config['epsilon'], x_s_t.detach(), EXP_DIR+'mcmc_chains/'+'train_iter_{:>06d}/mcmc_step_{:>06d}.pdf'.format(train_iter+1, mcmc_step+1), mcmc_chains=True)

# initialize and update states with langevin dynamics to obtain negative samples from MCMC distribution s_t
def sample_s_t(batch_size, L=config['num_mcmc_steps'], init_type=config['init_type'], update_s_t_0=True, train_iter=0):
    # get initial mcmc states for langevin updates ("persistent", "data", "uniform", or "gaussian")
    def sample_s_t_0():
        if init_type == 'persistent':
            return sample_state_set(s_t_0, batch_size)
        elif init_type == 'data':
            return sample_q(batch_size), None
        elif init_type == 'uniform':
            if config['toy_type'] == 'gmm_2':
                result = (config['noise_init_factor'] * 2 * t.rand([batch_size, 2, 1, 1]) + 3).to(device), None
            else:
                result = config['noise_init_factor'] * (2 * t.rand([batch_size, 2, 1, 1]) - 1).to(device), None
            return result
        elif init_type == 'gaussian':
            if config['toy_type'] == 'gmm_2':
                result = (config['noise_init_factor'] * t.randn([batch_size, 2, 1, 1]) + 5).to(device), None
            else:
                result = config['noise_init_factor'] * t.randn([batch_size, 2, 1, 1]).to(device), None
            return result
        else:
            raise RuntimeError('Invalid method for "init_type" (use "persistent", "data", "uniform", or "gaussian")')

    # initialize MCMC samples
    x_s_t_0, s_t_0_inds = sample_s_t_0() # e.g. (100,2,1,1), None

    # iterative langevin updates of MCMC samples
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True) # e.g. (100,2,1,1)
    r_s_t = t.zeros(1).to(device)  # variable r_s_t (Section 3.2) to record average normalized gradient magnitude
    for ell in range(L):
        f_x_s_t = f(x_s_t) # e.g. (100)
        f_x_s_t_sum = f_x_s_t.sum() # scalar
        f_prime = t.autograd.grad(f_x_s_t_sum, [x_s_t])[0] # gradient magnitude wrt negative samples, e.g. (100,2,1,1)
        # noise = 0 if config['toy_type'] == 'gmm_2' else config['epsilon'] * t.randn_like(x_s_t)
        noise = config['epsilon'] * t.randn_like(x_s_t)
        x_s_t.data += - f_prime + noise # samples += - gradient + noise

        f_prime_norm = f_prime.view(f_prime.shape[0], -1).norm(dim=1) # normalized gradient magnitude, e.g. (100)
        r_s_t += f_prime_norm.mean() # scalar

        # visualize movement of MCMC chains during a training iteration
        plot_mcmc_chains(x_s_t, train_iter, ell)
        
    if init_type == 'persistent' and update_s_t_0:
        # update persistent state bank
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    # each time we draw negative samples, we return:
    # 1) the samples, e.g. (100,2,1,1)
    # 2) the normalized gradient magnitude averaged over 100 negative samples and over L MCMC steps, lending a scalar
    return x_s_t.detach(), r_s_t.squeeze() / L


#######################
# ## TRAINING LOOP ## #
#######################

# containers for diagnostic records (see Section 3)
# energy difference between positive and negative samples, e.g. (200000)
d_s_t_record = t.zeros(config['num_train_iters']).to(device)
# average state gradient magnitude along Langevin path, e.g. (200000)
r_s_t_record = t.zeros(config['num_train_iters']).to(device)

print('Training has started.')
for i in range(config['num_train_iters']):
    # obtain positive and negative samples
    x_q = sample_q() # positive samples, e.g. (100,2,1,1)
    x_s_t, r_s_t = sample_s_t(batch_size=config['batch_size'], train_iter=i) # e.g. (100,2,1,1), scalar

    # calculate ML computational loss d_s_t (Section 3) for data and shortrun samples
    d_s_t = f(x_q).mean() - f(x_s_t).mean()
    if config['epsilon'] > 0:
        # scale loss with the langevin implementation
        d_s_t *= 2 / (config['epsilon'] ** 2)

    # stochastic gradient ML update for model weights
    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    # record diagnostics
    d_s_t_record[i] = d_s_t.detach().data
    r_s_t_record[i] = r_s_t

    # anneal learning rate
    for lr_gp in optim.param_groups:
        lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    # if toy_type is gmm_2, anneal SGD learning rate by steps, e.g. [0.05, 0.037625, 0.02525, 0.012875, 0.0005]
    if config['toy_type'] == 'gmm_2' and i+1 in [2500,5000,7500,10000]: # [25000, 50000, 75000, 100000]
        # lr_step = (config['lr_init'] - (config['lr_init']/100)) / 4
        lr_step = (config['lr_init'] - (config['lr_init']/1000)) / 4
        for lr_gp in optim.param_groups:
            lr_gp['lr'] -= lr_step

    # print and save learning info
    if (i + 1) == 1 or (i + 1) % config['log_info_freq'] == 0:
        print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}'.format(i+1, d_s_t.detach().data, r_s_t))
        # save network weights
        t.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i+1))
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t
        if (i + 1) > 1:
            plot_diagnostics(i, d_s_t_record, r_s_t_record, EXP_DIR + 'plots/')

    # visualize density and log-density for groundtruth, learned energy, and short-run distributions
    if (i + 1) % config['log_viz_freq'] == 0:
        print('{:>6}   Visualizing true density, learned density, and short-run KDE.'.format(i+1))
        # draw negative samples for kernel density estimate
        x_kde = sample_s_t(batch_size=config['batch_size_kde'], update_s_t_0=False, train_iter=i)[0] # (10000,2,1,1)
        q.plot_toy_density(True, f, config['epsilon'], x_kde, EXP_DIR+'landscape/'+'toy_viz_{:>06d}.pdf'.format(i+1))
        print('{:>6}   Visualizations saved.'.format(i + 1))
