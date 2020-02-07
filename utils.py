# download Oxford Flowers 102, plotting functions, and toy dataset

import torch as t
import torchvision as tv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


##########################
# ## DOWNLOAD FLOWERS ## #
##########################
# Code with minor modification from https://github.com/microsoft/CNTK/tree/master/Examples/Image/DataSets/Flowers
# Original Version: Copyright (c) Microsoft Corporation

def download_flowers_data():
    import tarfile
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    dataset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/flowers/')
    if not os.path.exists(os.path.join(dataset_folder, "jpg")):
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        print('Downloading data from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ ...')
        tar_filename = os.path.join(dataset_folder, "102flowers.tgz")
        if not os.path.exists(tar_filename):
            urlretrieve("http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", tar_filename)

        # extract flower images from tar file
        print('Extracting ' + tar_filename + '...')
        tarfile.open(tar_filename).extractall(path=dataset_folder)

        # clean up
        os.remove(tar_filename)
        print('Done.')
    else:
        print('Data available at ' + dataset_folder)


##################
# ## PLOTTING ## #
##################

# visualize negative samples synthesized from energy
def plot_ims(p, x): tv.utils.save_image(t.clamp(x, -1., 1.), p, normalize=True, nrow=int(x.shape[0] ** 0.5))

# plot diagnostics for learning
def plot_diagnostics(batch, en_diffs, grad_mags, exp_dir, fontsize=10):
    # axis tick size
    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)
    fig = plt.figure()

    def plot_en_diff_and_grad_mag():
        # energy difference
        ax = fig.add_subplot(221)
        ax.plot(en_diffs[0:(batch+1)].data.cpu().numpy())
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Energy Difference', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$d_{s_t}$', fontsize=fontsize)
        # mean langevin gradient
        ax = fig.add_subplot(222)
        ax.plot(grad_mags[0:(batch+1)].data.cpu().numpy())
        ax.set_title('Average Langevin Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$r_{s_t}$', fontsize=fontsize)

    def plot_crosscorr_and_autocorr(t_gap_max=2000, max_lag=15, b_w=0.35):
        t_init = max(0, batch + 1 - t_gap_max)
        t_end = batch + 1
        t_gap = t_end - t_init
        max_lag = min(max_lag, t_gap - 1)
        # rescale energy diffs to unit mean square but leave uncentered
        en_rescale = en_diffs[t_init:t_end] / t.sqrt(t.sum(en_diffs[t_init:t_end] * en_diffs[t_init:t_end])/(t_gap-1))
        # normalize gradient magnitudes
        grad_rescale = (grad_mags[t_init:t_end]-t.mean(grad_mags[t_init:t_end]))/t.std(grad_mags[t_init:t_end])
        # cross-correlation and auto-correlations
        cross_corr = np.correlate(en_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        en_acorr = np.correlate(en_rescale.cpu().numpy(), en_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        grad_acorr = np.correlate(grad_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        # x values and indices for plotting
        x_corr = np.linspace(-max_lag, max_lag, 2 * max_lag + 1)
        x_acorr = np.linspace(0, max_lag, max_lag + 1)
        t_0_corr = int((len(cross_corr) - 1) / 2 - max_lag)
        t_0_acorr = int((len(cross_corr) - 1) / 2)

        # plot cross-correlation
        ax = fig.add_subplot(223)
        ax.bar(x_corr, cross_corr[t_0_corr:(t_0_corr + 2 * max_lag + 1)])
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Cross Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('lag', fontsize=fontsize)
        ax.set_ylabel('correlation', fontsize=fontsize)
        # plot auto-correlation
        ax = fig.add_subplot(224)
        ax.bar(x_acorr-b_w/2, en_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='en. diff. $d_{s_t}$')
        ax.bar(x_acorr+b_w/2, grad_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='grad. mag. $r_{s_t}}$')
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Auto-Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('lag', fontsize=fontsize)
        ax.set_ylabel('correlation', fontsize=fontsize)
        ax.legend(loc='upper right', fontsize=fontsize-4)

    # make diagnostic plots
    plot_en_diff_and_grad_mag()
    plot_crosscorr_and_autocorr()
    # save figure
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.savefig(os.path.join(exp_dir, 'diagnosis_plot.pdf'), format='pdf')
    plt.close()


#####################
# ## TOY DATASET ## #
#####################

class ToyDataset:
    # Background on Zhu Question 6:
    # If you take a bimodal Gaussian, and make one Gaussian much smaller by reducing its sigma variable, Zhu is afraid short-run MCMC
    # can match the 2 modes but not the ratio.
    # With an 80/20 ratio in a bimodal Guassian, if we count the chains which are attracted to each mode, we want the ratio to be 8/2.
    # We want the probability mass to shift correctly.
    # With persistent chains there is most likely no issue. But for short-run (noise initialized), it is good to double-check.

    # Note: Mitch eventually got good short-run results with noise initialization, but he couldn't use a uniform distribution;
    # he had to use a Gaussian. With just a uniform, it started in a box and got weird corners and edges.

    # With noise_init_factor, we can make the range of the uniform initialization larger, so it's less likely the probability mass
    # will concentrate in edges or a corner.
    # With noise_init_factor to shrink or enlarge our proposal uniform distribution range, it alleviates this issue.

    # Depending on how we choose 3 means, may need to change in config the initial proposal noise noise_init_factor.
    # noise_init_factor: how much we mutiply our means by when we draw samples of noise.
    # Example: with plot_val_max=4, just using Guassian noise does not lend good short-run results.
    # With an initial noise distribution too close to the center, it's difficult to capture data.
    # It's easier for widely spread data to shrink and converge than for narrowly spread data to enlarge to converge.

    # Tip: Be aware of the minimum std of your Gaussians, which is important for tuning the langevin dynamics of the long-run MCMC.
    # For every covariance matrix, what is the minimum eigenvalue and what is the minimum (used for tuning) over all 3 of those matrices?

    # GMM parameters: choose mean, std, covariance, weights to be diverse (means should be non-symmetric, noticeably different shapes)
    # 3 Guassian means:
    # 1) circle
    # 2) high weight, oblong convariance (skinny and tilted),
    # 3) low weight, oblong covariance (skinny and tilted in another way)

    def __init__(self, toy_type='gmm', toy_groups=8, toy_sd=0.15, toy_radius=1, viz_res=500, kde_bw=0.05):
        # import helper functions
        from scipy.stats import gaussian_kde
        from scipy.stats import multivariate_normal
        self.gaussian_kde = gaussian_kde
        self.mvn = multivariate_normal

        # toy dataset parameters
        self.toy_type = toy_type
        self.toy_groups = toy_groups

        # hardcode toy_sd, radius, and weights if toy type is 'gmm_2'
        self.toy_sd = toy_sd
        self.toy_radius = toy_radius
        self.weights = np.ones(toy_groups) / toy_groups

        if toy_type == 'gmm_2':
            self.toy_sd = [0.15, 0.15, 0.15]
            self.toy_radius = [0.7, 0.9, 1.]
            self.weights = [0.15, 0.2, 0.65]

        if toy_type == 'gmm':
            means_x = np.cos(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            means_y = np.sin(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            self.means = toy_radius * np.concatenate((means_x, means_y), axis=1)
        elif toy_type == 'gmm_2':
            accum_means = None
            mean_radius = [0., 0.375, 0.75]
            for i in range(self.toy_groups):
                mean_x = np.cos(2*np.pi*mean_radius[i]).reshape(1, 1, 1, 1)
                mean_y = np.sin(2*np.pi*mean_radius[i]).reshape(1, 1, 1, 1)
                if i == 0:
                    accum_means = self.toy_radius[i] * np.concatenate((mean_x, mean_y), axis=1)
                else:
                    mean_xy = self.toy_radius[i] * np.concatenate((mean_x, mean_y), axis=1)
                    accum_means = np.concatenate((accum_means, mean_xy), axis=0)
            self.means = accum_means # (3,2,1,1)
        else:
            self.means = None

        # ground truth density
        if self.toy_type == 'gmm':
            def true_density(x):
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k]*self.mvn.pdf(np.array([x[1], x[0]]), mean=self.means[k].squeeze(),
                                                            cov=(self.toy_sd**2)*np.eye(2))
                return density
        elif self.toy_type == 'gmm_2':
            def true_density(x):
                density = 0

                # val = np.sqrt(2)/2
                # rotate_45 = [[val, -val], [val, val]]
                # rotate_135 = [[-val, -val], [val, -val]]
                # A = rotate_45 @ np.diag([.005,.05]) @ np.linalg.inv(rotate_45)
                # B = rotate_135 @ np.diag([.02,.05]) @ np.linalg.inv(rotate_135)
                # covariances = [(sd[0]**2)*np.eye(2), A, B]

                covariances = [(sd**2)*np.eye(2) for sd in self.toy_sd]
                for k in range(toy_groups):
                    # last axis of input to mvn.pdf denotes the actual components for the PDF
                    density += self.weights[k]*self.mvn.pdf(np.array([x[1], x[0]]), mean=self.means[k].squeeze(),
                                                            cov=covariances[k])
                return density
        elif self.toy_type == 'rings':
            def true_density(x):
                radius = np.sqrt((x[0] ** 2) + (x[1] ** 2))
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k] * self.mvn.pdf(radius, mean=self.toy_radius * (k + 1),
                                                              cov=(self.toy_sd**2))/(2*np.pi*self.toy_radius*(k+1))
                return density
        else:
            raise RuntimeError('Invalid option for toy_type (use "gmm", "gmm_2", or "rings")')
        self.true_density = true_density

        # viz parameters
        self.viz_res = viz_res
        self.kde_bw = kde_bw
        if toy_type == 'rings':
            self.plot_val_max = toy_groups * toy_radius + 4 * toy_sd
        elif toy_type == 'gmm_2':
            self.plot_val_max = max(self.toy_radius) + 4 * max(self.toy_sd)
        else:
            self.plot_val_max = toy_radius + 4 * toy_sd

        # save values for plotting groundtruth landscape
        self.xy_plot = np.linspace(-self.plot_val_max, self.plot_val_max, self.viz_res) # (200)
        self.z_true_density = np.zeros((self.viz_res, self.viz_res)) # (200,200)
        for x_ind in range(len(self.xy_plot)):
            for y_ind in range(len(self.xy_plot)):
                self.z_true_density[x_ind, y_ind] = self.true_density([self.xy_plot[x_ind], self.xy_plot[y_ind]])

    def sample_toy_data(self, num_samples):
        toy_sample = np.zeros(0).reshape(0, 2, 1, 1)
        # Sample from a multinomial across toy_groups according to their Gaussian mixture weights
        sample_group_sz = np.random.multinomial(num_samples, self.weights)
        if self.toy_type == 'gmm':
            for i in range(self.toy_groups):
                sample_group = self.means[i] + self.toy_sd * np.random.randn(2*sample_group_sz[i]).reshape(-1, 2, 1, 1)
                toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
        elif self.toy_type == 'gmm_2':
            # With 3 toy groups and weights=[1/3, 1/3, 1/3], we generate 33 samples from the first, second, and third means, for each toy group
            for i in range(self.toy_groups):
                sample_group = self.means[i] + self.toy_sd[i] * np.random.randn(2*sample_group_sz[i]).reshape(-1, 2, 1, 1)
                toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
        elif self.toy_type == 'rings':
            for i in range(self.toy_groups):
                sample_radii = self.toy_radius*(i+1) + self.toy_sd * np.random.randn(sample_group_sz[i])
                sample_thetas = 2 * np.pi * np.random.random(sample_group_sz[i])
                sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(-1, 1)
                sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(-1, 1)
                sample_group = np.concatenate((sample_x, sample_y), axis=1)
                toy_sample = np.concatenate((toy_sample, sample_group.reshape(-1, 2, 1, 1)), axis=0)
        else:
            raise RuntimeError('Invalid option for toy_type ("gmm", "gmm_2", or "rings")')

        return toy_sample

    def plot_toy_density(self, plot_truth=False, f=None, epsilon=0.0, x_s_t=None, save_path='toy.pdf'):
        num_plots = 0
        if plot_truth:
            num_plots += 1

        # density of learned EBM
        # if f is not None:
        #     num_plots += 1
        #     xy_plot_torch = t.Tensor(self.xy_plot).view(-1, 1, 1, 1).to(next(f.parameters()).device)
        #     # y values for learned energy landscape of descriptor network
        #     z_learned_energy = np.zeros([self.viz_res, self.viz_res])
        #     for i in range(len(self.xy_plot)):
        #         y_vals = float(self.xy_plot[i]) * t.ones_like(xy_plot_torch)
        #         vals = t.cat((xy_plot_torch, y_vals), 1)
        #         z_learned_energy[i] = f(vals).data.cpu().numpy()
        #     # rescale y values to correspond to the groundtruth temperature
        #     if epsilon > 0:
        #         z_learned_energy *= 2 / (epsilon ** 2)

        #     # transform learned energy into learned density
        #     z_learned_density_unnormalized = np.exp(- z_learned_energy)
        #     bin_area = (self.xy_plot[1] - self.xy_plot[0]) ** 2 # e.g. 0.002
        #     z_learned_density = z_learned_density_unnormalized / (bin_area * np.sum(z_learned_density_unnormalized))

        # kernel density estimate of shortrun samples
        if x_s_t is not None:
            num_plots += 1
            # density_estimate: dataset is (2,10000), covariance (2,2), weights (10000,)
            density_estimate = self.gaussian_kde(x_s_t.squeeze().cpu().numpy().transpose(), bw_method=self.kde_bw)
            z_kde_density = np.zeros([self.viz_res, self.viz_res]) # (200,200)
            for i in range(len(self.xy_plot)): # self.xy_plot (200,)
                for j in range(len(self.xy_plot)):
                    z_kde_density[i, j] = density_estimate((self.xy_plot[j], self.xy_plot[i]))

        # plot results
        plot_ind = 0
        fig = plt.figure()

        # true density
        if plot_truth:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('True density')
            plt.imshow(self.z_true_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('True log-density')
            plt.imshow(np.log(self.z_true_density + 1e-10), cmap='viridis')
            plt.axis('off')
        # learned ebm
        # if f is not None:
        #     plot_ind += 1
        #     ax = fig.add_subplot(2, num_plots, plot_ind)
        #     ax.set_title('EBM density')
        #     plt.imshow(z_learned_density, cmap='viridis')
        #     plt.axis('off')
        #     ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
        #     ax.set_title('EBM log-density')
        #     plt.imshow(np.log(z_learned_density + 1e-10), cmap='viridis')
        #     plt.axis('off')
        # shortrun kde
        if x_s_t is not None:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('Short-run KDE')
            plt.imshow(z_kde_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('Short-run log-KDE')
            plt.imshow(np.log(z_kde_density + 1e-10), cmap='viridis')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.close()
