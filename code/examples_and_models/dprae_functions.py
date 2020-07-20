# dprae function files
import numpy as np
import sklearn.mixture
from scipy.stats import multivariate_normal as normal


def gmm_model(mus, ncomps, xs, sigma_diag, weights, dim=2):
    GMM_model = sklearn.mixture.GaussianMixture(n_components=ncomps, covariance_type="diag")
    GMM_model.fit(xs.reshape(-1, dim))
    GMM_model.means_ = mus
    GMM_model.covariances_ = sigma_diag
    GMM_model.weights_ = weights
    return GMM_model
    
def gmm_pdf(xs, GMM_model, dim=2):
    gmm_mus = GMM_model.means_
    ncomps = GMM_model.n_components
    gmm_sigma = GMM_model.covariances_[0]
    gmm_phis = GMM_model.weights_
    

    pdf_values = np.zeros([xs.shape[0], ncomps])
    for i in range(ncomps):
        pdf_values[:, i] = normal.pdf(xs, gmm_mus[i], gmm_sigma*np.eye(2))

    pdf_output = np.dot(pdf_values, gmm_phis)
    return pdf_output


def draw_cp(hs_all, hcurrent, xstar):
    plt.figure(figsize=(5, 5))
    plt.rcParams["font.size"] = 18
    h1 = hs_all[-1]
    h1s = h1(xs).reshape(x1s.shape)

    for it in range(hs_all.shape[0]):  
        #checking if in the first quadrant
        if np.logical_and(xstar[it,0] >=0, xstar[it,1] >=0):
            hcurrent = hs_all[it]    
            h1s = hcurrent(xs).reshape(x1s.shape)                    
            plt.contour(x1s, x2s, h1s, cmap="gray", levels=[-1000, 0, 1000])

    plt.contourf(x1s, x2s, ys,  levels=[-100, 0, 100], cmap="coolwarm")
    #plt.contour(x1s, x2s, ps, cmap="coolwarm", alpha=0.8)
    plt.contour(x1s, x2s, g_s, cmap="gray", alpha=0.8, levels=[-1000, gamma, 1000])
    plt.scatter(xstar[:, 0], xstar[:,1])
    plt.xlim(lb, ub)
    plt.ylim(lb, ub)        
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')    
    plt.show()

def plot_cp_gi(g_i_set):
    plt.figure(figsize=(5, 5))
    plt.rcParams["font.size"] = 18

    plt.contourf(x1s, x2s, ys, levels=[-1, 0, 1], cmap="coolwarm")
    plt.scatter(X_pool[:,0], X_pool[:,1],  s=2, c="C3", label="Sample from p")

    plt.scatter(xstar_ll[:,0], xstar_ll[:,1], c='C0')

    plt.contour(x1s, x2s, g_i_s, cmap="gray", levels=[-1000, 0, 1000])
    
    #iterate on the g_i_set
    for it in range(len(g_i_set)):
        g_i_it = g_i_set[it]
        g_i_s_it = g_i_it(xs).reshape(x1s.shape)
        plt.contour(x1s, x2s, g_i_s_it, cmap="gray", levels=[-1000, 0, 1000])

    plt.scatter(mu_0[0], mu_0[1], c="k", label="$\mu_0$")

    plt.contour(x1s, x2s, ps, cmap="coolwarm")

    plt.xlim(lb, ub)
    plt.ylim(lb, ub)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()    