import os
import logging
from itertools import combinations, product

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

log = logging.getLogger(__name__)


def figsize(scale, nplots=1):
    """Calculate the figure size based on a given scale and number of plots.

    :param scale: Scaling factor for the figure size.
    :param nplots: Number of subplots in the figure (default is 1).
    :return: Calculated figure size in inches.
    """

    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt


def newfig(width, nplots=1):
    """Create a new figure with a specified width and number of subplots.

    :param width: Width of the figure.
    :param nplots: Number of subplots in the figure (default is 1).
    :return: Created figure and subplot axis.
    """

    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    """Save a figure to the specified filename with optional cropping.

    :param filename: Name of the output file (without extension).
    :param crop: Whether to apply tight cropping to the saved image (default is True).
    """
    
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    

    if crop:
        plt.savefig(f"{filename}.pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(f"{filename}.eps", bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(f"{filename}.pdf")
        plt.savefig(f"{filename}.eps")

    log.info(f"Image saved at {filename}")


def plot_navier_stokes(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot Navier-Stokes continuous inverse PDE."""

    x, t, u = train_datasets[0][:]
    p_star = mesh.solution["p"][:, 100]
    p_pred = preds["p"].reshape(p_star.shape)
    X_star = np.hstack(mesh.spatial_domain)
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]

    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method="cubic")
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method="cubic")

    fig, ax = newfig(1.015, 0.8)
    ax.axis("off")
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1 - 1 / 2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(
        PP_star,
        interpolation="nearest",
        cmap="rainbow",
        extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", "box")
    ax.set_title("Predicted pressure", fontsize=10)

    # Exact p(t,x,y)
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(
        P_exact,
        interpolation="nearest",
        cmap="rainbow",
        extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", "box")
    ax.set_title("Exact pressure", fontsize=10)
    savefig(file_name + "/fig")


def plot_kdv(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot KdV discrete inverse PDE."""

    fig, ax = newfig(1.0, 1.2)
    ax.axis("off")

    x0 = train_datasets[0].spatial_domain_sampled[0].detach().numpy()
    u0 = train_datasets[0].solution_sampled[0].detach().numpy()
    idx_t0 = train_datasets[0].idx_t
    x1 = train_datasets[1].spatial_domain_sampled[0].detach().numpy()
    u1 = train_datasets[1].solution_sampled[0].detach().numpy()
    idx_t1 = train_datasets[1].idx_t
    exact_u = mesh.solution["u"]

    # Row 0: h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="rainbow",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(mesh.spatial_domain[:].min(), mesh.spatial_domain[:].max(), 2)[:, None]
    ax.plot(mesh.time_domain[idx_t0] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(mesh.time_domain[idx_t1] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # Row 1: h(t,x) slices
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 1 / 2 - 0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t0].T, "b-", linewidth=2)
    ax.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t0]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t1], "b-", linewidth=2)
    ax.plot(x1, u1, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t1]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])

    savefig(file_name + "/fig")


def plot_ac(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot Allen-Cahn discrete forward PDE."""

    fig, ax = newfig(1.0, 1.2)

    x0 = train_datasets[0].spatial_domain_sampled[0].detach().numpy()
    u0 = train_datasets[0].solution_sampled[0].detach().numpy()
    exact_u = mesh.solution["u"]
    idx_t0 = train_datasets[0].idx_t
    idx_t1 = val_dataset.idx_t
    U1_pred = preds["u"]

    ax.axis("off")

    # Row 0: h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="seismic",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(mesh.spatial_domain[:].min(), mesh.spatial_domain[:].max(), 2)[:, None]
    ax.plot(mesh.time_domain[idx_t0] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(mesh.time_domain[idx_t1] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # Row 1: h(t,x) slices
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 1 / 2 - 0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t0], "b-", linewidth=2)
    ax.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t0]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t1], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain[:], U1_pred[:, -1], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t1]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])

    ax.legend(loc="upper center", bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

    savefig(file_name + "/fig")


def plot_burgers_discrete_forward(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot burgers discrete forward PDE."""

    fig, ax = newfig(1.0, 1.2)

    x0 = train_datasets[0].spatial_domain_sampled[0].detach().numpy()
    u0 = train_datasets[0].solution_sampled[0].detach().numpy()
    exact_u = mesh.solution["u"]
    idx_t0 = train_datasets[0].idx_t
    idx_t1 = val_dataset.idx_t
    U1_pred = preds["u"]

    ax.axis("off")

    # Row 0: h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="seismic",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(mesh.spatial_domain[:].min(), mesh.spatial_domain[:].max(), 2)[:, None]
    ax.plot(mesh.time_domain[idx_t0] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(mesh.time_domain[idx_t1] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # Row 1: h(t,x) slices
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 1 / 2 - 0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t0], "b-", linewidth=2)
    ax.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t0]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t1], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain[:], U1_pred[:, -1], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t1]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])

    ax.legend(loc="upper center", bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

    savefig(file_name + "/fig")


def plot_burgers_discrete_inverse(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot burgers continuous forward PDE."""

    fig, ax = newfig(1.0, 1.2)

    x0 = train_datasets[0].spatial_domain_sampled[0].detach().numpy()
    u0 = train_datasets[0].solution_sampled[0].detach().numpy()
    x1 = train_datasets[1].spatial_domain_sampled[0].detach().numpy()
    u1 = train_datasets[1].solution_sampled[0].detach().numpy()
    exact_u = mesh.solution["u"]
    idx_t0 = train_datasets[0].idx_t
    idx_t1 = val_dataset.idx_t

    ax.axis("off")

    fig, ax = newfig(1.0, 1.5)
    ax.axis("off")

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3 + 0.05, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="rainbow",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(mesh.spatial_domain[:].min(), mesh.spatial_domain[:].max(), 2)[:, None]
    ax.plot(mesh.time_domain[idx_t0] * np.ones((2, 1)), line, "w-", linewidth=1.0)
    ax.plot(mesh.time_domain[idx_t1] * np.ones((2, 1)), line, "w-", linewidth=1.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("$u(t,x)$", fontsize=10)

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 1 / 3 - 0.1, bottom=1 - 2 / 3, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t0][:, None], "b", linewidth=2, label="Exact")
    ax.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title(
        "$t = %.2f$\n%d training data" % (mesh.time_domain[idx_t0], u0.shape[0]), fontsize=10
    )

    ax = plt.subplot(gs1[0, 1])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t1][:, None], "b", linewidth=2, label="Exact")
    ax.plot(x1, u1, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title(
        "$t = %.2f$\n%d training data" % (mesh.time_domain[idx_t1], u1.shape[0]), fontsize=10
    )
    ax.legend(loc="upper center", bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

    savefig(file_name + "/fig")


def plot_schrodinger(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot schrodinger continuous forward PDE."""

    h_pred = preds["h"]
    Exact_h = mesh.solution["h"]
    H_pred = h_pred.reshape(Exact_h.shape)

    # Row 1: u(t,x) slices
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    x0, t0, u0 = train_datasets[1][:]
    x_b, t_b, _ = train_datasets[2][:]
    mid = t_b.size()[0] // 2

    X0 = np.hstack((x0[0], t0))
    X_ub = np.hstack((x_b[0][:mid], t_b[:mid]))
    X_lb = np.hstack((x_b[0][mid:], t_b[mid:]))
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis("off")

    # Row 0: h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        H_pred,
        interpolation="nearest",
        cmap="YlGnBu",
        extent=[mesh.lb[1], mesh.ub[1], mesh.lb[0], mesh.ub[0]],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )
    line = np.linspace(mesh.spatial_domain_mesh[:].min(), mesh.spatial_domain_mesh[:].max(), 2)[:, None]

    ax.plot(mesh.time_domain[75] * np.ones((2, 1)), line, "k--", linewidth=1)
    ax.plot(mesh.time_domain[100] * np.ones((2, 1)), line, "k--", linewidth=1)
    ax.plot(mesh.time_domain[125] * np.ones((2, 1)), line, "k--", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$|h(t,x)|$", fontsize=10)

    # Row 1: h(t,x) slices
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain_mesh[:, 75, 0], Exact_h[:, 75], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain_mesh[:, 75, 0], H_pred[:, 75], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[75]), fontsize=10)
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(mesh.spatial_domain_mesh[:, 100, 0], Exact_h[:, 100], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain_mesh[:, 100, 0], H_pred[:, 100], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title("$t = %.2f$" % (mesh.time_domain[100]), fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(mesh.spatial_domain_mesh[:, 125, 0], Exact_h[:, 125], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain_mesh[:, 125, 0], H_pred[:, 125], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title("$t = %.2f$" % (mesh.time_domain[125]), fontsize=10)

    savefig(file_name + "/fig")


def plot_burgers_continuous_forward(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot burgers continuous forward PDE."""

    U_pred = preds["u"]
    exact_u = mesh.solution["u"]
    x = mesh.spatial_domain[:]
    x_i, t_i, _ = train_datasets[1][:]
    x_b, t_b, _ = train_datasets[2][:]

    U_pred = U_pred.reshape(exact_u.shape)
    X_u_train = np.vstack([x_i[0], x_b[0]])

    X_u_time = np.vstack([t_i, t_b])

    X_u_train = np.hstack([X_u_train, X_u_time])
    fig, ax = newfig(1.5, 0.9)
    ax.axis("off")

    # Row 0: u(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        U_pred,
        interpolation="nearest",
        cmap="rainbow",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )

    line = np.linspace(mesh.spatial_domain[:].min(), mesh.spatial_domain[:].max(), 2)[:, None]
    ax.plot(mesh.time_domain[25] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(mesh.time_domain[50] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(mesh.time_domain[75] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # Row 1: u(t,x) slices
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, exact_u[:, 25], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[:, 25], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = 0.25$", fontsize=10)
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, exact_u[:, 50], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[:, 50], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.50$", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, exact_u[:, 75], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[:, 75], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.75$", fontsize=10)

    savefig(file_name + "/fig")


def plot_burgers_continuous_inverse(mesh, preds, train_datasets, val_dataset, file_name):
    """Plot burgers continuous inverse PDE."""

    U_pred = preds["u"]

    exact_u = mesh.solution["u"]
    U_pred = U_pred.reshape(exact_u.shape)

    x_i, t_i, _ = train_datasets[0][:]

    X_u_train = np.hstack([x_i[0], t_i])

    fig, ax = newfig(1.0, 0.9)
    ax.axis("off")

    # Row 0: u(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        U_pred,
        interpolation="nearest",
        cmap="rainbow",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )

    line = np.linspace(mesh.spatial_domain[:].min(), mesh.spatial_domain[:].max(), 2)[:, None]
    ax.plot(mesh.time_domain[25] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(mesh.time_domain[50] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(mesh.time_domain[75] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # Row 1: u(t,x) slices
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain[:], exact_u[:, 25], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain[:], U_pred[:, 25], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = 0.25$", fontsize=10)
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(mesh.spatial_domain[:], exact_u[:, 50], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain[:], U_pred[:, 50], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.50$", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(mesh.spatial_domain[:], exact_u[:, 75], "b-", linewidth=2, label="Exact")
    ax.plot(mesh.spatial_domain[:], U_pred[:, 75], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.75$", fontsize=10)

    savefig(file_name + "/fig")
