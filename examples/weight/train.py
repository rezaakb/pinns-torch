from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstorch.utils.load_data(root_path, "truck_data_2380kg_12mps_bumpy.mat")
    x = 0.0
    t = np.linspace(1, 1+0.03*201, 200)[:, None]
    exact_body_x = np.real(data["truck_data"][100:300, 0]).astype("float32")[None, :]            # car z-axis displacement
    exact_body_y = np.real(data["truck_data"][100:300, 1]).astype("float32")[None, :]            # car z-axis displacement
    
    dt = 0.03  # [s] sample period
    exact_body_vx = np.gradient(exact_body_x, dt, axis=1).astype("float32")  # [m/s]
    exact_body_vy = np.gradient(exact_body_y, dt, axis=1).astype("float32")  # [m/s]
    exact_body_z = -np.real(data["truck_data"][100:300, 2]).astype("float32")[None, :]            # car z-axis displacement
    exact_body_roll = np.real(data["truck_data"][100:300, 3]).astype("float32")[None, :]    # car pitch
    exact_body_pitch = -np.real(data["truck_data"][100:300, 4]).astype("float32")[None, :]    # car pitch
    exact_tire_fl_z = np.real(data["truck_data"][100:300, 6]).astype("float32")[None, :]      # front left tire z-axis displacement
    exact_tire_fr_z = np.real(data["truck_data"][100:300, 7]).astype("float32")[None, :]      # front left tire z-axis displacement
    exact_tire_rl_z = np.real(data["truck_data"][100:300, 8]).astype("float32")[None, :]      # rear left tire z-axis displacement
    exact_tire_rr_z = np.real(data["truck_data"][100:300, 9]).astype("float32")[None, :]      # rear left tire z-axis displacement

    return pinnstorch.data.PointCloudData(spatial=[x], time=[t], 
                                          solution={"bz": exact_body_z,
                                                    "br": exact_body_roll,
                                                    "bp": exact_body_pitch,
                                                    "tflz": exact_tire_fl_z,
                                                    "tfrz": exact_tire_fr_z,
                                                    "trlz": exact_tire_rl_z,
                                                    "trrz": exact_tire_rr_z,
                                                    "vx": exact_body_vx,
                                                    "vy": exact_body_vy
                                                    })

def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor,
           extra_variables: Dict[str, torch.Tensor]):
    """
    7-DOF (heave/roll/pitch + 4 unsprung) in z-up.
    Residuals: m*q̈ − ΣF = 0, I*α − ΣM = 0.
    Supports ARB, asymmetric damping, bump/top-out, aero.
    NEW (optional): residual generalized forces/moments to absorb unmodeled dynamics.
    """

    # safe getter: if a key isn't in outputs, return zeros (no effect)
    def get_out(key: str, like: torch.Tensor) -> torch.Tensor:
        v = outputs.get(key, None)
        return v if v is not None else torch.zeros_like(like)

    # states (z-up)
    bz   = outputs["bz"]; br = outputs["br"]; bp = outputs["bp"]
    tflz = outputs["tflz"]; tfrz = outputs["tfrz"]; trlz = outputs["trlz"]; trrz = outputs["trrz"]
    vx = outputs['vx']; vy = outputs['vy']

    # optional road heights (latent)
    y1 = get_out("y1", tflz); y2 = get_out("y2", tfrz); y3 = get_out("y3", trlz); y4 = get_out("y4", trrz)

    # time derivs
    grad = pinnstorch.utils.gradient
    bz_t = grad(bz, t)[0]; br_t = grad(br, t)[0]; bp_t = grad(bp, t)[0]
    tflz_t = grad(tflz, t)[0]; tfrz_t = grad(tfrz, t)[0]; trlz_t = grad(trlz, t)[0]; trrz_t = grad(trrz, t)[0]
    y1_t = grad(y1, t)[0]; y2_t = grad(y2, t)[0]; y3_t = grad(y3, t)[0]; y4_t = grad(y4, t)[0]

    # second derivs
    bz_tt   = grad(bz_t,   t)[0]
    br_tt   = grad(br_t,   t)[0]
    bp_tt   = grad(bp_t,   t)[0]
    tflz_tt = grad(tflz_t, t)[0]
    tfrz_tt = grad(tfrz_t, t)[0]
    trlz_tt = grad(trlz_t, t)[0]
    trrz_tt = grad(trrz_t, t)[0]

    # denorm helper for extra_variables in [-1,1]
    def denorm(p_norm, p_min, p_max):
        return 0.5 * (p_min + p_max) + 0.5 * (p_max - p_min) * p_norm

    g = 9.81

    # properties (sprung/unsprung)
    m   = denorm(extra_variables["m"],   1500.0, 3000.0)
    I_x = denorm(extra_variables["I_x"], 1000.0, 2000.0)
    I_y = denorm(extra_variables["I_y"], 2000.0, 3000.0)
    m_f = denorm(extra_variables["m_f"], 200.0, 300.0)
    m_r = denorm(extra_variables["m_r"], 200.0, 300.0)

    # suspension/tire (linear baseline + asymmetric damping)
    kf = denorm(extra_variables["kf"],  1.0e4, 1.0e5)
    kr = denorm(extra_variables["kr"], 1.0e4, 1.0e5)
    cf_comp = denorm(extra_variables["cf_comp"], 1.0e3, 1.0e4)
    cf_reb  = denorm(extra_variables["cf_reb"],  1.0e3, 1.0e4)
    cr_comp = denorm(extra_variables["cr_comp"], 1.0e3, 1.0e4)
    cr_reb  = denorm(extra_variables["cr_reb"],  1.0e3, 1.0e4)

    ktf = denorm(extra_variables["ktf"], 1.0e5, 5.0e5)
    ktr = denorm(extra_variables["ktr"], 1.0e5, 5.0e5)
    ctf_comp = denorm(extra_variables["ctf_comp"], 1.0e3, 1.0e4)
    ctf_reb  = denorm(extra_variables["ctf_reb"],  1.0e3, 1.0e4)
    ctr_comp = denorm(extra_variables["ctr_comp"], 1.0e3, 1.0e4)
    ctr_reb  = denorm(extra_variables["ctr_reb"],  1.0e3, 1.0e4)

    # ARB + bump/top-out + aero
    kR = denorm(extra_variables["kR"], 0.0, 1.0e5)
    j0_f  = denorm(extra_variables["j0_f"],  0.0, 0.05)
    j0_r  = denorm(extra_variables["j0_r"], 0.0, 0.05)
    top0_f= denorm(extra_variables["top0_f"], 0.0, 0.06)
    top0_r= denorm(extra_variables["top0_r"], 0.0, 0.06)
    kf_bump= denorm(extra_variables["kf_bump"],  0.0, 2.0e5)
    kr_bump= denorm(extra_variables["kr_bump"],  0.0, 2.0e5)
    kf_top = denorm(extra_variables["kf_top"],   0.0, 1.0e5)
    kr_top = denorm(extra_variables["kr_top"],  0.0, 1.0e5)
    v2 = vx**2 + vy**2
    cdown  = denorm(extra_variables["cdown"],  0.0, 5.0e3)
    cpitch = denorm(extra_variables["cpitch"],  -5.0e3, 5.0e3)

    # geometry
    b1 = 0.856
    b2 = 0.856
    a1 = 1.491
    a2 = 1.629

    relu = torch.nn.functional.relu

    # suspension kinematics
    dsusp_fl = (bz - tflz) + br * b1 - bp * a1
    dsusp_fr = (bz - tfrz) - br * b2 - bp * a1
    dsusp_rl = (bz - trlz) + br * b1 + bp * a2
    dsusp_rr = (bz - trrz) - br * b2 + bp * a2

    vsusp_fl = (bz_t - tflz_t) + br_t * b1 - bp_t * a1
    vsusp_fr = (bz_t - tfrz_t) - br_t * b2 - bp_t * a1
    vsusp_rl = (bz_t - trlz_t) + br_t * b1 + bp_t * a2
    vsusp_rr = (bz_t - trrz_t) - br_t * b2 + bp_t * a2

    # asymmetric damping bits
    vpos_fl, vneg_fl = relu(vsusp_fl), relu(-vsusp_fl)
    vpos_fr, vneg_fr = relu(vsusp_fr), relu(-vsusp_fr)
    vpos_rl, vneg_rl = relu(vsusp_rl), relu(-vsusp_rl)
    vpos_rr, vneg_rr = relu(vsusp_rr), relu(-vsusp_rr)

    # bump/top-out
    comp_ex_fl = relu(-dsusp_fl - j0_f); comp_ex_fr = relu(-dsusp_fr - j0_f)
    comp_ex_rl = relu(-dsusp_rl - j0_r); comp_ex_rr = relu(-dsusp_rr - j0_r)
    top_ex_fl  = relu(dsusp_fl - top0_f); top_ex_fr  = relu(dsusp_fr - top0_f)
    top_ex_rl  = relu(dsusp_rl - top0_r); top_ex_rr  = relu(dsusp_rr - top0_r)

    # forces on BODY (up +)
    Fs_fl = (-kf * dsusp_fl) + (cf_comp * vneg_fl - cf_reb * vpos_fl) + (kf_bump * comp_ex_fl) - (kf_top * top_ex_fl)
    Fs_fr = (-kf * dsusp_fr) + (cf_comp * vneg_fr - cf_reb * vpos_fr) + (kf_bump * comp_ex_fr) - (kf_top * top_ex_fr)
    Fs_rl = (-kr * dsusp_rl) + (cr_comp * vneg_rl - cr_reb * vpos_rl) + (kr_bump * comp_ex_rl) - (kr_top * top_ex_rl)
    Fs_rr = (-kr * dsusp_rr) + (cr_comp * vneg_rr - cr_reb * vpos_rr) + (kr_bump * comp_ex_rr) - (kr_top * top_ex_rr)

    # tires on UNSPRUNG (up + on wheel)
    vt_fl = tflz_t - y1_t; vt_fr = tfrz_t - y2_t; vt_rl = trlz_t - y3_t; vt_rr = trrz_t - y4_t
    vtpos_fl, vtneg_fl = relu(vt_fl), relu(-vt_fl)
    vtpos_fr, vtneg_fr = relu(vt_fr), relu(-vt_fr)
    vtpos_rl, vtneg_rl = relu(vt_rl), relu(-vt_rl)
    vtpos_rr, vtneg_rr = relu(vt_rr), relu(-vt_rr)

    Ft_fl = (-ktf * (tflz - y1)) + (ctf_comp * vtneg_fl - ctf_reb * vtpos_fl)
    Ft_fr = (-ktf * (tfrz - y2)) + (ctf_comp * vtneg_fr - ctf_reb * vtpos_fr)
    Ft_rl = (-ktr * (trlz - y3)) + (ctr_comp * vtneg_rl - ctr_reb * vtpos_rl)
    Ft_rr = (-ktr * (trrz - y4)) + (ctr_comp * vtneg_rr - ctr_reb * vtpos_rr)

    # ARB + aero
    b_avg = 0.5 * (b1 + b2)
    M_arb = kR * ((dsusp_fr - dsusp_fl) + (dsusp_rr - dsusp_rl)) * b_avg
    F_aero = -cdown * v2
    M_aero =  cpitch * v2

    # ---------- OPTIONAL residual generalized forces (default 0) ----------
    rFz   = get_out("rFz", bz);  rMx   = get_out("rMx", bz);  rMy   = get_out("rMy", bz)
    rFs_fl = get_out("rFs_fl", bz); rFs_fr = get_out("rFs_fr", bz); rFs_rl = get_out("rFs_rl", bz); rFs_rr = get_out("rFs_rr", bz)
    rFt_fl = get_out("rFt_fl", bz); rFt_fr = get_out("rFt_fr", bz); rFt_rl = get_out("rFt_rl", bz); rFt_rr = get_out("rFt_rr", bz)

    # residuals (body)
    sum_Fs = Fs_fl + Fs_fr + Fs_rl + Fs_rr
    sum_Fs_aug = sum_Fs + rFz + (rFs_fl + rFs_fr + rFs_rl + rFs_rr)
    outputs["res_bz"] = m * bz_tt - (sum_Fs_aug + F_aero - m * g)

    M_roll = (b1 * Fs_fl) - (b2 * Fs_fr) + (b1 * Fs_rl) - (b2 * Fs_rr) + M_arb
    M_roll = M_roll + rMx + (b1*rFs_fl - b2*rFs_fr + b1*rFs_rl - b2*rFs_rr)
    outputs["res_br"] = I_x * br_tt - M_roll

    M_pitch = (-a1 * Fs_fl) + (-a1 * Fs_fr) + (a2 * Fs_rl) + (a2 * Fs_rr) + M_aero
    M_pitch = M_pitch + rMy + (-a1*rFs_fl - a1*rFs_fr + a2*rFs_rl + a2*rFs_rr)
    outputs["res_bp"] = I_y * bp_tt - M_pitch

    # residuals (unsprung)
    outputs["res_tflz"] = m_f * tflz_tt - ((-Fs_fl) + (Ft_fl + rFt_fl) - m_f * g)
    outputs["res_tfrz"] = m_f * tfrz_tt - ((-Fs_fr) + (Ft_fr + rFt_fr) - m_f * g)
    outputs["res_trlz"] = m_r * trlz_tt - ((-Fs_rl) + (Ft_rl + rFt_rl) - m_r * g)
    outputs["res_trrz"] = m_r * trrz_tt - ((-Fs_rr) + (Ft_rr + rFt_rr) - m_r * g)

    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstorch.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstorch.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
