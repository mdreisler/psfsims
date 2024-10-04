import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import imageio
import os
import json
import tifffile as tif
from skimage import io
from subpix import subpix_signals
from tqdm import tqdm
import pandas as pd


def measure_SNR_subpix(image4d, pos, frame):
    ss = image4d[
        pos[:, 0, frame].astype(np.int64),
        pos[:, 1, frame].astype(np.int64),
        pos[:, 2, frame].astype(np.int64),
        frame,
    ]
    "bs is image4d where not pos"
    bs = np.delete(image4d, pos.astype(np.int64), axis=0)
    snr = np.round((np.mean(ss) - np.mean(bs)) / np.std(bs), 2)
    return np.array(snr, dtype=np.float16)


def create_trajectory(
    mean_trans,
    sigma_trans,
    fov_dimx,
    fov_dimy,
    fov_dimz,
    steps,
    confine_to_z_plane=None,
):
    phi = np.random.uniform(-np.pi, 2 * np.pi, steps)
    theta = np.random.uniform(-np.pi, np.pi, steps)
    trans = np.random.normal(mean_trans, sigma_trans, steps)
    xs = np.cumsum(np.cos(phi) * np.sin(theta) * trans) + np.random.uniform(
        0.1 * fov_dimx, 0.9 * fov_dimx
    )
    ys = np.cumsum(np.sin(phi) * np.sin(theta) * trans) + np.random.uniform(
        0.1 * fov_dimy, 0.9 * fov_dimy
    )
    zs = np.cumsum(np.cos(theta) * trans) + np.random.uniform(
        0.3 * fov_dimz, 0.7 * fov_dimz
    )
    if confine_to_z_plane is not None:
        zs = np.ones_like(zs) * confine_to_z_plane
    return xs, ys, zs


def create_background(fov_dimx, fov_dimy, fov_dimz, offset, sigma_noise, steps):
    return (
        np.random.poisson(
            sigma_noise * sigma_noise, [fov_dimx, fov_dimy, fov_dimz, steps]
        )
        + offset
    )


def non_diffraction_limited_signals(pos, fluorophore_params):
    """pos is a 3d array of shape (n_p, 3) - takes in one frame of positions"""
    new_pos = np.array([])
    n_non_d = int(fluorophore_params["n"])
    intensity = fluorophore_params["intensity"]
    radius = np.array(fluorophore_params["radius"])
    density = fluorophore_params["density"]
    if radius.shape[0] == 1:
        radius = np.ones(n_non_d) * radius
    elif radius.shape[0] != n_non_d:
        raise ValueError("radius must be scalar or same shape as n")
    for rad, pos_i in zip(radius, pos[:n_non_d]):
        r_lim = rad
        n_fluorophores = int(density * 4 * np.pi * r_lim**2)
        print(n_fluorophores)
        x = pos_i[0]
        y = pos_i[1]
        z = pos_i[2]
        dx = np.random.uniform(-r_lim, r_lim, 3 * n_fluorophores)
        dy = np.random.uniform(-r_lim, r_lim, 3 * n_fluorophores)
        dz = np.random.uniform(-r_lim, r_lim, 3 * n_fluorophores)

        dr = np.sqrt(dx**2 + dy**2 + dz**2)

        "remove fluorophores outside of r_lim"
        dx = dx[dr < r_lim]
        dy = dy[dr < r_lim]
        dz = dz[dr < r_lim]
        "add x, y, z to dx, dy, dz"
        x_fluorophores = dx[:n_fluorophores] + x
        y_fluorophores = dy[:n_fluorophores] + y
        z_fluorophores = dz[:n_fluorophores] + z
        print(x_fluorophores.shape, "x_fluorophores.shape")
        fluorophore_pos = np.array([x_fluorophores, y_fluorophores, z_fluorophores]).T
        print(fluorophore_pos.shape, "fluorophore_pos.shape")
        "stack fluorophore_pos on top of new_pos"
        if new_pos.size == 0:
            new_pos = fluorophore_pos
        else:
            new_pos = np.concatenate((new_pos, fluorophore_pos))
    print("new_pos shape: ", new_pos.shape)
    return new_pos


def gen_particle_sim(
    n_p,
    mean_trans,
    sigma_trans,
    signal_strength,
    signal_strength_std,
    offset,
    sigma_noise,
    sigma_psf,
    fov_dimx,
    fov_dimy,
    fov_dimz,
    n_steps,
    existing_img=None,
    approximate_background=False,
    pulsate=0,
    confine_to_z_plane=None,
    non_d_fluorophore_params=None,
):
    df = pd.DataFrame(
        columns=["x", "y", "z", "frame", "particle", "signal_strength", "snr", "radius"]
    )
    pos = np.zeros((n_p, 3, n_steps))
    for i in tqdm(range(n_p)):
        xs, ys, zs = create_trajectory(
            mean_trans=mean_trans,
            sigma_trans=sigma_trans,
            fov_dimx=fov_dimx,
            fov_dimy=fov_dimy,
            fov_dimz=fov_dimz,
            steps=n_steps,
            confine_to_z_plane=confine_to_z_plane,
        )
        pos[i, 0, :] = xs
        pos[i, 1, :] = ys
        pos[i, 2, :] = zs

    pos = pos.astype(np.float64)

    if existing_img is None:
        background = create_background(
            fov_dimx=fov_dimx,
            fov_dimy=fov_dimy,
            fov_dimz=fov_dimz,
            offset=offset,
            sigma_noise=sigma_noise,
            steps=n_steps,
        )
    else:
        background = existing_img

    if approximate_background is True and existing_img is not None:
        bckdist = np.percentile(existing_img, 99.98)
        signal_std = np.std(existing_img[existing_img > bckdist])
        signal_mean = np.mean(existing_img[existing_img > bckdist])
        existing_img[existing_img > bckdist] = np.mean(existing_img)
        background = existing_img.transpose(1, 2, 0)
        background_shuffled = np.copy(background).flatten()
        idx = np.random.permutation(background_shuffled.shape[0])
        background_shuffled = background_shuffled[idx]
        background_shuffled = background_shuffled.reshape(background.shape)
        background = background_shuffled
        print("std of signals", signal_std, "mean of signals", signal_mean)
        existing_img = np.zeros_like(existing_img)

    sigmas = sigma_psf

    ss_list = np.random.normal(
        signal_strength, signal_strength_std * signal_strength, n_p
    )

    image4d = np.zeros((fov_dimx, fov_dimy, fov_dimz, n_steps))
    for frame in tqdm(range(n_steps)):
        if pulsate > 0:
            ss_list = ss_list * np.random.uniform(1 - pulsate, 1 + pulsate, n_p)
        if non_d_fluorophore_params["n"] > 0:
            ss_list[: int(non_d_fluorophore_params["n"])] = 1
        pic = subpix_signals(
            pos[:, :, frame],
            sigmas,
            picture=background[:, :, :, frame].astype(np.float64),
            intensities=ss_list,
            normalize=False,
        )

        df_temp = pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "frame",
                "particle",
                "signal_strength",
                "snr",
                "radius",
            ]
        )
        df_temp.loc[:, "y"] = pos[:, 0, frame]
        df_temp.loc[:, "x"] = pos[:, 1, frame]
        df_temp.loc[:, "z"] = pos[:, 2, frame]
        df_temp.loc[:, "frame"] = frame
        df_temp.loc[:, "signal_strength"] = ss_list
        df_temp.loc[:, "snr"] = ss_list / sigma_noise
        df_temp.loc[:, "particle"] = np.arange(n_p)
        radii = np.array(non_d_fluorophore_params["radius"])
        n_non_d = int(non_d_fluorophore_params["n"])
        radius = radii
        if radius.shape[0] == 1:
            radius = np.ones(n_non_d) * radius
        elif radius.shape[0] != n_non_d:
            raise ValueError("radius must be scalar or same shape as n")
        radius.resize(n_p)
        radii = radius
        radii = radii.astype(np.float64)
        radii[radii == 0] = sigmas[0] * 2
        df_temp.loc[:, "radius"] = radii
        df = df.append(df_temp)
        if non_d_fluorophore_params["n"] > 0:
            non_d_fluorophore_params = non_d_fluorophore_params.copy()

            if non_d_fluorophore_params["intensity"] is None:
                non_d_fluorophore_params["intensity"] = signal_strength
            pos_ex = non_diffraction_limited_signals(
                pos[:, :, frame], non_d_fluorophore_params
            )
            print(pos_ex.shape)
            intensities = non_d_fluorophore_params["intensity"] * np.ones(
                pos_ex.shape[0]
            )
            print(intensities.shape)
            pic = subpix_signals(
                pos_ex,
                sigmas,
                picture=pic.astype(np.float64),
                intensities=intensities,
                normalize=False,
            )
            pos_ex = None
        image4d[:, :, :, frame] = pic
    snr_sim = [0, 0]
    return image4d, pos, background, ss_list, np.mean(snr_sim), df


if __name__ == "__main__":
    n_p = 80
    mean_trans = 4
    sigma_trans = 3
    signal_strength = 3500
    signal_strength_std = 0.1
    offset = 1000
    sigma_noise = 40
    sigma_psf = [1.2, 1.2, 2.4]
    fov_dimx = 512
    fov_dimy = 512
    fov_dimz = 20
    n_steps = 25
    existing_img = None
    approximate_background = False
    pulsate = 0
    confine_to_z_plane = None

    non_d_fluro_params = {
        "n": 12,
        "intensity": 500,
        "radius": [5],
        "density": 0.5,
    }

    # endosome_like_params = {
    #    "n": 15,
    #    "intensity": 1000,
    #    "r_range": [2.5, 2.5],
    #    "density": 0.5,
    # }


    image4d, pos, background, ss_list, snr_sim, df = gen_particle_sim(
        n_p=n_p,
        mean_trans=mean_trans,
        sigma_trans=sigma_trans,
        signal_strength=signal_strength,
        signal_strength_std=signal_strength_std,
        offset=offset,
        sigma_noise=sigma_noise,
        sigma_psf=sigma_psf,
        fov_dimx=fov_dimx,
        fov_dimy=fov_dimy,
        fov_dimz=fov_dimz,
        n_steps=n_steps,
        existing_img=existing_img,
        approximate_background=approximate_background,
        pulsate=pulsate,
        confine_to_z_plane=confine_to_z_plane,
        non_d_fluorophore_params=non_d_fluro_params,
    )

    