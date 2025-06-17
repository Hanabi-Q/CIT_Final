import numpy as np
import astra
import matplotlib.pyplot as plt
import os


def generate_full_3d_sinogram(
        dense_path="phantom_dense.npy",
        n_proj=180,
        add_noise=False,
        noise_std=0.05,
        save_prefix="full3D_sino"
):

    dense = np.load(dense_path)
    print("Loaded dense shape:", dense.shape)

    dense = np.transpose(dense, (2, 0, 1))
    nz, ny, nx = dense.shape
    print("Transposed volume shape (z,y,x):", (nz, ny, nx))

    angles = np.linspace(0, np.pi, n_proj, endpoint=False).astype(np.float32)

    sino_list = []
    for i in range(nz):
        slice_2d = dense[i].astype(np.float32)

        proj_geom = astra.create_proj_geom('parallel', 1.0, nx, angles)
        vol_geom = astra.create_vol_geom(nx, ny)

        sino_id = astra.data2d.create('-sino', proj_geom)
        vol_id = astra.data2d.create('-vol', vol_geom, slice_2d)
        proj_id = astra.create_projector('linear', proj_geom, vol_geom)

        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        sino_data = astra.data2d.get(sino_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete([sino_id, vol_id])
        astra.projector.delete(proj_id)

        sino_list.append(sino_data)

        if (i + 1) % 50 == 0 or i == nz - 1:
            print(f"Processed slice {i + 1}/{nz}")

    full_sino = np.stack(sino_list, axis=-1)
    print("Full 3D sinogram shape:", full_sino.shape)

    if add_noise:
        print("Adding Gaussian noise with std =", noise_std)
        full_sino += np.random.normal(0, noise_std, size=full_sino.shape).astype(np.float32)

    os.makedirs("output_3d", exist_ok=True)
    out_npy = f"output_3d/{save_prefix}_{n_proj}.npy"
    np.save(out_npy, full_sino)
    print("Full 3D sinogram saved to", out_npy)

    mid_proj = n_proj // 2
    plt.figure(figsize=(8, 6))
    plt.imshow(full_sino[mid_proj, :, :], cmap="gray", aspect="auto")
    plt.title(f"Full 3D Sinogram at projection {mid_proj}")
    plt.xlabel("Slice index (z)")
    plt.ylabel("Detector pixel")
    plt.colorbar()
    plt.tight_layout()
    out_png = f"output_3d/{save_prefix}_{n_proj}_vis.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Visualization saved to", out_png)


if __name__ == "__main__":
    generate_full_3d_sinogram(
        dense_path="phantom_dense.npy",
        n_proj=180,
        add_noise=True,
        noise_std=0.05,
        save_prefix="full3D_sino"
    )
