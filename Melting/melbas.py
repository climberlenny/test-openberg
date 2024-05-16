import numpy as np
import matplotlib.pyplot as plt


def melbas(depthib, lib, salnib, tempib, uoib, voib, uib, vib, dt):
    """
    Calculate the surface melt due to forced convection following Kubat et al.
    (An operational iceberg deterioration Model 2007).

    :param depthib: Iceberg depth (positive value)
    :param lib: Iceberg length scale
    :param salnib: Salinity profile in the ocean (array of length nz)
    :param tempib: Temperature profile in the ocean (array of length nz)
    :param dp: Thickness of each ocean layer (array of length nz)
    :param uoib: Ocean velocity components in the x-direction (array of length nz)
    :param voib: Ocean velocity components in the y-direction (array of length nz)
    :param uib: Iceberg drift velocity in the x-direction
    :param vib: Iceberg drift velocity in the y-direction
    :param nz: Number of layers in the ocean

    The function updates the depthib parameter.
    """

    # Temperature at the base of the iceberg
    absv = ((uoib[-1] - uib) ** 2 + (voib[-1] - vib) ** 2) ** 0.5
    TfS = -0.036 - 0.0499 * salnib[-1] - 0.000112 * salnib[-1] ** 2
    Tfp = TfS * 2.71828 ** (-0.19 * (tempib[-1] - TfS))
    deltat = tempib[-1] - Tfp

    Vf = 0.58 * absv**0.8 * deltat / (lib**0.2)
    Vf = Vf / 86400  # convert in m/s

    # Update the depth
    depthib = abs(depthib) - Vf * dt
    return max(0, depthib)


def test_melbas(days=30):
    nz = 5
    depthib = 100
    lib = 100  # Replace with your initial value for lib
    uoib = np.ones(nz) * 0.1  # Replace with your value for uoib
    voib = np.ones(nz) * 0.05  # Replace with your value for voib
    uib = 0.2
    vib = 0.05
    salnib = np.ones(nz) * 35
    tempib = np.linspace(-1, 3, nz)

    drift_period = np.arange(0, 86400 * days, 600)
    DEPTHIB = [depthib]
    for t1, t2 in zip(drift_period[:-1], drift_period[1:]):
        dt = t2 - t1
        depthib = melbas(depthib, lib, salnib, tempib, uoib, voib, uib, vib, dt)
        DEPTHIB.append(depthib)
    print(
        f"After {days} days the iceberg length has shrinked by {np.round((DEPTHIB[0]-DEPTHIB[-1])/DEPTHIB[0]*100,1)}% "
    )
    fig, ax = plt.subplots()
    ax.plot(drift_period / 86400, DEPTHIB, label="iceberg length (m)")
    ax.set_title("Iceberg decay according to basal melting")
    ax.set_xlabel("day")
    plt.legend()
    plt.show()


def test_parm_melbas():
    # Default values
    nz = 10
    depthib_def = 100
    tempib_def = np.linspace(-1, 3, nz)
    salnib_def = np.ones(nz) * 35
    lib_def = 100
    uoib_def = np.ones(nz) * 0.1
    voib_def = np.ones(nz) * 0.05
    uib_def = 0.15
    vib_def = 0.05

    lib = np.linspace(100, 1000, 100)
    depthib = np.linspace(100, 1000, 100)
    tempib = [np.ones(nz) * T for T in np.linspace(-2, 5, 100)]
    salnib = [np.ones(nz) * S for S in np.linspace(10, 100, 100)]
    uib = np.linspace(0, 5, 100)
    vib = vib_def

    dt = 30 * 86400  # 1 month
    decay_percentage_lib = []
    decay_percentage_depthib = []
    decay_percentage_tempib = []
    decay_percentage_salnib = []
    decay_percentage_speed = []
    for i in range(len(lib)):
        ndp = melbas(
            depthib_def,
            lib[i],
            salnib_def,
            tempib_def,
            uoib_def,
            voib_def,
            uib_def,
            vib_def,
            dt,
        )
        decay_percentage_lib.append((depthib_def - ndp) / depthib_def * 100)

        ndp = melbas(
            depthib[i],
            lib_def,
            salnib_def,
            tempib_def,
            uoib_def,
            voib_def,
            uib_def,
            vib_def,
            dt,
        )
        decay_percentage_depthib.append((depthib[i] - ndp) / depthib[i] * 100)

        ndp = melbas(
            depthib_def,
            lib_def,
            salnib_def,
            tempib[i],
            uoib_def,
            voib_def,
            uib_def,
            vib_def,
            dt,
        )
        decay_percentage_tempib.append((depthib_def - ndp) / depthib_def * 100)

        ndp = melbas(
            depthib_def,
            lib_def,
            salnib[i],
            tempib_def,
            uoib_def,
            voib_def,
            uib_def,
            vib_def,
            dt,
        )
        decay_percentage_salnib.append((depthib_def - ndp) / depthib_def * 100)

        ndp = melbas(
            depthib_def,
            lib_def,
            salnib_def,
            tempib_def,
            uoib_def,
            voib_def,
            uib[i],
            vib_def,
            dt,
        )
        decay_percentage_speed.append((depthib_def - ndp) / depthib_def * 100)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Iceberg decay sensitivity from the parameters after 30 days according to basal melting"
    )
    axs[0, 0].plot(lib, decay_percentage_lib)
    axs[0, 0].set_xlabel("initial iceberg length (m)")
    axs[0, 1].plot(np.mean(tempib, axis=1), decay_percentage_tempib)
    axs[0, 1].set_xlabel("Average Sea water far field temperature (deg C)")
    axs[0, 2].plot(np.mean(salnib, axis=1), decay_percentage_salnib)
    axs[0, 2].set_xlabel("Average Sea water salinity")
    axs[0, 0].set_ylabel("decay percentage")

    axs[1, 0].plot(lib, decay_percentage_depthib)
    axs[1, 0].set_xlabel("initial iceberg depth (m)")
    axs[1, 1].plot(uib, decay_percentage_speed)
    axs[1, 1].set_xlabel("Iceberg velocity")
    axs[1, 0].set_ylabel("decay percentage")
    # plt.legend()
    plt.show()


test_melbas()
test_parm_melbas()
