import numpy as np
import matplotlib.pyplot as plt


def mellat(lib, wib, tempib, salnib, nz, dt):
    # Lateral melting parameterization taken from Kubat et al. 2007
    # An operational iceberg deterioration model
    """_summary_

    Args:
        lib (_type_): iceberg length
        wib (_type_): icebrg width
        tempib (_type_): far field water temperature vector size nz
        salnib (_type_): water salinity vector size nz
        nz (_type_): number of ocean layer to use
        dt : timestep in second
    """

    if lib == 0:
        return 0.0, 0.0

    TfS = -0.036 - 0.0499 * salnib - 0.000112 * salnib**2
    Tfp = TfS * np.exp(-0.19 * (tempib - TfS))
    deltaT = tempib - Tfp
    deltaT = np.concatenate([deltaT, deltaT**2])
    coefs = np.concatenate([np.ones(nz) * 2.78, np.ones(nz) * 0.47])
    sumVb = np.dot(deltaT, coefs)  # Maybe cut the speed between -2 and 0 degC

    # Unit of sumVb [meter/year]
    # Convert to meter per second
    dx = sumVb / 365 / 86400 * dt

    # Change of iceberg length (on both sides?? -> 2.*)
    new_lib = max(0, lib - 2 * dx)
    new_wib = max(0, wib / lib * new_lib)  # keep the same width/length ratio
    return new_lib, new_wib


def test_mellat(days=30):
    nz = 5
    lib = 100
    wib = 30
    tempib = np.linspace(-1, 3, nz)
    salnib = np.ones(nz) * 35

    LIB = [lib]
    WIB = [wib]
    drift_period = np.arange(0, days * 86400, 600)
    for t1, t2 in zip(drift_period[:-1], drift_period[1:]):
        nl, nw = mellat(LIB[-1], WIB[-1], tempib, salnib, nz, t2 - t1)
        LIB.append(nl)
        WIB.append(nw)
    print(
        f"After {days} days the iceberg length has shrinked by {np.round((LIB[0]-LIB[-1])/LIB[0]*100,1)}% "
    )
    fig, ax = plt.subplots()
    ax.plot(drift_period / 86400, LIB, label="iceberg length (m)")
    ax.plot(drift_period / 86400, WIB, label="iceberg width (m)")
    ax.set_title("Iceberg decay according to lateral melting")
    ax.set_xlabel("day")
    plt.legend()
    plt.show()


def test_parm_mellat():
    # Default values
    nz = 10
    lib_def = 100
    wib_def = 30
    tempib_def = np.linspace(-1, 3, nz)
    salnib_def = np.ones(nz) * 35

    lib = np.linspace(100, 1000, 100)
    wib = 1 / 3 * lib
    tempib = [np.ones(nz) * T for T in np.linspace(-2, 5, 100)]
    salnib = [np.ones(nz) * S for S in np.linspace(10, 100, 100)]

    dt = 30 * 86400  # 30 days
    decay_percentage_lib = []
    decay_percentage_tempib = []
    decay_percentage_salnib = []
    for i in range(len(lib)):
        nl, _ = mellat(lib[i], wib_def, tempib_def, salnib_def, nz, dt)
        decay_percentage_lib.append((lib[i] - nl) / lib[i] * 100)
        nl, _ = mellat(lib_def, wib_def, tempib[i], salnib_def, nz, dt)
        decay_percentage_tempib.append((lib_def - nl) / lib_def * 100)
        nl, _ = mellat(lib_def, wib_def, tempib_def, salnib[i], nz, dt)
        decay_percentage_salnib.append((lib_def - nl) / lib_def * 100)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Iceberg decay sensitivity from the parameters after 30 days according to lateral melting"
    )
    axs[0].plot(lib, decay_percentage_lib)
    axs[0].set_xlabel("initial iceberg length (m)")
    axs[1].plot(np.mean(tempib, axis=1), decay_percentage_tempib)
    axs[1].set_xlabel("Average Sea water far field temperature (deg C)")
    axs[2].plot(np.mean(salnib, axis=1), decay_percentage_salnib)
    axs[2].set_xlabel("Average Sea water salinity")
    axs[0].set_ylabel("decay percentage")
    # plt.legend()
    plt.show()


test_mellat(30)
test_parm_mellat()
