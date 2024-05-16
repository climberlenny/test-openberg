import numpy as np
import matplotlib.pyplot as plt


def melwav(lib, wib, uaib, vaib, sst, conc, dt):
    """update length and width value according to wave melting

    Args:
        lib (_type_): iceberg length
        wib (_type_): iceberg width
        uaib (_type_): wind speed u component
        vaib (_type_): wind speed v component
        sst (_type_): Sea surface temperature
        conc (_type_): sea ice concentration
    """
    if lib == 0:
        return 0.0, 0.0
    Ss = -5 + np.sqrt(32 + 2 * np.sqrt(uaib**2 + vaib**2))
    Vsst = (1 / 6.0) * (sst + 2) * Ss
    Vwe = Vsst * 0.5 * (1 + np.cos(np.pi * conc**3))  # melting in m/day ?
    Vwe /= 86400  # melting in m/s ?

    # length lost only on one side
    new_lib = max(0, lib - Vwe * dt)
    new_wib = max(0, wib / lib * new_lib)
    return new_lib, new_wib


def test_melwav(days=30):
    lib = 100  # Replace with your initial value for lib
    wib = 30  # Replace with your initial value for wib
    uaib = 5.0  # Replace with your value for uaib
    vaib = 0  # Replace with your value for vaib
    sst = 2.0  # Replace with your value for sst
    conc = 0.0  # Replace with your value for conc

    drift_period = np.arange(0, 86400 * days, 600)
    LIB = [lib]
    WIB = [wib]
    for t1, t2 in zip(drift_period[:-1], drift_period[1:]):
        dt = t2 - t1
        lib, wib = melwav(lib, wib, uaib, vaib, sst, conc, dt)
        LIB.append(lib)
        WIB.append(wib)
    print(
        f"After {days} days the iceberg length has shrinked by {np.round((LIB[0]-LIB[-1])/LIB[0]*100,1)}% "
    )
    fig, ax = plt.subplots()
    ax.plot(drift_period / 86400, LIB, label="iceberg length (m)")
    ax.plot(drift_period / 86400, WIB, label="iceberg width (m)")
    ax.set_title("Iceberg decay according to wave melting")
    ax.set_xlabel("day")
    plt.legend()
    plt.show()


def test_parm_melwav():
    # Default values
    lib_def = 100
    wib_def = 30
    uaib_def = 5.0
    vaib_def = 0
    sst_def = 2.0
    conc_def = 0.0

    lib = np.linspace(100, 1000, 100)
    wib = 1 / 3 * lib
    wind = np.linspace(0, 50, 100)
    alpha = np.pi / 4
    uaib = np.cos(alpha) * wind
    vaib = np.sin(alpha) * wind
    sst = np.linspace(-1, 10, 100)
    conc = np.linspace(0, 1, 100)

    dt = 30 * 86400  # 30 days
    decay_percentage_lib = []
    decay_percentage_wind = []
    decay_percentage_sst = []
    decay_percentage_conc = []
    for i in range(len(lib)):
        nl, _ = melwav(lib[i], wib_def, uaib_def, vaib_def, sst_def, conc_def, dt)
        decay_percentage_lib.append((lib[i] - nl) / lib[i] * 100)
        nl, _ = melwav(lib_def, wib_def, uaib[i], vaib[i], sst_def, conc_def, dt)
        decay_percentage_wind.append((lib_def - nl) / lib_def * 100)
        nl, _ = melwav(lib_def, wib_def, uaib_def, vaib_def, sst[i], conc_def, dt)
        decay_percentage_sst.append((lib_def - nl) / lib_def * 100)
        nl, _ = melwav(lib_def, wib_def, uaib_def, vaib_def, sst_def, conc[i], dt)
        decay_percentage_conc.append((lib_def - nl) / lib_def * 100)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(
        "Iceberg decay sensitivity from the parameters after 30 days according to wave melting"
    )
    axs[0, 0].plot(lib, decay_percentage_lib)
    axs[0, 0].set_xlabel("initial iceberg length (m)")
    axs[0, 1].plot(wind, decay_percentage_wind)
    axs[0, 1].set_xlabel("wind speed (m/s)")
    axs[1, 0].plot(sst, decay_percentage_sst)
    axs[1, 0].set_xlabel("Sea Surface Temperature deg celcius")
    axs[1, 1].plot(conc, decay_percentage_conc)
    axs[1, 1].set_xlabel("Sea ice concentration %")
    axs[0, 0].set_ylabel("decay percentage")
    axs[1, 0].set_ylabel("decay percentage")
    # plt.legend()
    plt.show()


test_melwav()
test_parm_melwav()
