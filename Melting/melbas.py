def melbas(depthib, lib, salnib, tempib, dp, uoib, voib, dpuib, dpvib, uib, vib, nz, rhoib):
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
    :param dpuib: Iceberg drift velocity in the x-direction (array of length nz)
    :param dpvib: Iceberg drift velocity in the y-direction (array of length nz)
    :param uib: Iceberg drift velocity in the x-direction
    :param vib: Iceberg drift velocity in the y-direction
    :param nz: Number of layers in the ocean
    :param rhoib: Iceberg density

    The function updates the depthib parameter.
    """

    # Mean ocean velocity over the iceberg draft
    k = 0
    sumdpu = 0.0
    sumdpv = 0.0
    sumu = 0
    sumv = 0
    while k <= nz and (sumdpu <= abs(depthib) or sumdpv <= abs(depthib)):
        k += 1
        sumu += uoib[k - 1]
        sumv += voib[k - 1]
        sumdpu += dpuib[k - 1]
        sumdpv += dpvib[k - 1]

    meanu = sumu / k
    meanv = sumv / k

    # Relative velocity between iceberg and ocean
    velr = ((meanu - uib) ** 2 + (meanv - vib) ** 2) ** 0.5

    # Temperature at the base of the iceberg
    k = 1
    sumdp = 0
    while k <= nz and sumdp + 0.5 < abs(depthib):
        sumdp += dp[k - 1]
        k += 1

    k -= 1
    absv = ((uoib[k - 1] - uib) ** 2 + (voib[k - 1] - vib) ** 2) ** 0.5
    Tf = -0.036 - 0.0499 * salnib[k - 1] - 0.000112 * salnib[k - 1] * salnib[k - 1]
    Tfp = Tf * 2.71828 ** (-0.19 * (tempib[k - 1] - Tf))
    deltat = tempib[k - 1] - Tfp

    Vf = 0.58 * absv ** 0.8 * deltat / (lib ** 0.2)

    # Update the depth
    depthib = abs(depthib) - Vf

    # Thermal constants and calculations are commented out, as they are not used in this code.

    # Conversion in meters per day
    # Vf = Vf * 86400
    # Update the depth
    # depthib = abs(depthib) - Vf

    # Print or return results as needed
    # print(f'k: {k}, absv: {absv}, Tfp: {Tfp}, Vf: {Vf}, tempib: {tempib[k - 1]}')

# Example usage:
# depthib = 10.0
# lib = 1.0
# salnib = [35.0] * nz
# tempib = [0.0] * nz
# dp = [0.1] * nz
# uoib = [0.0] * nz
# voib = [0.0] * nz
# dpuib = [0.0] * nz
# dpvib = [0.0] * nz
# uib = 1.0
# vib = 0.0
# rhoib = 920.0
# melbas(depthib, lib, salnib, tempib, dp, uoib, voib, dpuib, dpvib, uib, vib, nz, rhoib)
