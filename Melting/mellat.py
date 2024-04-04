def mellat(depthib, lib, wib, tempib, salnib, dpiib, nz):
    # Lateral melting parameterization taken from Kubat et al. 2007
    # An operational iceberg deterioration model

    sumdp, dx, sumVb, deltat = 0.0, 0.0, 0.0, 0.0
    Tf, Tfp, Vb = [0.0] * nz, [0.0] * nz, [0.0] * nz
    k = 0

    while k < nz and (sumdp + 0.5) < abs(depthib):
        Tf[k] = -0.036 - 0.0499 * salnib[k] - 0.000112 * salnib[k] * salnib[k]
        Tfp[k] = Tf[k] * (tempib[k] - Tf[k]) * exp(-0.19 * (tempib[k] - Tf[k]))
        deltat = tempib[k] - Tfp[k]
        Vb[k] = 2.78 * deltat + 0.47 * (deltat ** 2)
        sumVb += Vb[k]
        sumdp += dpiib[k]
        k += 1

    # Unit of sumVb [meter/year]
    # Convert to meter per day
    dx = sumVb / 365

    # Change of iceberg length (on both sides?? -> 2.*)
    lib -= 2 * dx
    wib -= 2 * dx

    # Print the results (equivalent to Fortran write statement)
    print("k-1:", k - 1)
    print("dx:", dx)
    print("dpiib[0:5]:", dpiib[0:5])
    print("Tfp[0:5]:", Tfp[0:5])
    print("Vb[0:5]/365:", [x / 365 for x in Vb[0:5]])
    print("depthib:", depthib)
    print("sumdp:", sumdp)
    print("tempib[0:5]:", tempib[0:5)

# Call the function with your input data
depthib = 100.0
lib = 10.0
wib = 5.0
tempib = [0.0] * nz
salnib = [0.0] * nz
dpiib = [0.0] * nz
nz = len(tempib)

mellat(depthib, lib, wib, tempib, salnib, dpiib, nz)
