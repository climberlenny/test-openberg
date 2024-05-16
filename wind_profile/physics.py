import numpy as np
from scipy.optimize import fsolve, root, minimize
import matplotlib.pyplot as plt


def get_L(u_star, rho_air, cp, H, sst):
    """_summary_

    Args:
        u_star (_type_): friction velocity
        rho_air (_type_): air density
        cp (_type_): thermic capacity
        H (_type_): surface sensible heat flux
    """
    k = 0.4  # von Karman constant
    g = 9.81  # acceleration due to gravity
    if sst < 100:
        sst = sst + 273.15  # conversion in K

    return -(u_star**3 * rho_air * cp * sst) / (k * g * H)


def Psi_m(z, L):
    if L > 0:  # stable conditions
        # See eq 12 DOI 10.1007/s10546-007-9177-6
        am = 5
        bm = am / 6.5
        Bm = ((1 - bm) / bm) ** (1 / 3)
        zeta = z / L
        x = (1 + zeta) ** (1 / 3)
        return -3 * am / bm * (x - 1) + am * Bm / (2 * bm) * (
            2 * np.log((x + Bm) / (1 + Bm))
            - np.log((x**2 - x * Bm + Bm**2) / (1 - Bm + Bm**2))
            + 2
            * np.sqrt(3)
            * (
                np.arctan((2 * x - Bm) / (np.sqrt(3) * Bm))
                - np.arctan((2 - Bm) / (np.sqrt(3) * Bm))
            )
        )

    elif L == 0:
        return 0.0
    else:
        zeta = z / L
        x = (1 - 16 * zeta) ** (1 / 4)
        return (
            2 * np.log((1 + x) / 2)
            + np.log((1 + x**2) / 2)
            - 2 * np.arctan(x)
            + np.pi / 2
        )


def Psi_m2(z, L):
    if L > 0:  # stable conditions
        # see 16959-parametrization-planetary-boundary-layer.pdf eq 48
        a = 1.0
        b = 0.667
        c = 5.0
        d = 0.35
        zeta = z / L
        return -a * zeta - b * (zeta - c / d) * np.exp(-d * zeta) - b * c / d

    elif L == 0:
        return 0.0
    else:
        zeta = z / L
        x = (1 - 16 * zeta) ** (1 / 4)
        return (
            2 * np.log((1 + x) / 2)
            + np.log((1 + x**2) / 2)
            - 2 * np.arctan(x)
            + np.pi / 2
        )


def get_z0_Charnock(u_star, Charnock):
    g = 9.81
    return Charnock * u_star**2 / g


def func_U(z, u_star, k, rho_air, cp, H, Charnock, sst, z0=None, L=None):
    if L is None:
        L = get_L(u_star, rho_air, cp, H, sst)
    if z0 is None:
        z0 = get_z0_Charnock(u_star, Charnock)
    return u_star / k * (np.log(z / z0) - Psi_m(z, L) + Psi_m(z0, L))


def func_U2(z, u_star, k, rho_air, cp, H, Charnock, sst, tss_x, z0=None, L=None):
    if L is None:
        L = get_L(u_star, rho_air, cp, H, sst)
    if z0 is None:
        z0 = get_z0_Charnock(u_star, Charnock)
    return (
        tss_x / (rho_air * k * u_star) * (np.log(z / z0) - Psi_m2(z, L) + Psi_m2(z0, L))
    )


def solve(u10, u100, k, u_star0, rho_air, cp, H, Charnock, sst, z0=None):
    def sfunc_U(LU):
        L, u_star = LU
        return np.array(
            [
                func_U2(10, u_star, k, rho_air, cp, H, Charnock, sst, z0, L) - u10,
                func_U2(100, u_star, k, rho_air, cp, H, Charnock, sst, z0, L) - u100,
            ]
        )

    L = get_L(u_star0, rho_air, cp, H, sst)
    sol = fsolve(sfunc_U, (L, u_star0))
    return sol


def solve_L_gpt(u10, u100, u_star, k, rho_air, cp, H, Charnock, sst, z0=None):
    def sfunc_U(L):
        ret = np.array(
            [
                func_U2(10, u_star, k, rho_air, cp, H, Charnock, sst, z0, L) - u10,
                func_U2(100, u_star, k, rho_air, cp, H, Charnock, sst, z0, L) - u100,
            ]
        ).flatten()
        print(ret)
        return ret

    # Use root to solve for L
    result = root(sfunc_U, 100.0, nan_policy="raise")

    # Check the success of the root-finding
    if result.success:
        return result.x
    else:
        raise ValueError("Failed to find a solution for L: " + result.message)


def solve_u_star(u10, k, rho_air, cp, H, Charnock, sst, z0=None):
    """solve u_star knowing U at 10m

    Args:
        u10 (_type_): _description_
        k (_type_): _description_
        rho_air (_type_): _description_
        cp (_type_): _description_
        H (_type_): _description_
        Charnock (_type_): _description_
        sst (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = 10

    def sfunc_U(u_star):
        return func_U(z, u_star, k, rho_air, cp, H, Charnock, sst, z0) - u10

    u_star = fsolve(sfunc_U, 0.5)
    return u_star


def solve_u_star2(u10, k, rho_air, cp, H, Charnock, sst, tss_x, z0=None):
    """solve u_star knowing U at 10m

    Args:
        u10 (_type_): _description_
        k (_type_): _description_
        rho_air (_type_): _description_
        cp (_type_): _description_
        H (_type_): _description_
        Charnock (_type_): _description_
        sst (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = 10

    def sfunc_U(u_star):
        return func_U2(z, u_star, k, rho_air, cp, H, Charnock, sst, tss_x, z0) - u10

    u_star = fsolve(sfunc_U, 0.5)
    return u_star


u10 = 9.76770536066395  # 10
u100 = 11.307481875269797
z0 = 0.00023747208929880692
H = -72_367.72441365421 / 3600  # J/m**2
t2m = 15.563698849990885
u_star_data = 0.3627694618960379
rho_air = 1.2092323289979807
tss_x = 560.1134374213188 / 3600
g = 9.81
k = 0.4
# k_viscosity = 1.5e-5
cp = 1005  # J/(kg*K)
# Charnock = np.linspace(0.01, 0.06, 100)
Charnock = 0.01826902039019031
U_star = []
# for chnk in Charnock:
#     u_star = solve_u_star(u10, k, rho_air, cp, H, chnk, sst)
#     print(u_star)
#     U_star.append(u_star[0])
# U_star = np.array(U_star)
# plt.plot(Charnock, U_star)
# plt.show()
# print(f"u_star = {u_star[0]}")
# Z0 = get_z0_Charnock(U_star, Charnock)
# plt.plot(Charnock, Z0)
# plt.show()
L, u_star = solve(u10, u100, u_star_data, k, rho_air, cp, H, Charnock, t2m, z0)

# u_star = solve_u_star(u10, k, rho_air, cp, H, Charnock, t2m, z0)
# u_star2 = solve_u_star2(u10, k, rho_air, cp, H, Charnock, t2m, tss_x)
# z0_sol = get_z0_Charnock(u_star, Charnock)
# z0_sol2 = get_z0_Charnock(u_star2, Charnock)
# print(
#     f"z0 = {z0} vs z0_sol = {z0_sol} vs z0_sol2 = {z0_sol2}, delta_r = {(z0-z0_sol[0])/z0*100}% | delta_r2 = {(z0-z0_sol2[0])/z0*100}%"
# )
# print(
#     f"u_star_data = {u_star_data} vs u_star_sol = {u_star} vs u_star_sol2 = {u_star2}, delta_r = {(u_star_data-u_star[0])/u_star_data*100}% | delta_r2 = {(u_star_data-u_star2[0])/u_star_data*100}% "
# )
Z = np.linspace(1, 100, 100)
# L = np.linspace(-500, 500, 1000)
# U_10 = np.array(
#     [
#         func_U(
#             10,
#             u_star=u_star_data,
#             k=k,
#             rho_air=rho_air,
#             cp=cp,
#             H=H,
#             Charnock=Charnock,
#             sst=t2m,
#             z0=z0,
#             L=l,
#         )
#         for l in L
#     ]
# )
# U_100 = np.array(
#     [
#         func_U(
#             100,
#             u_star=u_star_data,
#             k=k,
#             rho_air=rho_air,
#             cp=cp,
#             H=H,
#             Charnock=Charnock,
#             sst=t2m,
#             z0=z0,
#             L=l,
#         )
#         for l in L
#     ]
# )
# U2_10 = np.array(
#     [
#         func_U2(
#             10,
#             u_star=u_star_data,
#             k=k,
#             rho_air=rho_air,
#             cp=cp,
#             H=H,
#             Charnock=Charnock,
#             sst=t2m,
#             tss_x=tss_x,
#             z0=z0,
#             L=l,
#         )
#         for l in L
#     ]
# )
# U2_100 = np.array(
#     [
#         func_U2(
#             100,
#             u_star=u_star_data,
#             k=k,
#             rho_air=rho_air,
#             cp=cp,
#             H=H,
#             Charnock=Charnock,
#             sst=t2m,
#             tss_x=tss_x,
#             z0=z0,
#             L=l,
#         )
#         for l in L
#     ]
# )
# plt.plot(L, np.abs(U2_10 - u10), label="U2_10")
# plt.plot(L, np.abs(U2_100 - u100), label="U2_100")
# plt.plot(L, np.abs(U_10 - u10), label="U_10")
# plt.plot(L, np.abs(U_100 - u100), label="U_100")
# plt.hlines(0, L.min(), L.max(), linestyle=":", color="k")
# plt.legend()
# plt.show()


U = func_U(
    Z,
    u_star=u_star,
    k=k,
    rho_air=rho_air,
    cp=cp,
    H=H,
    Charnock=Charnock,
    sst=t2m,
    z0=z0,
)
U2 = func_U2(
    Z,
    u_star=u_star,
    k=k,
    rho_air=rho_air,
    cp=cp,
    H=H,
    Charnock=Charnock,
    sst=t2m,
    tss_x=tss_x,
    z0=z0,
    L=L,
)
plt.plot(U, Z, label="U1")
plt.plot(U2, Z, label="U2")
plt.yscale("log")
plt.scatter(u10, 10, color="r")
plt.scatter(u100, 100, color="r")
plt.hlines(10, U.min(), U.max(), linestyle=":", color="k")
plt.hlines(100, U.min(), U.max(), linestyle=":", color="k")
plt.vlines(U.mean(), Z.min(), Z.max(), linestyle=":", color="g", label="u_mean")
plt.vlines(u10, Z.min(), Z.max(), linestyle=":", color="r", label="u10")
plt.vlines(u100, Z.min(), Z.max(), linestyle=":", color="b", label="u100")
plt.xlabel("Wind velocity U (m/s)")
plt.ylabel("height z (m)")
plt.legend()
plt.show()
