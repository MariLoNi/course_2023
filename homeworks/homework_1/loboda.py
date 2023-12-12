
#Моделирование распределения давления в нагнетательной скважине.
#Реализовать программу по расчёту забойного давления нагнетательной скважины и построить зависимость забойного давления, атм от дебита закачиваемой жидкости, м3/сут (VLP).
#При расчёте учитывайте, что температура меняется согласно геотермическому градиенту (нет теплопотерь). В скважину спущена НКТ до верхних дыр перфорации, угол искривления скважины постоянный. Солёность зависит от входной плотности. Диапазон дебитов жидкости для генерации VLP 1 - 400 м3/сут.
import matplotlib
from math import pi, log, radians, cos
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp
import json

def calc_ws(
        gamma_water: float
) -> float:
    """
    Функция для расчета солесодержания в воде

    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.

    :return: солесодержание в воде, г/г
    """
    ws = (
            1 / (gamma_water * 1000)
            * (1.36545 * gamma_water * 1000 - (3838.77 * gamma_water * 1000 - 2.009 * (gamma_water * 1000) ** 2) ** 0.5)
    )
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws > 0:
        return ws
    else:
        return 0

def calc_rho_w(
        ws: float,
        t: float
) -> float:
    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания

    :param ws: солесодержание воды, г/г
    :param t: температура, К

    :return: плотность воды, кг/м3
    """
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)

    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))


def calc_mu_w(
        ws: float,
        t: float,
        p: float
) -> float:
    """
    Функция для расчета динамической вязкости воды по корреляции Matthews & Russel

    :param ws: солесодержание воды, г/г
    :param t: температура, К
    :param p: давление, Па

    :return: динамическая вязкость воды, сПз
    """
    a = (
            109.574
            - (0.840564 * 1000 * ws)
            + (3.13314 * 1000 * ws ** 2)
            + (8.72213 * 1000 * ws ** 3)
    )
    b = (
            1.12166
            - 2.63951 * ws
            + 6.79461 * ws ** 2
            + 54.7119 * ws ** 3
            - 155.586 * ws ** 4
    )

    mu_w = (
            a * (1.8 * t - 460) ** (-b)
            * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
    )
    return mu_w


def calc_n_re(
        rho_w: float,
        q_ms: float,
        mu_w: float,
        d_tub: float
) -> float:
    """
    Функция для расчета числа Рейнольдса

    :param rho_w: плотность воды, кг/м3
    :param q_ms: дебит жидкости, м3/с
    :param mu_w: динамическая вязкость воды, сПз
    :param d_tub: диаметр НКТ, м

    :return: число Рейнольдса, безразмерн.
    """
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000


def calc_ff_churchill(
        n_re: float,
        roughness: float,
        d_tub: float
) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Churchill

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1/12)
    return ff

def calc_ff_jain(
        n_re: float,
        roughness: float,
        d_tub: float
) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Jain

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    if n_re < 3000:
        ff = 64 / n_re
    else:
        ff = 1 / (1.14 - 2 * np.log10(roughness / d_tub + 21.25 / (n_re**0.9))) ** 2
    return ff

#Расчет распределения давления

def calc_del_p(
        p, l,
        t_wh: float,
        temp_grad: float,
        gamma_water: float,
        roughness: float,
        angle: float,
        d_tub: float,
        q_liq: float
):
    csi = 1 / 10 ** 6
    t = (t_wh + (temp_grad * l) / 100) + 273
    ws = calc_ws(gamma_water)
    rho_w = calc_rho_w(ws, t)
    g = 9.81

    mu_w = calc_mu_w(ws, t, p)
    n_re = calc_n_re(rho_w, q_liq, mu_w, d_tub)
    f = calc_ff_churchill(n_re, roughness, d_tub)

    del_p = csi * (rho_w * g * cos(radians(angle)) - 0.815 * f * rho_w / d_tub ** 5 * q_liq ** 2)
    return del_p

# Функция для расчета градиента давления для произвольного участка скважины
# :param t_wh: температура жидкости у буферной задвижки, градусы цельсия
# :param temp_grad: геотермический градиент, градусы цельсия/100 м
# :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
# :param angle: угол наклона скважины к горизонтали, градусы
# :param q_liq: дебит закачиваемой жидкости
# :param d_tub: диаметр НКТ, м
# :return: градиент давления для произвольного участка скважины

def main_function(data):

    gamma_water = data['gamma_water'] # относительная плотность по пресной воде с плотностью 1000 кг/м3
    md_vdp = data['md_vdp'] # измеренная глубина забоя скважины
    d_tub = data['d_tub'] # диаметр НКТ, м
    angle = data['angle'] # угол наклона скважины к горизонтали, градусы
    roughness = data['roughness'] # шероховатость трубы, м
    p_wh = data['p_wh'] # давление на устье, Па
    t_wh = data['t_wh'] + 273  # температура на устье скважины, K
    temp_grad = data['temp_grad'] * 0.01 # геотермический градиент, К/м * (1e-2)
ql = []
p_wf = []
for q in np.arange(1, 400, 20):
    q_liq = q / (60 * 60 * 24)
    res = solve_ivp(
        calc_del_p,
        t_span=[0, md_vdp],
        y0=[p_wh],
        args=(gamma_water, q_liq, d_tub, angle, roughness, temp_grad, t_wh),
        t_eval=[md_vdp]
        )
    p_wf.append(res.y[0][0])
    ql.append(int(q))

if __name__ == "__main__":

    with open("11.json") as file:
        data = json.load(file)
       res = main_function(data)

    with open(r"output.json", "w", ) as file:
       json.dump(res, file)



