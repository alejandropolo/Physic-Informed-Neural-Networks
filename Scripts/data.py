import sys
import importlib

import torch
import numpy as np
import pandas as pd
from typing import Callable

from scipy.integrate import solve_ivp
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.append("../Scripts/")
import PINN_moredim

importlib.reload(PINN_moredim)
from PINN_moredim import DiffEquation, min_max_normalize

# from train import train

import warnings

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)


def nth_order_ode_solver(
    f: Callable, initial_conditions: list[float], t: np.ndarray
) -> np.ndarray:
    """
    Solve an nth-order ODE by converting it into a system of first-order ODEs.

    Args:
        f: Function that defines the ODE.
        initial_conditions: List of initial conditions.
        t: Time points at which to evaluate the solution.
    Returns:
        Solution to the ODE.
    """

    def system(state, t):
        derivatives = state[1:].tolist() + [f(*state, t)]
        return derivatives

    solution = odeint(system, initial_conditions, t)
    return solution[:, 0]  # Return only the u values


def load_data_ode(
    diff_equation: DiffEquation,
    params_to_optimize: list,
    initial_conditions: list,
    t_0: float,
    t_fin: float,
    n_points: int = 1,
    noise: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ###### With new generalization
    order = diff_equation.order
    ini_conditions = [0 for _ in range(order)]
    for equation in initial_conditions:
        lhs, rhs = equation.split("=")
        if "." not in lhs:
            ini_conditions[0] = float(rhs)
        else:
            order_str = lhs.split("_")[0].split(".")[1]
            order = len(order_str)
            ini_conditions[order] = float(rhs)

    ode_function: Callable = diff_equation(definition=True)

    # Define the time points at which to evaluate the solution
    n_steps = 1000
    times = np.linspace(t_0, t_fin, n_steps)
    ## Now use the numerical solution
    temps = nth_order_ode_solver(ode_function, ini_conditions, times)

    # Make training data
    if params_to_optimize or diff_equation.func_to_optimize:
        t = np.linspace(t_0, t_fin, n_points)
    else:  # Solve equation
        n_points = 1
        t = np.linspace(t_0, t_0 + (t_fin - t_0) / 10, n_points)
    T = nth_order_ode_solver(ode_function, ini_conditions, t) + noise * np.random.randn(
        n_points
    )

    ### Generate normalized test data
    X_test = times.reshape(-1, 1)
    y_test = temps.reshape(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    X_test_tensor, tensor_min, tensor_max = min_max_normalize(X_test_tensor)

    ## Generate training data
    X = t.reshape(-1, 1)
    y = T.reshape(-1, 1)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    X_tensor, _, _ = min_max_normalize(X_tensor, tensor_min, tensor_max)

    plt.plot(times, temps)
    plt.plot(t, T, "o")
    plt.legend(["Equation", "Training data"])
    plt.ylabel("Temperature (C)")
    plt.xlabel("Time (s)")
    plt.title("Simulated data")
    plt.show()

    # Lo devolvemos así para poder cargar en
    diff_equation.tensor_max = tensor_max
    diff_equation.tensor_min = tensor_min
    return X_tensor, y_tensor, X_test_tensor, y_test_tensor


def load_data_pde(
    diff_equation: DiffEquation,
    params_to_optimize: bool,
    L_min: float,
    L: float,
    T: float,
    N_train: int,
    N_test: int,
    initial_conditions: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates the data for equations with 2 variables (x, t).

    Args:
        diff_equation: Differential equation to solve.
        params_to_optimize: If True, the parameters are optimized.
        L: Length of the plate.
        T: Final time.
        N_train: Number of training points.
        N_test: Number of test points.
        initial_conditions: Initial conditions of the heat equation.

    """
    t_train = np.linspace(0, T, N_train)
    x_train = np.linspace(L_min, L, N_train)

    dx_train = x_train[1] - x_train[0]
    T_mesh_train, X_mesh_train = np.meshgrid(t_train, x_train, indexing="ij")  # cambiar

    part_function = diff_equation(definition=True)

    # Inicialización de la matriz u
    u = np.zeros((N_train, N_train))
    for eq in initial_conditions:
        lhs, rhs = eq.split("=")
        ic = lhs.strip().split("_")[1:]
        ini_func = lambda x: eval(
            str(rhs), {"x": x, "t": 0}, {"np": np, "torch": torch}
        )
        if ic[1] == "x":
            # To be able to evaluate the string as a function
            u[int(ic[0]), :] = ini_func(X_mesh_train[0, :])
        elif ic[0] == "t":
            if ic[1].strip() == "L":
                u[:, -1] = ini_func(X_mesh_train[:, -1])
            else:
                u[:, int(ic[1])] = ini_func(X_mesh_train[:, 0])
        else:
            u[int(ic[0]), int(ic[1])] = ini_func(X_mesh_train[int(ic[0]), int(ic[1])])

    # Resolución de la ecuación del calor for the current u_L_t value
    sol_train = solve_ivp(
        part_function, (0, T), u[0, :], t_eval=t_train, args=(dx_train,)
    )
    u_train = sol_train.y.T

    ### Generamos los datos de train
    mesh_train = np.hstack((T_mesh_train.reshape(-1, 1), X_mesh_train.reshape(-1, 1)))
    input_mesh_train = torch.tensor(mesh_train).float()
    y_train = torch.tensor(u_train).reshape(len(input_mesh_train), 1).float()

    if not (diff_equation.func_to_optimize or params_to_optimize):
        # Cogemos solo los datos en x == 0 y x == L
        mask = (
            (input_mesh_train[:, 0] == 0)
            | (input_mesh_train[:, 1] == L_min)
            | (input_mesh_train[:, 1] == L)
        )
        input_mesh_train = input_mesh_train[mask]
        y_train = y_train[mask]

    t_test = np.linspace(0, T, N_test)
    x_test = np.linspace(L_min, L, N_test)
    dx_test = x_test[1] - x_test[0]
    T_mesh_test, X_mesh_test = np.meshgrid(t_test, x_test, indexing="ij")

    u = np.zeros((N_test, N_test))

    for eq in initial_conditions:
        lhs, rhs = eq.split("=")
        ic = lhs.strip().split("_")[1:]
        ini_func = lambda x: eval(
            str(rhs), {"x": x, "t": 0}, {"np": np, "torch": torch}
        )
        if ic[1] == "x":
            # To be able to evaluate the string as a function
            u[int(ic[0]), :] = ini_func(X_mesh_test[0, :])
        elif ic[0] == "t":
            if ic[1].strip() == "L":
                u[:, -1] = ini_func(X_mesh_test[:, -1])
            else:
                u[:, int(ic[1])] = ini_func(X_mesh_test[:, 0])
        else:
            u[int(ic[0]), int(ic[1])] = ini_func(X_mesh_test[int(ic[0]), int(ic[1])])

    # Resolución de la ecuación del calor for the current u_L_t value
    sol_test = solve_ivp(part_function, (0, T), u[0, :], t_eval=t_test, args=(dx_test,))
    u_test = sol_test.y.T

    mesh_test = np.hstack((T_mesh_test.reshape(-1, 1), X_mesh_test.reshape(-1, 1)))
    input_mesh_test = torch.tensor(mesh_test).float()
    y_test = torch.tensor(u_test).reshape(len(input_mesh_test), 1).float()

    # Crear Scatter3d para datos de entrenamiento y prueba
    scatter_train = go.Scatter3d(
        x=input_mesh_train[:, 1],
        y=input_mesh_train[:, 0],
        z=y_train[:, 0],
        mode="markers",
        marker=dict(size=4, opacity=0.7, color="blue"),
        name="Train Data",
    )

    scatter_test = go.Scatter3d(
        x=input_mesh_test[:, 1],
        y=input_mesh_test[:, 0],
        z=y_test[:, 0],
        mode="markers",
        marker=dict(size=2, opacity=0.2, color="red"),
        name="Test Data",
    )

    # Crear la superficie para la función subyacente
    surface_true = go.Surface(
        x=x_test,
        y=t_test,
        z=u_test,
        opacity=0.5,
        colorscale="Blues",
        showscale=False,
        name="True Function",
    )

    # Crear el gráfico
    fig = go.Figure(data=[scatter_test, scatter_train, surface_true])
    fig.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="t", zaxis_title="u(t,x)", aspectmode="cube"
        ),
        width=1000,  # Cambia el ancho de la figura
        height=800,
        legend=dict(x=-0.1, y=1.0, font=dict(size=16)),
    )  # Ajusta el tamaño del texto de la leyenda
    fig.show()

    return (input_mesh_train, y_train, input_mesh_test, y_test)


def load_data_moredim(
    diff_equation: DiffEquation,
    f: str,
    params_to_optimize: bool,
    L: float,
    T: float,
    plate_length: int,
    max_iter_time: int,
):
    # solution
    func = lambda t, x, y: eval(
        f, {"t": t, "x": x, "y": y, "torch": torch, "np": np}, diff_equation.constants
    )

    train_t = np.linspace(0, T, max_iter_time)
    train_x = np.linspace(0, L, plate_length)
    train_y = np.linspace(0, L, plate_length)

    T_mesh, X_mesh, Y_mesh = np.meshgrid(
        train_t, train_x, train_y, indexing="ij"
    )  # cambiar
    mesh = np.hstack(
        (T_mesh.reshape(-1, 1), X_mesh.reshape(-1, 1), Y_mesh.reshape(-1, 1))
    )
    input_mesh = torch.tensor(mesh).float()
    y = func(input_mesh[:, 0], input_mesh[:, 1], input_mesh[:, 2])
    y = torch.tensor(y).reshape(len(input_mesh), 1).float()

    if not (diff_equation.func_to_optimize or params_to_optimize):  # False: #
        # Cogemos solo los datos en x == 0 y x == L
        mask = (
            (input_mesh[:, 0] == 0)
            | (input_mesh[:, 1] == 0)
            | (input_mesh[:, 1] == L)
            | (input_mesh[:, 2] == 0)
            | (input_mesh[:, 2] == L)
        )
        input_mesh_train = input_mesh[mask]
        y_train = y[mask]

    return (input_mesh_train, y_train, input_mesh, y)


def load_data_neumann(
    diff_equation: DiffEquation,
    f: str,
    params_to_optimize: bool,
    L_min: float,
    L: float,
    T: float,
    N_train: int,
):
    # solution

    func = lambda t, x: eval(
        f,
        {"t": t, "x": x, "torch": torch, "np": np, "pi": np.pi},
        diff_equation.constants,
    )

    train_t = np.linspace(0, T, N_train)
    train_x = np.linspace(eval(str(L_min)), eval(str(L)), N_train)

    T_mesh, X_mesh = np.meshgrid(train_t, train_x, indexing="ij")  # cambiar
    mesh = np.hstack((T_mesh.reshape(-1, 1), X_mesh.reshape(-1, 1)))
    input_mesh = torch.tensor(mesh).float()
    y = func(input_mesh[:, 0], input_mesh[:, 1])
    y = torch.tensor(y).reshape(len(input_mesh), 1).float()

    if not (diff_equation.func_to_optimize or params_to_optimize):
        # Cogemos solo los datos en x == 0 y x == L
        mask = (
            input_mesh[:, 0]
            == 0
            # | (input_mesh[:, 0] < 0.12)
            # (input_mesh[:, 1] == L_min) |
            # (input_mesh[:, 1] == L)
        )
        input_mesh_train = input_mesh[mask]
        y_train = y[mask]

    # Crear Scatter3d para datos de entrenamiento y prueba
    scatter_train = go.Scatter3d(
        x=input_mesh_train[:, 1],
        y=input_mesh_train[:, 0],
        z=y_train[:, 0],
        mode="markers",
        marker=dict(size=4, opacity=0.7, color="blue"),
        name="Train Data",
    )

    scatter_test = go.Scatter3d(
        x=input_mesh[:, 1],
        y=input_mesh[:, 0],
        z=y[:, 0],
        mode="markers",
        marker=dict(size=2, opacity=0.2, color="red"),
        name="Test Data",
    )

    # Crear la superficie para la función subyacente
    surface_true = go.Surface(
        x=train_x,
        y=train_t,
        z=y.reshape(N_train, N_train),
        opacity=0.5,
        colorscale="Blues",
        showscale=False,
        name="True Function",
    )

    # Crear el gráfico
    fig = go.Figure(data=[scatter_test, scatter_train, surface_true])
    fig.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="t", zaxis_title="u(t,x)", aspectmode="cube"
        ),
        width=1000,  # Cambia el ancho de la figura
        height=800,
        legend=dict(x=-0.1, y=1.0, font=dict(size=16)),
    )  # Ajusta el tamaño del texto de la leyenda
    fig.show()
    return (input_mesh_train, y_train, input_mesh, y)


def load_csv_data(
    diff_equation: DiffEquation,
    filename: str,
    params_to_optimize: list[str] = [],
    noise: float = 0.0,
):
    col = "Voltage_Eq"

    df = pd.read_csv(filename)
    df["Time"] = df["Time"].round(
        2
    )  ## Evita efectos raros en las derivasdas por valores temporales muy proximos
    df.drop_duplicates(subset=["Time"], keep="first", inplace=True)
    df.drop_duplicates(subset=["v_cell"], keep="first", inplace=True)
    df.drop_duplicates(subset=["t_membrane"], keep="first", inplace=True)

    df = df.loc[df.Time > 200]
    df = df[(df["T"] > 79) & (df["T"] < 81)]
    k_1_mean = df["k_1"].mean()
    k_2_mean = df["k_2"].mean()
    k_3_mean = df["k_3"].mean()

    df["Voltage_Eq"] = (
        k_1_mean
        + k_2_mean * np.log(df["power_elec"] / df["v_cell"])
        + k_3_mean * df["power_elec"] * df["t_membrane"] / df["v_cell"]
    )
    k_1_mean_mod = k_1_mean + k_2_mean * np.log(100000.0) - k_2_mean

    X_test_tensor = torch.tensor(
        df["Time"].values, dtype=torch.float32, device=DEVICE
    ).view(-1, 1)
    y_test_tensor = torch.tensor(
        df[col].values, dtype=torch.float32, device=DEVICE
    ).view(-1, 1)

    ## Min Max Scale
    X_test_tensor, tensor_min, tensor_max = min_max_normalize(X_test_tensor)

    ## Select n first points as train
    n_points = 1
    t = np.round(np.linspace(0, len(X_test_tensor) // 2, n_points))
    if params_to_optimize:
        n_points = 6
        t = np.round(np.linspace(0, len(X_test_tensor) // 2, n_points))

        t = np.concatenate((t, np.array([len(X_test_tensor) - 40])))
        t = np.concatenate((t, np.array([len(X_test_tensor) - 25])))
        # t = np.concatenate((t,np.array([len(X_test_tensor)-10])))
        # t = np.concatenate((t,np.array([len(X_test_tensor)-1])))

    X_tensor = X_test_tensor[t]
    y_tensor = y_test_tensor[t]

    ## Add noise to y_tensor but not to the first point
    y_tensor[1:] = y_tensor[1:] + noise * torch.randn(len(y_tensor) - 1, 1)

    ## Scatterplot of the training data and test data
    plt.scatter(
        X_test_tensor.detach().numpy(),
        y_test_tensor.detach().numpy(),
        label="Test data",
        s=5,
    )
    plt.scatter(
        X_tensor.detach().numpy(),
        y_tensor.detach().numpy(),
        label="Training data",
        color="orange",
    )
    ## Plot also real voltage
    # plt.plot(X_test_tensor.detach().numpy(), df['Voltage_Eq'].values,label='Real Voltage',color='red')
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.title("Training data")
    plt.legend()
    plt.show()

    diff_equation.tensor_max = tensor_max
    diff_equation.tensor_min = tensor_min
    constants = {
        "k_1_mean_mod": k_1_mean_mod,
        "k_1_mean": k_1_mean,
        "k_2_mean": k_2_mean,
        "k_3_mean": k_3_mean,
    }
    diff_equation.set_constants(constants)

    return (X_tensor, y_tensor, X_test_tensor, y_test_tensor)
