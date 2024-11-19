import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import numpy as np
import sympy as sp
import re
import math
import sys
import importlib
from typing import Callable, Optional, Union
from types import ModuleType

sys.path.append("../Scripts")

import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme()
torch.manual_seed(42)


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Computes the partial derivative of
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    Returns:
        (N, D) tensor
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )


def calculate_derivative(
    outputs: torch.Tensor, inputs: torch.Tensor, order: int, idx: int = 0
) -> list[torch.Tensor]:
    """
    Calculate the derivative of the function

    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
        order: Order of the derivative
        idx: Index of the variable to derive
            - 0 for the first variable (usually x in PDEs) (t in ODEs)
            - 1 for the second variable (usually t)
    Returns:
        List of derivatives of the function
    """
    derivatives_list = []
    if order == 0:
        return [outputs]
    derivatives_list.append(grad(outputs, inputs)[0][:, idx : idx + 1])
    for _ in range(order - 1):
        # aqui solo devolvemos el indice(t, x, ...) que queremos (idx) pero para el normal necesitaremos todos
        derivatives_list.append(grad(derivatives_list[-1], inputs)[0][:, idx : idx + 1])

    return derivatives_list


def numerical_derivative(u: np.ndarray, dx: float, orden: int) -> np.ndarray:
    """
    Calculate the numerical derivative of the function for the x variable.
    """
    if orden == 1:
        return (u[2:] - u[:-2]) / (2 * dx)
    elif orden == 2:
        return (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
    elif orden == 3:
        return (u[:-4] - 2 * u[1:-3] + 2 * u[3:-1] - u[4:]) / (2 * dx**3)
    elif orden == 4:
        return (u[:-4] - 4 * u[1:-3] + 6 * u[2:-2] - 4 * u[3:-1] + u[4:]) / dx**4
    else:
        raise ValueError("Derivative order not supported")


def min_max_normalize(
    tensor: torch.Tensor,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> tuple[torch.Tensor, float, float]:
    """
    Normalizes a tensor to the range [0, 1].
    Args:
        tensor: (N, D) tensor
        min_value: float
        max_value: float
    Returns:
        (N, D) tensor, float, float

    """
    if min_value is None:
        min_value = torch.min(tensor).item()
    if max_value is None:
        max_value = torch.max(tensor).item()
    return (tensor - min_value) / (max_value - min_value), min_value, max_value


class DiffEquation(torch.nn.Module):
    def __init__(
        self,
        equation: str,
        func_to_optimize: bool,
        hidden_layers: list = [10],
        activation: str = "sigmoid",
    ) -> None:
        super(DiffEquation, self).__init__()
        # Check if the equation is a PDE
        self.dx = "u_x" in equation
        self.dy = "u_y" in equation
        assert (
            self.dx or not self.dy
        ), "When using 1D PDEs, the equation must contain u_x and not u_y"

        # Store the equation (= 0)
        self.equation = equation
        # Store the constants (can be set later with parameters of the model)
        self.constants: dict[str, Union[float, torch.Tensor]] = {}

        # Get the highest order derivative of the variable
        self.orderx = 0
        self.ordery = 0
        # Formula stores the equation solved for the highest order derivative of t
        self.order, self.formula = self.parse_equation(equation)

        # Default values for the tensor
        self.tensor_max = 1
        self.tensor_min = 0

        self.func_to_optimize = func_to_optimize

        if func_to_optimize:  # NO DY IN THE EQUATION
            # input size is the number of inputs of the pinn + 1 for the solution (U)
            self.input_size = 2 + int(self.dx) * 3
            self.output_size = 1

            # Create the model
            self.activation: torch.nn.Module
            if activation == "sigmoid":
                self.activation = torch.nn.Sigmoid()
            elif activation == "tanh":
                self.activation = torch.nn.Tanh()
            else:
                raise ValueError("Activation function not supported.")

            self.sizes = [self.input_size] + hidden_layers + [self.output_size]
            self.layers = []
            for i in range(1, len(self.sizes) - 1):
                self.layers += [
                    torch.nn.Linear(self.sizes[i - 1], self.sizes[i]),
                    self.activation,
                ]

            self.layers += [torch.nn.Linear(self.sizes[-2], self.sizes[-1])]

            self.model = torch.nn.Sequential(*self.layers)

    def parse_equation(self, equation: str) -> tuple[int, str]:
        """
        Parse the equation string to extract the order and the variable.

        Args:
            equation: Equation as a string.

        Returns:
            order: Order of the equation.
            variable: Formula solved for the highest order derivative of t
        """

        # Regex to match the highest order derivative (e.g., u_tt, u_ttt)
        order_match = re.findall(r"u_t{1,}", equation)

        if not order_match:
            raise ValueError("Equation format is incorrect.")

        # Count the number of 't's to determine the order
        order = len(max(order_match, key=len)) - 2

        if self.dx:
            # Regex to match the highest order derivative for the x variable (e.g., u_xx, u_xxx)
            orderx_match = re.findall(r"u_x{1,}", equation)
            if orderx_match:
                # Count the number of 'x's to determine the order
                self.orderx = len(max(orderx_match, key=len)) - 2
        if self.dy:
            # Regex to match the highest order derivative for the y variable (e.g., u_yy, u_yyy)
            ordery_match = re.findall(r"u_y{1,}", equation)
            if ordery_match:
                # Count the number of 'y's to determine the order
                self.ordery = len(max(ordery_match, key=len)) - 2

        return order, self.clean_variable(equation, max(order_match, key=len))

    def clean_variable(self, equation_str: str, variable_str: str) -> str:
        """
        Solves for the variable string to remove the highest order derivative with sympy.

        Args:
            equation_str: Equation as a string.
            variable_str: Variable to clean as a string.

        Returns:
            variable_str: Cleaned variable string.
        """

        functions = ["exp", "log", "sin", "cos", "pi"]

        # replace the torch. and np. from before the functions
        for f in functions:
            equation_str = equation_str.replace(f"torch.{f}", f)
            equation_str = equation_str.replace(f"np.{f}", f)

        # Define symbolic variables
        variables = sp.symbols(variable_str)

        # Convert the equation string to a symbolic expression
        equation = sp.sympify(
            equation_str,
            locals={function: getattr(sp, function) for function in functions},
        )

        # Solve for the desired variable
        solution = str(sp.solve(equation, variables)[0])

        # Replace the torch. and np. from before the functions
        for f in functions:
            solution = solution.replace(f, f"torch.{f}")

        return solution

    def create_function(self, formula: str, definition=False) -> Callable:
        """
        Create a function from the formula string.

        Args:
            formula: Formula as a string.
            definition: If True, the function is created for the load_data function.
        Returns:
            Function to be used in the ODE solver.
        """

        def replace_funcs(equation_str: str) -> str:
            """
            Replace the mathematical functions with the corresponding library functions.

            I.e., log -> math.log, exp -> math.exp, ... for when the element inside the equation is a constant.
            If the element inside contains a variable, the function is replaced with:

                - Numpy for loading the data, e.g., log(u) -> np.log(u), exp(u) -> np.exp(u), ...

                - Torch for creating the model, e.g., log(u) -> torch.log(u), exp(u) -> torch.exp(u), ...

            Args:
                equation_str: Equation as a string.
            Returns:
                Equation string with the mathematical functions replaced.
            """

            def replace_if_constant(match: re.Match, func: str, library: str) -> str:
                """
                Replace the function with the corresponding library function if the variable is a constant.
                """

                # Extract the variable from the match
                variable = match.group(0)[
                    len(func) + 1 : -1
                ].strip()  # Remove func( and )

                # Check if the variable is in the constants
                # If it is, replace the function with the library function
                if (
                    variable in self.constants
                    and str(self.constants[variable]).isnumeric()
                ):
                    return match.group(0).replace(func, "math." + func)
                else:
                    return match.group(0).replace(func, library + func)

            for f in ["log", "exp", "sin", "cos", "tan", "sqrt"]:
                log_number_pattern = rf"{f}\(\d+(\.\d+)?\)"

                # Replace log of numbers with math.log
                equation_str = re.sub(
                    log_number_pattern,
                    lambda match: match.group(0).replace(f, "math." + f),
                    equation_str,
                )

                # Regex to find logarithms of variables, e.g., log(u)
                log_variable_pattern = rf"(?<![\w.]){f}\(([^)]*[a-zA-Z*][^)]*)\)"
                # Replace log of variables with math.log
                library = "torch."
                if definition:
                    library = "np."
                equation_str = re.sub(
                    log_variable_pattern,
                    lambda match: replace_if_constant(match, f, library),
                    equation_str,
                )

            return equation_str

        formula = replace_funcs(formula)
        if definition:
            formula = formula.replace("torch.", "np.")
        else:
            formula = formula.replace("np.", "torch.")

        def ode_function(
            u: torch.Tensor, t: torch.Tensor, *derivatives
        ) -> torch.Tensor:
            """
            Create a function from the formula string to solve ODEs

            Creates a dictionary (local_dict) with the variables and their values to evaluate the formula.

            Args:
                u: Target.
                t: Time.
                derivatives: Derivatives of the function.
            Returns:

                Function to be used in the PDE solver and to train the model
            """

            # Derivatives of the function
            if definition:
                local_dict = {
                    f'u_t{"t" * I}': derivatives[I]
                    / (self.tensor_max - self.tensor_min) ** (I + 1)
                    for I in range(len(derivatives))
                }
            else:
                derivatives_list = calculate_derivative(u, t, self.order, 0)
                local_dict = {
                    f'u_t{"t" * i}': derivatives_list[i].view(-1, 1)
                    / (self.tensor_max - self.tensor_min) ** (i + 1)
                    for i in range(self.order)
                }

            local_dict["u"] = u
            local_dict["t"] = self.tensor_min + t * (self.tensor_max - self.tensor_min)
            # Add constants to the local dictionary
            local_dict.update(self.constants)
            local_dict["np"] = np
            local_dict["torch"] = torch
            local_dict["math"] = math

            return eval(formula, {}, local_dict)

        def partial_function(
            inputs: torch.Tensor, u: torch.Tensor, dx: float = 1
        ) -> torch.Tensor:
            """
            Create a function from the formula string to solve PDEs

            Creates a dictionary (local_dict) with the variables and their values to evaluate the formula.

            Args:
                inputs: x and t.
                u: Target.
                dx: Step size.
            Returns:

                Function to be used in the PDE solver and to train the model
            """

            local_dict: dict[
                str, Union[torch.Tensor, np.ndarray, float, ModuleType]
            ] = {}

            # Derivatives of the function
            if definition:
                # SOLO PARA ORDEN 1 DE DERIVADA DE TIEMPO
                num_derivatives = [
                    np.pad(numerical_derivative(u, dx, i), (i + 1) // 2)
                    for i in range(1, self.orderx + 1)
                ]
                local_dict.update(
                    {f'u_x{"x" * i}': num_derivatives[i] for i in range(self.orderx)}
                )
                local_dict["x"] = inputs
                # local_dict = {'u_t': derivatives / (self.tensor_max - self.tensor_min)}
                local_dict["t"] = self.tensor_min + inputs * (
                    self.tensor_max - self.tensor_min
                )
            else:
                derivatives_list = calculate_derivative(u, inputs, self.order, 0)
                local_dict.update(
                    {f'u_t{"t" * i}': derivatives_list[i] for i in range(self.order)}
                )

                derivatives_list = calculate_derivative(u, inputs, self.orderx, 1)
                local_dict.update(
                    {f'u_x{"x" * i}': derivatives_list[i] for i in range(self.orderx)}
                )
                if self.dy:
                    derivatives_list = calculate_derivative(u, inputs, self.ordery, 2)
                    local_dict.update(
                        {
                            f'u_y{"y" * i}': derivatives_list[i]
                            for i in range(self.ordery)
                        }
                    )
                local_dict["t"] = inputs[:, 0].view(-1, 1)
                local_dict["x"] = inputs[:, 1].view(-1, 1)

            local_dict["u"] = u

            # Add constants to the local dictionary
            local_dict.update(self.constants)
            local_dict["np"] = np
            local_dict["torch"] = torch
            local_dict["math"] = math

            return eval(formula, {}, local_dict)

        return ode_function if not self.dx else partial_function

    def set_constants(self, constants: dict) -> None:
        """
        Set the values of the constants used in the equation.

        Args:
            constants: Dictionary of constants.
        """
        self.constants.update(constants)

    def forward_pass(self, inputs, outputs) -> torch.Tensor:  # NO DY IN THE EQUATION
        """
        Forward pass of the model if we want to find the function

        Args:
            inputs: Input data.
        Returns:
            Output of the model.
        """

        dT = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        ts_p = torch.cat((inputs, outputs), dim=1)
        if self.dx:
            dT = dT[:, 1:2]
            # Calculate the derivative of the function
            u_x, u_xx = calculate_derivative(
                outputs=outputs, inputs=inputs, order=2, idx=0
            )
            ## Concat ts and temps
            ts_p = torch.cat((inputs, outputs, u_x, u_xx), dim=1)

        ## Compute the linear relation
        P_stack = self.model(ts_p)
        ## Compute the PDE and try to minimize it
        return P_stack - dT

    def __call__(self, definition: bool = False) -> Callable:
        """
        Create the function to be used either in the ODE/PDE solver or to load the data.
        """
        if definition:
            # If we want to load the data, return the function with numpy
            return self.create_function(self.formula, definition=True)
        if self.func_to_optimize:
            # If we want to optimize the function, return the forward pass of the model
            return self.forward_pass
        # If we want to solve the ODE/PDE, return the function with torch (physics loss and so on)
        return self.create_function(self.equation)


class BoundaryCondition:
    def __init__(self, eq: str, constants: dict) -> None:
        self.eq = eq
        self.constants = constants
        self.lhs, self.rhs = eq.split("=")
        self.elements_to_replace = re.findall(r"u[\w.]*", self.lhs)
        self.orders = []
        self.variables_to_derive = []
        for elem in self.elements_to_replace:
            u_deriv = elem.split("_")[0].split("d")
            # We have to calculate the order of the derivative
            if len(u_deriv) > 1:
                self.orders.append(len(u_deriv[1]))
                # We have to know which variable we are deriving (x, y)
                self.variables_to_derive.append(u_deriv[1])
            else:
                self.orders.append(0)
                self.variables_to_derive.append(None)

            # We already know where we will evaluate (udx_t_0 and udx_t_L) because we have the mask

        self.local_dict = {"np": np, "torch": torch}

    def give_value(self, temps: torch.Tensor, ts: torch.Tensor, mask: torch.Tensor):
        mask, normal_matrix = mask
        self.local_dict["u"] = temps[mask != 0]
        self.local_dict["t"] = ts[mask != 0][:, 0:1]
        self.local_dict["x"] = ts[mask != 0][:, 1:2]
        if ts.shape[1] > 2:
            self.local_dict["y"] = ts[mask != 0][:, 2:3]
        
        # Esto depende de que derivada esté calculando
        
        dx = calculate_derivative(temps, ts, max(self.orders), idx=1)
        if ts.shape[1] > 2:
            dy = calculate_derivative(temps, ts, max(self.orders), idx=2)

        for i, elem in enumerate(self.elements_to_replace):
            # Aqui habría que multiplicar por el normal con el tensor de normales
            if self.orders[i] == 0:
                # Aqui cogemos dx porque nos va a devolver el output por ser "derivada" de orden 0
                # Podríamos también coger dy
                self.local_dict[elem] = temps[mask != 0]
            elif self.orders[i] > 0:
                # Concatenamos dx y dy
                if ts.shape[1] <= 2:
                    derivatives = dx[self.orders[i] - 1][mask != 0]
                else:
                    derivatives = torch.cat(
                        (
                            dx[self.orders[i] - 1][mask != 0] if self.variables_to_derive[i] == "x" else temps[mask != 0],
                            dy[self.orders[i] - 1][mask != 0] if self.variables_to_derive[i] == "y" else temps[mask != 0],
                        ),
                        dim=1,
                    )
                
                self.local_dict[elem] = torch.sum(derivatives * normal_matrix, dim=1).view(-1, 1)
                # self.local_dict[elem] = torch.einsum('nm, nm -> n', derivatives, normal_matrix).view(-1, 1)
        self.local_dict.update(self.constants)

        # Check if the shapes are correct
        assert (
            self.local_dict["t"].shape == self.local_dict["u"].shape
        ), f"Shapes don't match (t, u): {self.local_dict['t'].shape} != {self.local_dict['u'].shape}"
        assert (
            self.local_dict["x"].shape == self.local_dict["u"].shape
        ), f"Shapes don't match (x, u): {self.local_dict['x'].shape} != {self.local_dict['u'].shape}"
        if ts.shape[1] > 2:
            assert (
                self.local_dict["y"].shape == self.local_dict["u"].shape
            ), f"Shapes don't match (y, u): {self.local_dict['y'].shape} != {self.local_dict['u'].shape}"
        for elem in self.elements_to_replace:
            assert (
                self.local_dict[elem].shape == self.local_dict["u"].shape
            ), f"Shapes don't match ({elem}, u): {self.local_dict[elem].shape} != {self.local_dict['u'].shape}"

        return torch.mean(
            (eval(self.lhs, {}, self.local_dict) - eval(self.rhs, {}, self.local_dict))
            ** 2
        )


class PINN_inference(nn.Module):
    """
    Physics-Informed Neural Network for Inference.
    Model that learns the solution to a differential equation.
    """

    def __init__(
        self,
        params: list[str] = [],
        input_size: int = 1,
        hidden_layers: list[int] = [10],
        activation: str = "sigmoid",
        bc_type: str = "diri",
        data_from_csv: bool = False,
    ) -> None:
        super(
            PINN_inference,
            self,
        ).__init__()

        self.activation: nn.Module
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise Exception(f"Invalid activation function: {activation}")

        # Define the layers with the activation function
        self.layers: list[torch.nn.Module] = []
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(input_size, hidden_layers[0]),
                        self.activation,
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                        self.activation,
                    )
                )

        self.layers.append(nn.Linear(hidden_layers[-1], 1))

        self.model = nn.Sequential(*self.layers)
        self.params = params

        # Initialize the parameters if we want to predict any (K_H, K_Q, K_th, etc.)
        for key in params:
            setattr(self, key, nn.Parameter(data=torch.abs(torch.randn(1))))

        self.bc_type = bc_type
        self.data_from_csv = data_from_csv

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Forces initial conditions to be satisfied if working with ODEs and the boundary conditions if working with PDEs.
        """

        if self.data_from_csv:
            return (
                torch.exp(self.model(inputs) * inputs) + self.initial_conditions[0] - 1
            )
        return self.model(inputs)

    def physics_loss(
        self, data: torch.Tensor, diff_equation: DiffEquation
    ) -> torch.Tensor:
        """
        Compute the physics loss of the model with the differential equation.
        """

        ts = data.clone().requires_grad_(True).to(DEVICE)
        temps = self(ts)
        # First call the differential equation class to create the function and then evaluates it
        if not diff_equation.dx and not diff_equation.func_to_optimize:
            pde = diff_equation()(temps, ts)
        else:
            pde = diff_equation()(ts, temps)

        return torch.mean(pde**2)

    def initial_conditions_loss(
        self, X_test_tensor: torch.Tensor, diff_equation: DiffEquation
    ) -> torch.Tensor:
        """
        Compute the initial conditions loss of the model.
        """
        ts = X_test_tensor.clone().requires_grad_(True).to(DEVICE)
        temps = self(ts)

        if diff_equation.dx:
            return torch.mean(
                (temps[self.mask_initial_conditions] - self.initial_conditions) ** 2
            )
        else:
            loss = 0
            # le resto 1 al la longitud porque la primera condición inicial no es de una derivada
            derivatives_list = calculate_derivative(
                temps, ts, len(self.initial_conditions) - 1, idx=0
            )
            loss += torch.mean((temps[0] - self.initial_conditions[0]) ** 2)
            for i in range(len(self.initial_conditions) - 1):
                loss += torch.mean(
                    (derivatives_list[i][0] - self.initial_conditions[i + 1]) ** 2
                )
            return loss

    def boundary_conditions_loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the boundary conditions loss of the model.
        """

        ts = data.clone().requires_grad_(True).to(DEVICE)
        temps = self(ts)
        # f = lambda t, x, y: torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-2 * 0.05 * torch.pi ** 2 * t)
        # temps = f(ts[:, 0], ts[:, 1], ts[:, 2]).view(-1, 1)
        loss = 0
        for bc, mask in zip(self.boundary_conditions, self.boundary_masks):
            bc_class = BoundaryCondition(bc, self.constant_values)
            loss += bc_class.give_value(temps, ts, mask)
        return loss

    def update_constants(
        self, diff_equation: DiffEquation, constant_values: dict
    ) -> None:
        """
        Update the constants in the differential equation if we want to predict their value.

        Args:
            model: nn.Module
            diff_equation: DiffEquation
            constant_values: dict
        """
        for key in constant_values.keys():
            # If the model has the constant as a Parameter, update the constant's dictionary
            # in the differential equation to include the parameter
            if hasattr(self, key):
                constant_values[key] = getattr(self, key)

        diff_equation.constants = constant_values

    def create_bc_masks(self, diff_equation: DiffEquation, inputs: torch.Tensor):
        """
        We create the masks for the boundary conditions.

        """
        self.boundary_masks = []

        initial_normal_matrix = torch.ones_like(inputs[:, 1:])

        mask = inputs[:, 1] == torch.min(inputs[:, 1])
        normal_vector = [-1] if not diff_equation.dy else [-1, 0]
        normal_matrix = initial_normal_matrix[mask] * torch.tensor(normal_vector)
        self.boundary_masks.append((mask, normal_matrix))

        mask = inputs[:, 1] == torch.max(inputs[:, 1])
        normal_vector = [1] if not diff_equation.dy else [1, 0]
        normal_matrix = initial_normal_matrix[mask] * torch.tensor(normal_vector)
        self.boundary_masks.append((mask, normal_matrix))

        if diff_equation.dy:
            mask = inputs[:, 2] == torch.min(inputs[:, 2])
            normal_vector = [0, -1]
            normal_matrix = initial_normal_matrix[mask] * torch.tensor(normal_vector)
            self.boundary_masks.append((mask, normal_matrix))

            mask = inputs[:, 2] == torch.max(inputs[:, 2])
            normal_vector = [0, 1]
            normal_matrix = initial_normal_matrix[mask] * torch.tensor(normal_vector)
            self.boundary_masks.append((mask, normal_matrix))

    def process_bc(
        self, diff_equation: DiffEquation, data: tuple, initial_conditions: list[str]
    ):
        _, _, X_test_tensor, y_test_tensor = data

        self.mask_initial_conditions = X_test_tensor[:, 0] == 0
        if diff_equation.dx:
            self.create_bc_masks(diff_equation, X_test_tensor)

        self.initial_conditions = y_test_tensor[self.mask_initial_conditions]

        if diff_equation.dx:
            initial_conditions = [
                ic.replace("np.", "torch.") for ic in initial_conditions
            ]
            self.boundary_conditions = initial_conditions[1:]

        else:
            # Load the initial conditions into the model
            if diff_equation.order > 1:
                self.initial_conditions = list(initial_conditions.values())

    def train(
        self,
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        diff_equation: DiffEquation,
        constant_values: dict,
        initial_conditions: dict,
        lr: float,
        n_epochs: int,
        optimizer_name: str = "lbfgs",
        lambda_pin: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_bc: float = 1.0,
    ) -> None:
        X_tensor, y_tensor, X_test_tensor, y_test_tensor = data

        # Load the initial and boundary conditions into the model
        self.process_bc(diff_equation, data, initial_conditions)
        self.constant_values = constant_values
        # Inicializar el optimizador
        optimizer: torch.optim.Optimizer
        if optimizer_name == "lbfgs":
            print("Using LBFGS")
            optimizer = torch.optim.LBFGS(
                list(self.parameters()) + list(diff_equation.parameters()), lr
            )
        elif optimizer_name == "sgd":
            print("Using SGD")
            optimizer = torch.optim.SGD(
                list(self.parameters()) + list(diff_equation.parameters()), lr
            )
        else:
            print("Using Adam")
            optimizer = torch.optim.Adam(
                list(self.parameters()) + list(diff_equation.parameters()),
                lr,
                amsgrad=True,
            )

        # Definir la función de pérdida MSE
        mse_loss = nn.MSELoss()

        # Listas para almacenar las pérdidas
        self.mse_losses: list[float] = []
        self.physics_losses: list[float] = []
        self.losses: list[float] = []
        self.validation_losses: list[float] = []

        # Entrenar los modelos

        def closure() -> torch.Tensor:
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            T_pred = self(X_tensor)

            self.update_constants(diff_equation, constant_values)

            # Compute the loss function
            # 4 terms: MSE Loss, Physics Loss, Initial Conditions Loss, Boundary Conditions Loss (only if pde)
            mse = mse_loss(T_pred, y_tensor)
            physics_loss = self.physics_loss(X_test_tensor, diff_equation)

            loss = mse + float(lambda_pin) * physics_loss

            # Initial and Boundary Conditions losses
            # if not diff_equation.dy:
            initial_conditions_loss = self.initial_conditions_loss(
                X_test_tensor, diff_equation
            )
            loss += lambda_ic * initial_conditions_loss

            if diff_equation.dx:
                boundary_conditions_loss = self.boundary_conditions_loss(X_test_tensor)
                loss += lambda_bc * boundary_conditions_loss

            loss.backward()

            # Store the training losses
            self.mse_losses.append(mse.item())
            self.physics_losses.append(physics_loss.item())
            self.losses.append(loss.item())

            # Compute and store validation loss
            return loss

        for epoch in range(n_epochs):
            self.model.train()
            optimizer.step(closure)

            self.model.eval()
            with torch.no_grad():
                T_val_pred = self(X_test_tensor)
                val_loss = mse_loss(T_val_pred, y_test_tensor)
                self.validation_losses.append(val_loss.item())

            if epoch % (n_epochs // 10) == 0:
                msg = f"""Epoch: {epoch}, Loss: {self.losses[-1]}, MSE Loss: {self.mse_losses[-1]}, Physics Loss: {self.physics_losses[-1]}, Validation Loss: {self.validation_losses[-1]}"""
                if diff_equation.dx:
                    msg += (
                        f" Boundary Loss:{self.boundary_conditions_loss(X_test_tensor)}"
                    )
                print(msg)

    def plot(self, data: torch.Tensor) -> None:
        X_tensor, y_tensor, X_test_tensor, y_test_tensor = data
        # Predecir y trazar los resultados
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot the first figure
        axs[0].plot(X_test_tensor, y_test_tensor, "r-", label="Real")
        axs[0].plot(
            X_test_tensor,
            self.forward(X_test_tensor).detach().numpy(),
            "b--",
            label="Predicción",
        )
        ## Plot as scatterplot training points
        axs[0].scatter(
            X_tensor,
            self.forward(X_tensor).detach().numpy(),
            color="orange",
            label="Predicted Training points",
        )
        axs[0].scatter(
            X_tensor,
            y_tensor.detach().numpy(),
            color="green",
            label="True Training points",
        )
        axs[0].set_xlabel("Tiempo")
        axs[0].set_ylabel("Temperatura")
        axs[0].legend()

        # Plot the second figure
        axs[1].plot(self.mse_losses, label="MSE Loss")
        axs[1].plot(self.physics_losses, label="Physics Loss")
        axs[1].plot(self.validation_losses, label="Validation Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].set_yscale("log")
        axs[1].legend()

        plt.show()

    def plot_solution(self, data: torch.Tensor) -> None:
        X_train, y_train, X_test, y_test = data
        # Crear Scatter3d para datos de entrenamiento y prueba
        num_test_divisions = int(np.sqrt(len(y_test)))
        t_test = np.linspace(X_test[0, 0], X_test[-1, 0], num_test_divisions)
        x_test = np.linspace(X_test[0, 1], X_test[-1, 1], num_test_divisions)
        u_test = y_test.reshape((num_test_divisions, num_test_divisions))
        scatter_test = go.Scatter3d(
            x=X_test[:, 1],
            y=X_test[:, 0],
            z=y_test[:, 0],
            mode="markers",
            marker=dict(size=4, opacity=0.5, color="blue"),
            name="test Data",
        )

        y_pred = self.forward(X_test).detach().numpy()
        u_test = y_pred.reshape((num_test_divisions, num_test_divisions))
        scatter_pred_all = go.Scatter3d(
            x=X_test[:, 1],
            y=X_test[:, 0],
            z=y_pred[:, 0],
            mode="markers",
            marker=dict(size=2, opacity=0.35, color="red"),
            name="Test Data",
        )
        # TODO: pintar los puntos de boundary conditions e initials
        boundary_condition_points = []
        for mask in self.boundary_masks:
            boundary_condition_points.append(
                go.Scatter3d(
                    x=X_test[:, 1][mask != 0],
                    y=X_test[:, 0][mask != 0],
                    z=y_pred[:, 0][mask != 0],
                    mode="markers",
                    marker=dict(size=4, opacity=0.9, color="red"),
                    name="Test Data",
                )
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
        surface_pred = go.Surface(
            x=x_test,
            y=t_test,
            z=u_test,
            opacity=0.5,
            colorscale="Blues",
            showscale=False,
            name="True Function",
        )

        # Crear el gráfico
        fig = go.Figure(
            data=[scatter_test, scatter_pred_all, surface_true, surface_pred]
            + boundary_condition_points
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="x",
                yaxis_title="t",
                zaxis_title="u(t,x)",
                aspectmode="cube",
            ),
            width=1000,  # Cambia el ancho de la figura
            height=800,
            legend=dict(x=-0.1, y=1.0, font=dict(size=16)),
        )  # Ajusta el tamaño del texto de la leyenda
        fig.show()

    def plot_2d(self, data: torch.Tensor, plate_length: int, max_iter_time: int) -> None:
        print("Generating GIF")
        delta_t = 0.1

        # Precompute the surfaces outside the animate function
        surface_pred = (
            self(data[2])
            .reshape(max_iter_time, plate_length, plate_length)
            .detach()
            .numpy()
        )
        surface_true = (
            data[3].reshape(max_iter_time, plate_length, plate_length).detach().numpy()
        )

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax1, ax2 = axs

        # Initialize the images on both axes
        u_k_pred = surface_pred[0]
        u_k_true = surface_true[0]

        im1 = ax1.imshow(u_k_pred, cmap=plt.cm.jet, vmin=0, vmax=1)
        im2 = ax2.imshow(u_k_true, cmap=plt.cm.jet, vmin=0, vmax=1)

        # Add colorbars to both subplots
        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)

        # Set initial titles and labels
        ax1.set_title("Predicted Temperature at t = 0.000")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        ax2.set_title("True Temperature at t = 0.000")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        def animate(k):
            u_k_pred = surface_pred[k]
            u_k_true = surface_true[k]

            # Update the data in the images
            im1.set_data(u_k_pred)
            im2.set_data(u_k_true)

            # Update the titles to reflect the current time
            ax1.set_title(f"Predicted Temperature at t = {k*delta_t:.3f}")
            ax2.set_title(f"True Temperature at t = {k*delta_t:.3f}")

        # Create the animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_iter_time, interval=1, blit=False
        )

        # Save the animation as a GIF
        writer = animation.PillowWriter(fps=10)
        anim.save("heat_equation_solution.gif", writer=writer)
