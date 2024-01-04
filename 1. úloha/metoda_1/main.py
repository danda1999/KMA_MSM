import sympy as sp
import numpy as np

# Define the symbols
x, y = sp.symbols('x y')
fun = (1 - x - y)
size = 100
U = np.random.uniform(low=0, high=1, size=size)
V = np.random.uniform(low=0, high=1, size=size)

eq1 = sp.integrate(fun, (x, 0, 1 - y))
eq2 = sp.integrate(eq1, (y, 0, 1))
c = 1 / eq2

# Define the function
function = c * (1 - x - y)

# Integrate with respect to x
cdf = sp.integrate(function, (y, 0, 1 - x))

# Simplify the result
cdf_x = sp.simplify(cdf)

print(cdf_x)

# Integrate with respect to x
cdf = sp.integrate(cdf_x, (x, 0, x))

# Simplify the result
cdf = sp.simplify(cdf)

x_l = []
y_l = []
for i in range(size):
    res_x_fun = sp.solve([x >= 0, x <= 1, cdf - U[i]], x)
    x_i = sp.solve(res_x_fun, x)[0]
    x_l.append(x_i)

    # Combine the numerator and denominator
    expression = function / cdf_x

    # Simplify the expression
    simplified_expression = sp.simplify(expression)
    
    simplified_expression = simplified_expression.subs(x, x_i)

    F_y_x = sp.integrate(simplified_expression, (y, 0, y))

    F_y_x = sp.simplify(F_y_x)

    res_y_fun = sp.solve([y >= 0, y <= 1, F_y_x - V[i]], y)
    y_i = sp.solve(res_y_fun, y)[0]
    print(res_y_fun)
        
        

