from sympy import *

t, l = symbols('t l')

H = Matrix([l*cos(t), l*sin(t)])
Hdot = H.jacobian([t])

print(Hdot)