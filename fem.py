import numpy as np
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx.mesh import create_interval
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem

# Mesh
mesh = create_interval(MPI.COMM_WORLD, 50, [0,1])
V = FunctionSpace(mesh, ("CG", 1))

# Boundary conditions
def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], 1)

uL = Function(V)
uL.x.array[:] = 0

uR = Function(V)

# Time dependent BC
t = 0
def update_bc():
    uR.x.array[:] = np.sin(2*np.pi*t)

bcL = dirichletbc(uL, locate_dofs_geometrical(V, left))
bcR = dirichletbc(uR, locate_dofs_geometrical(V, right))

# Initial condition
u_n = Function(V)
u_n.interpolate(lambda x: np.sin(np.pi*x[0]) + 0.5*np.sin(3*np.pi*x[0]))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

dt = 0.01
alpha = 1.0

a = u*v*ufl.dx + dt*alpha*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx
L = (u_n*v)*ufl.dx

problem = LinearProblem(a, L, bcs=[bcL, bcR])

# Time loop
for step in range(50):
    t += dt
    update_bc()
    
    uh = problem.solve()
    u_n.x.array[:] = uh.x.array