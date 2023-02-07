from fenics import *
import numpy as np
from scipy.fftpack import fft, fft2, fftn
import time
from mcmc import Metropolis_in_Gibbs_burn_in
from conv_checks import *

class make_observation:
    def __init__(self, c, r, sigma_1, theta):
        np.random.seed(420)
        # Generate unit square mesh
        self.m = 3                             # we choose M as a power of 2 to speed up fft computation.
        self.n = 2**self.m                          # Data mesh number of gridpoints
        self.mesh = UnitCubeMesh(self.n, self.n, self.n)       # Data mesh
        # Define spaces
        self.V = FunctionSpace(self.mesh, "CG", 1)   #First order Continuous Galerkin (linear Lagrange element)
        self.V_grad = VectorFunctionSpace(self.mesh, "DG", 0)  # We use a vector function space since the gradient is 2D
        self.uleft = Constant("0.00")                # rhs of weak formulation
        self.r = r
        self.c = c
        self.theta = theta
        self.sigma_1 = sigma_1
        self.bcs = DirichletBC(self.V, Expression('x[0]',degree=1), "on_boundary")
        self.sigma_0 = 1.0   # conductivity of 1 outside ball
        
        def solver(self):
            c, r, theta = self.c, self.r, self.theta
            sigma_0, sigma_1 = self.sigma_0, self.sigma_1
            # True conductivity function, we make a random circle as the object inside the unit square
            class sigmafun(UserExpression):
                def eval(self, values, x):
                    if (((x[0]-c[0])*np.cos(theta)+(x[1]-c[1])*np.sin(theta))**2/r[0]**2\
                        +((x[1]-c[1])*np.cos(theta)-(x[0]-c[0])*np.sin(theta))**2/r[1]**2\
                        +(x[2]-c[2])**2/r[2]**2)<=1:
                        values[0] = sigma_1

                    else:
                        values[0] = sigma_0
                        
            # Initialize conductivity sigma         
            femel = self.V.ufl_element()
            sigma_tru = sigmafun(element=femel)
            sigma_true = project(sigma_tru, self.V, solver_type='bicgstab',preconditioner_type='sor')
            sigma = ConsSig(sigma_true)
            # Define functions and variational form
            self.u = TrialFunction(self.V)
            self.v = TestFunction(self.V)
            a = inner(sigma*grad(self.u),grad(self.v))*dx     # lhs of the weak formulation
            L = self.uleft*self.v*dx
            # Solve the system
            u = Function(self.V)
            solve(a == L, u, self.bcs, solver_parameters = {'linear_solver':'bicgstab', 'preconditioner':'sor'})        
            sigma_f = Function(self.V)
            sigma_f = interpolate(sigma_true, self.V)
            return u, sigma_f
        
        def ConsSig(sigma):
            nu=0;        # lower bound [S/mm],    # lower and upper bounds for conductivity
            mu=100;      # upper bound [S/mm]
            value=sigma.vector().get_local()
            value[np.where(value < nu)]=nu
            value[np.where(value > mu)]=mu
            sigma.vector()[:]=value
            return sigma  
        
        def J_field(self):
            # Compute the current density field
            J_val = project(- self.sigma*grad(self.u), self.V_grad, solver_type='bicgstab',preconditioner_type='sor').compute_vertex_values(self.mesh)
            Jx = J_val[0:(self.n+1)**3].reshape(self.n+1, self.n+1, self.n+1)
            Jy = J_val[(self.n+1)**3:2*(self.n+1)**3].reshape(self.n+1, self.n+1, self.n+1)
            Jz = J_val[2*(self.n+1)**3:].reshape(self.n+1, self.n+1, self.n+1)
            return Jx, Jy
        
        def FFTCONV(Jx, Jy, n):
            ''' Calculates the magnetic field caused by a current density distribution Jx, Jy, and Jz 
            From Kim and Hassans 2020 paper on Biot-Savart integral                               '''
            h = 1/n
            #Fourier transform of the Biot-Savart kernel
            k = 2*np.pi*np.fft.fftfreq(Jx.shape[0],h)
            kx, ky, kz = np.meshgrid(k,k,k,indexing='ij',sparse=True)
            kx[0,0,0] = 1e-9
            K=1j/(kx**2+ky**2+kz**2)
            K[0,0,0]=0
            Jx_fft = np.fft.rfftn(Jx)
            Bzh = np.multiply(-(K*ky)[:,:,0:len(K)//2+1],Jx_fft)
            del Jx_fft
            Jy_fft = np.fft.rfftn(Jy)
            Bzh += np.multiply((K*kx)[:,:,0:len(K)//2+1],Jy_fft)
            del Jy_fft
            Bzh = np.fft.irfftn(Bzh)
            return Bzh
        
        def sigma_to_B(c,r,sigma_1,theta):
            self.r = r
            self.c = c
            self.sigma_1 = sigma_1
            self.theta = theta
            self.u, self.sigma = solver(self)
            self.Jx, self.Jy = J_field(self)
            self.Bz = FFTCONV(self.Jx, self.Jy, self.n)
            return self.Bz
        
        # generates a noisy Bz observation on a size 2^(m)+2 grid and interpolates to a 2^m grid, which is the size we 
        # use for the reconstruction, to avoid inverse crime
        def gen_obs_no_crime(c,r,sigma1, theta):
            n = 2**(self.m)+2                          # Data mesh number of gridpoints
            mesh = UnitCubeMesh(n, n, n)       # Data mesh
            V = FunctionSpace(mesh, "CG", 1)   #First order Continuous Galerkin (linear Lagrange element)
            V_grad = VectorFunctionSpace(mesh, "DG", 0)  # We use a vector function space since the gradient is 2D
            bcs = DirichletBC(V, Expression('x[0]',degree=1), "on_boundary")
            sigma_0 = 1.0   # conductivity of 1 outside ball
            sigma_1 = sigma1   # conductivity of sigma_1 inside ball
            class sigmafun(UserExpression):
                def eval(self, values, x):
                    if (((x[0]-c[0])*np.cos(theta)+(x[1]-c[1])*np.sin(theta))**2/r[0]**2\
                        +((x[1]-c[1])*np.cos(theta)-(x[0]-c[0])*np.sin(theta))**2/r[1]**2\
                        +(x[2]-c[2])**2/r[2]**2)<=1:
                        values[0] = sigma_1

                    else:
                        values[0] = sigma_0         
            # Initialize conductivity sigma         
            femel = V.ufl_element()
            sigma_tru = sigmafun(element=femel)
            sigma_true = project(sigma_tru, V, solver_type='bicgstab',preconditioner_type='sor')
            sigma = ConsSig(sigma_true)
            # Define functions and variational form
            u = TrialFunction(V)
            v = TestFunction(V)
            a = inner(sigma*grad(u),grad(v))*dx     # lhs of the weak formulation
            L = Constant("0.00")*v*dx
            # Solve the system
            u = Function(V)
            solve(a == L, u, bcs, solver_parameters = {'linear_solver':'bicgstab', 'preconditioner':'sor'})        
            # the solution now lives on the data space, we now interpolate it to reconstruction space
            sigma_f = Function(V)
            sigma_f = interpolate(sigma_true, V)
            J_val = project(- sigma_f*grad(u), V_grad, solver_type='bicgstab',preconditioner_type='sor')
            # interpolate to smaller reconstruction mesh to avoid inverse crime
            parameters["allow_extrapolation"] = True
            J_val = interpolate(J_val, self.V_grad).compute_vertex_values(self.mesh)
            Jx = J_val[0:(self.n+1)**3].reshape(self.n+1, self.n+1, self.n+1)
            Jy = J_val[(self.n+1)**3:2*(self.n+1)**3].reshape(self.n+1, self.n+1, self.n+1)
            Bz = FFTCONV(Jx, Jy, self.n)
            e_Bz = np.random.standard_normal(Bz.shape)
            e_Bz = e_Bz / np.linalg.norm(e_Bz)
            noise_lvl = 10
            sigma_noise = (np.linalg.norm(Bz) / 100)*noise_lvl     # 10% noise level
            y_obs = Bz + sigma_noise*e_Bz
            return y_obs, sigma_noise

        self.y_obs, self.sigma_noise = gen_obs_no_crime(c,r,sigma_1,theta)
        self.log_like = lambda c, r, sigma_1, theta: -0.5*np.linalg.norm(self.y_obs - sigma_to_B(c,r,sigma_1,theta))**2/self.sigma_noise**2
        def save_obs(self):
            np.save('noise_sigma', self.sigma_noise)
            np.save('y_obs', self.y_obs)

## make observation object
r = np.array([0.15, 0.3, 0.2])
c = np.random.uniform(np.max(r), 1-np.max(r), 3)
np.random.seed()
sigma_1 = 2
theta = 45*np.pi/180
obs = make_observation(c,r,sigma_1,theta)

# initialize sampler
theta0, sigma1_0, r0, c0 = np.array([0]), np.array([1.0]), np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])
beta = np.array([0.03, 0.04, 0.05, 0.04])
within_loop_size = 20 
burn_in = 0
sampler = Metropolis_in_Gibbs_burn_in(obs.log_like, c0, r0, sigma1_0, theta0, np.array( [[0,1],[0,1],[0,1]] ), np.array([0.05,0.5]), np.array([0,100]), np.array([0,np.pi]), beta, within_loop_size, burn_in)
sampler.sample(5000, 20)
samples_c, samples_r, samples_sigma1, samples_theta = sampler.give_stats()

if burn_in == 1:
	betas = np.array([sampler.beta1, sampler.beta2, sampler.beta3, sampler.beta4])
	print(betas)
	np.save('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/3D_samples_burnin_beta_values', betas)
else:
    np.save('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_c_beta_'+str(beta[0])+'_c0_'+str(c0[0])+'_'+str(c0[1])+'_'+str(c0[2]), samples_c)
    np.save('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_r_beta_'+str(beta[1])+'_r0_'+str(r0[0]), samples_r)
    np.save('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_sigma_beta_'+str(beta[2])+'_sigma0_'+str(sigma1_0[0]), samples_sigma1)
    np.save('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_theta_beta_'+str(beta[3])+'_theta0_'+str(theta0[0]), samples_theta)
