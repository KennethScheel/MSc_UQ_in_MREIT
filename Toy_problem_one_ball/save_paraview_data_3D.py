import numpy as np
from fenics import *
parameters["reorder_dofs_serial"] = False

#%% load samples
r0, c0 = np.array([0.1]), np.array([0.5, 0.5, 0.5])
beta = np.array([0.03, 0.04])

samples_c = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Samples/final_3D_samples_c_beta_'+str(beta[0])+'_c0_'+str(c0[0])+'_'+str(c0[1])+'_'+str(c0[2])+'.npy')
samples_r = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Samples/final_3D_samples_r_beta_'+str(beta[1])+'_r0_'+str(r0[0])+'.npy')

#%% print true parameters
np.random.seed(420)
r = np.random.uniform(0.1, 0.4)
c = np.random.uniform(r, 1-r, 3)

#%% applying burn-in and thinning
burn = 1000
# apply burn-in
samples_r_burnt = samples_r[burn:]
samples_c_burnt = samples_c[burn:,:]

#%% visualize the uncertainty in an image
m = 64
MM = samples_c.shape[0]-burn
ims = np.ones((m,m,m,MM), dtype='float32')
x,y,z = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m), np.linspace(0, 1, m))

for i in range(MM):
    ims[(x-samples_c_burnt[i,0])**2 + (y-samples_c_burnt[i,1])**2 + (z-samples_c_burnt[i,2])**2 <= samples_r_burnt[i]**2,i] = 2

img = np.ones((m,m,m))
im_tru = np.ones((m,m,m))

img[(x-samples_c_burnt.mean(axis=0)[0])**2 + (y-samples_c_burnt.mean(axis=0)[1])**2 + (z-samples_c_burnt.mean(axis=0)[2])**2 <= samples_r_burnt.mean(axis=0)**2+1e-16] = 2
im_tru[(x-c[0])**2+ (y-c[1])**2 + (z-c[2])**2 <= r**2 +1e-16] = 2

#%% saves data to .vtu files for visualization in Paraview
folder = '/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Figures/'
n = m
mesh = UnitCubeMesh(n,n,n)
V = FunctionSpace(mesh, 'CG', 1)


sigma_true = Function(V)
sigma_true.vector().set_local(im_tru.ravel())
sigma_true_file = project(sigma_true, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'toy1_3D_sigma_true.pvd')<<sigma_true_file

sigma_param_mean = Function(V)
sigma_param_mean.vector().set_local(img.ravel())
sigma_param_mean_file = project(sigma_param_mean, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'toy1_3D_sigma_parameter_mean.pvd')<<sigma_param_mean

sigma_push_mean = Function(V)
sigma_push_mean.vector().set_local(ims.mean(axis=3).ravel())
sigma_push_mean_file = project(sigma_push_mean, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'toy1_3D_sigma_pushforward_mean.pvd')<<sigma_push_mean_file

sigma_push_std = Function(V)
sigma_push_std.vector().set_local(ims.std(axis=3).ravel())
sigma_push_std_file = project(sigma_push_std, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'toy1_3D_sigma_pushforward_std.pvd')<<sigma_push_std_file



