import numpy as np
from fenics import *
#%% load samples
theta0, sigma1_0, r0, c0 = np.array([0]), np.array([1.0]), np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])
beta = np.array([0.03, 0.04, 0.05, 0.04])

samples_c = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_c_beta_'+str(beta[0])+'_c0_'+str(c0[0])+'_'+str(c0[1])+'_'+str(c0[2])+'.npy')
samples_r = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_r_beta_'+str(beta[1])+'_r0_'+str(r0[0])+'.npy')
samples_sigma = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_sigma_beta_'+str(beta[2])+'_sigma0_'+str(sigma1_0[0])+'.npy')
samples_theta = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_theta_beta_'+str(beta[3])+'_theta0_'+str(theta0[0])+'.npy')

#%% print true parameters
np.random.seed(420)
r = np.array([0.15, 0.3, 0.2])
c = np.random.uniform(np.max(r), 1-np.max(r), 3)
sigma1 = 2
theta = 45*np.pi/180

#%% applying burn-in and thinning
burn = 1000
# apply burn-in
samples_r_burnt = samples_r[burn:]
samples_c_burnt = samples_c[burn:,:]
samples_sigma_burnt = samples_sigma[burn:]
samples_theta_burnt = samples_theta[burn:]

#%% visualize the uncertainty in an image
imsize = 65
MM = samples_c_burnt.shape[0]
ims = np.ones((imsize,imsize,imsize,MM), dtype='float16')
x, y, z = np.meshgrid(np.linspace(0, 1, imsize), np.linspace(
    0, 1, imsize), np.linspace(0, 1, imsize))

for i in range(MM):
    ims[(((x-samples_c_burnt[i,0])*np.cos(samples_theta_burnt[i])+(y-samples_c_burnt[i,1])*np.sin(samples_theta_burnt[i]))**2/samples_r_burnt[i,0]**2\
        +((y-samples_c_burnt[i,1])*np.cos(samples_theta_burnt[i])-(x-samples_c_burnt[i,0])*np.sin(samples_theta_burnt[i]))**2/samples_r_burnt[i,1]**2\
            + (z-samples_c_burnt[i,2])**2/samples_r_burnt[i,2]**2) <=1,i] = samples_sigma_burnt[i]

im_tru = np.ones((imsize,imsize,imsize))
img = np.ones((imsize,imsize,imsize))
im_tru[(((x-c[0])*np.cos(theta)+(y-c[1])*np.sin(theta))**2/r[0]**2\
    +((y-c[1])*np.cos(theta)-(x-c[0])*np.sin(theta) )**2/r[1]**2)\
       +(z-c[2])**2/r[2]**2 <=1] = sigma1

img[(((x-samples_c_burnt.mean(axis=0)[0])*np.cos(samples_theta_burnt.mean(axis=0)[0])+(y-samples_c_burnt.mean(axis=0)[1])*np.sin(samples_theta_burnt.mean(axis=0)[0]))**2/samples_r_burnt.mean(axis=0)[0]**2\
    +((y-samples_c_burnt.mean(axis=0)[1])*np.cos(samples_theta_burnt.mean(axis=0)[0])-(x-samples_c_burnt.mean(axis=0)[0])*np.sin(samples_theta_burnt.mean(axis=0)[0]))**2/samples_r_burnt.mean(axis=0)[1]**2\
        + (z-samples_c_burnt.mean(axis=0)[2])**2/samples_r_burnt.mean(axis=0)[2]**2)<=1] = samples_sigma_burnt.mean(axis=0)[0]

#%% saves data to .vtu files for visualization in Paraview
folder = '/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/'
n = 64
mesh = UnitCubeMesh(n,n,n)
V = FunctionSpace(mesh, 'CG', 1)

parameters["reorder_dofs_serial"] = False


sigma_true = Function(V)
sigma_true.vector().set_local(im_tru.ravel())
sigma_true_file = project(sigma_true, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'abstract_problem62_3D_sigma_true.pvd')<<sigma_true_file

sigma_param_mean = Function(V)
sigma_param_mean.vector().set_local(img.ravel())
sigma_param_mean_file = project(sigma_param_mean, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'abstract_problem62_3D_sigma_parameter_mean.pvd')<<sigma_param_mean

sigma_push_mean = Function(V)
sigma_push_mean.vector().set_local(ims.mean(axis=3).ravel())
sigma_push_mean_file = project(sigma_push_mean, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'abstract_problem62_3D_sigma_pushforward_mean.pvd')<<sigma_push_mean_file

sigma_push_std = Function(V)
sigma_push_std.vector().set_local(ims.std(axis=3).ravel())
sigma_push_std_file = project(sigma_push_std, V, solver_type='bicgstab', preconditioner_type='sor')
File(folder + 'abstract_problem62_3D_sigma_pushforward_std.pvd')<<sigma_push_std_file



