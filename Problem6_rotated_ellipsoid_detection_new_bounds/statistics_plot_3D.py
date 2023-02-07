import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#%% load samples
theta0, sigma1_0, r0, c0 = np.array([0]), np.array([1.0]), np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])
#theta0, sigma1_0, r0, c0 = np.array([0]), np.array([1.0]), np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])
beta = np.array([0.03, 0.04, 0.05, 0.04])

samples_c = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/final_3D_samples_c_beta_'+str(beta[0])+'_c0_'+str(c0[0])+'_'+str(c0[1])+'_'+str(c0[2])+'.npy')
samples_r = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/final_3D_samples_r_beta_'+str(beta[1])+'_r0_'+str(r0[0])+'.npy')
samples_sigma = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/final_3D_samples_sigma_beta_'+str(beta[2])+'_sigma0_'+str(sigma1_0[0])+'.npy')
samples_theta = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/final_3D_samples_theta_beta_'+str(beta[3])+'_theta0_'+str(theta0[0])+'.npy')

print(samples_c.shape)
print(samples_r.shape)
print(samples_sigma.shape)
print(samples_theta.shape)
#%% print true parameters
np.random.seed(420)
r = np.array([0.15, 0.3, 0.2])
c = np.random.uniform(np.max(r), 1-np.max(r), 3)
sigma1 = 2
theta = 45*np.pi/180
print(r)
print(c)
print(sigma1)
print(theta)
#%% applying burn-in
burn = 100
# apply burn-in
samples_r_burnt = samples_r[burn:,:]
samples_c_burnt = samples_c[burn:,:]
samples_sigma_burnt = samples_sigma[burn:]
samples_theta_burnt = samples_theta[burn:]
r1_samples = samples_r_burnt[:,0]
r2_samples = samples_r_burnt[:,1]
r3_samples = samples_r_burnt[:,2]
cx_samples = samples_c_burnt[:,0]
cy_samples = samples_c_burnt[:,1]
cz_samples = samples_c_burnt[:,2]
sigma_samples = samples_sigma_burnt
theta_samples = samples_theta_burnt
print('r1 mean:  ', r1_samples.mean(), '    r1 true:  ', r[0])
print('r2 mean:  ', r2_samples.mean(), '    r2 true:  ', r[1])
print('r3 mean:  ', r3_samples.mean(), '    r3 true:  ', r[2])
print('cx mean: ', cx_samples.mean(axis=0), '     cx true: ', c[0])
print('cy mean: ', cy_samples.mean(axis=0), '    cy true: ', c[1])
print('cz mean: ', cz_samples.mean(axis=0), '    cz true: ', c[2])
print('sigma mean: ', sigma_samples.mean(), '  sigma true: ', sigma1)
print('theta mean: ', theta_samples.mean(), '  theta true: ', theta)
# credibility intervals without thinning
percent = 95
lb = (100-percent)/2
up = 100-lb
lo_conf_r1, up_conf_r1 = np.percentile(r1_samples, [lb, up], axis=0)
lo_conf_r2, up_conf_r2 = np.percentile(r2_samples, [lb, up], axis=0)
lo_conf_r3, up_conf_r3 = np.percentile(r3_samples, [lb, up], axis=0)
lo_conf_cx, up_conf_cx = np.percentile(cx_samples, [lb, up], axis=0)
lo_conf_cy, up_conf_cy = np.percentile(cy_samples, [lb, up], axis=0)
lo_conf_cz, up_conf_cz = np.percentile(cz_samples, [lb, up], axis=0)
lo_conf_sigma, up_conf_sigma = np.percentile(sigma_samples, [lb, up], axis=0)
lo_conf_theta, up_conf_theta = np.percentile(theta_samples, [lb, up], axis=0)
print('95% credibility interval r1: [' + str(lo_conf_r1) + ', ' + str(up_conf_r1)+']')
print('95% credibility interval r2: [' + str(lo_conf_r2) + ', ' + str(up_conf_r2)+']')
print('95% credibility interval r3: [' + str(lo_conf_r3) + ', ' + str(up_conf_r3)+']')
print('95% credibility interval cx: [' + str(lo_conf_cx) + ', ' + str(up_conf_cx)+']')
print('95% credibility interval cy: [' + str(lo_conf_cy) + ', ' + str(up_conf_cy)+']')
print('95% credibility interval cz: [' + str(lo_conf_cz) + ', ' + str(up_conf_cz)+']')
print('95% credibility interval sigma: [' + str(lo_conf_sigma[0]) + ', ' + str(up_conf_sigma[0])+']')
print('95% credibility interval theta: [' + str(lo_conf_theta[0]) + ', ' + str(up_conf_theta[0])+']')

#%%

plt.figure()
# vertical lines for cred. intervals
plt.semilogy(np.ones((5)), np.linspace(lo_conf_r1, up_conf_r1,5), '-r', label='95% cred.', linewidth=3)
plt.semilogy(np.ones((5))*2, np.linspace(lo_conf_r2, up_conf_r2,5), '-r', linewidth=3)
plt.semilogy(np.ones((5))*3, np.linspace(lo_conf_r3, up_conf_r3,5), '-r', linewidth=3)
plt.semilogy(np.ones((5))*4, np.linspace(lo_conf_cx, up_conf_cx,5), '-r', linewidth=3)
plt.semilogy(np.ones((5))*5, np.linspace(lo_conf_cy, up_conf_cy,5), '-r', linewidth=3)
plt.semilogy(np.ones((5))*6, np.linspace(lo_conf_cz, up_conf_cz,5), '-r', linewidth=3)
plt.semilogy(np.ones((5))*7, np.linspace(lo_conf_sigma, up_conf_sigma,5), '-r', linewidth=3)
plt.semilogy(np.ones((5))*8, np.linspace(lo_conf_theta, up_conf_theta,5), '-r', linewidth=3)   
# horizontal lines for cred. intervals
plt.semilogy(np.linspace(-0.1,0.1,5)+1, np.ones(5)*lo_conf_r1, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+1, np.ones(5)*up_conf_r1, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+2, np.ones(5)*lo_conf_r2, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+2, np.ones(5)*up_conf_r2, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+3, np.ones(5)*lo_conf_r3, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+3, np.ones(5)*up_conf_r3, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+4, np.ones(5)*lo_conf_cx, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+4, np.ones(5)*up_conf_cx, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+5, np.ones(5)*lo_conf_cy, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+5, np.ones(5)*up_conf_cy, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+6, np.ones(5)*lo_conf_cz, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+6, np.ones(5)*up_conf_cz, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+7, np.ones(5)*lo_conf_sigma, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+7, np.ones(5)*up_conf_sigma, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+8, np.ones(5)*lo_conf_theta, '-r', linewidth=3)
plt.semilogy(np.linspace(-0.1,0.1,5)+8, np.ones(5)*up_conf_theta, '-r', linewidth=3)
# true values and estimates
plt.semilogy(np.array([1,2,3,4,5,6,7,8]), np.array([r[0],r[1],r[2],c[0],c[1],c[2],sigma1,theta]), 'xk', label='true', markersize=10)
plt.scatter(np.array([1,2,3,4,5,6,7,8]), (np.array([r1_samples.mean(),r2_samples.mean(), r3_samples.mean(),cx_samples.mean(axis=0)\
                                            ,cy_samples.mean(axis=0), cz_samples.mean(axis=0), sigma_samples.mean(),theta_samples.mean()])), 60,facecolors='none', edgecolors='k',label='estimate')
# layout
plt.xticks(np.array([1,2,3,4,5,6,7,8]), ['$r_x$', '$r_y$', '$r_z$', '$c_x$', '$c_y$', '$c_z$', '$\kappa$', '$\delta$'], fontsize=20)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f cm'))
plt.yticks(fontsize=20)
plt.legend(fontsize=16)
plt.ylim([0.05,6])
plt.title('parameter UQ', fontsize=21)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/Problem62_3D_parameter_UQ.png',bbox_inches='tight')







