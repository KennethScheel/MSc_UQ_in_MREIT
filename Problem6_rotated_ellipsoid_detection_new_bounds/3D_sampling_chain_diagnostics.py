import numpy as np
import matplotlib.pyplot as plt
from conv_checks import Geweke, iact 
from mpl_toolkits.axes_grid1 import ImageGrid
#%% load samples
theta0, sigma1_0, r0, c0 = np.array([0]), np.array([1.0]), np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])
beta = np.array([0.03, 0.04, 0.05, 0.04])

samples_c = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_c_beta_'+str(beta[0])+'_c0_'+str(c0[0])+'_'+str(c0[1])+'_'+str(c0[2])+'.npy')
samples_r = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_r_beta_'+str(beta[1])+'_r0_'+str(r0[0])+'.npy')
samples_sigma = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_sigma_beta_'+str(beta[2])+'_sigma0_'+str(sigma1_0[0])+'.npy')
samples_theta = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Samples/abstract_3D_samples_theta_beta_'+str(beta[3])+'_theta0_'+str(theta0[0])+'.npy')

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


#%% plot chains
plt.figure()
plt.plot(samples_r[:,0])
plt.title('$r_x$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_rx.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_r[:,1])
plt.title('$r_y$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_ry.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_r[:,2])
plt.title('$r_z$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_rz.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_c[:,0])
plt.title('$c_x$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_cx.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_c[:,1])
plt.title('$c_y$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_cy.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_c[:,2])
plt.title('$c_z$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_cz.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_sigma)
plt.title('$\kappa$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(S/cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_sigma.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_theta)
plt.title('$\delta$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(rad)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_trace_theta.png',bbox_inches='tight')
plt.show()

#%% applying burn-in and thinning
burn = 1000
# apply burn-in
samples_r_burnt = samples_r[burn:]
samples_c_burnt = samples_c[burn:,:]
samples_sigma_burnt = samples_sigma[burn:]
samples_theta_burnt = samples_theta[burn:]
# apply thinning according to the iact
r_samples = samples_r_burnt
cx_samples = samples_c_burnt[:,0]
cy_samples = samples_c_burnt[:,1]
cz_samples = samples_c_burnt[:,2]
sigma_samples = samples_sigma_burnt
theta_samples = samples_theta_burnt
#%% chain diagnostics
iact_r = iact(samples_r_burnt)
iact_c = iact(samples_c_burnt)
iact_sigma = iact(samples_sigma_burnt)[0]
iact_theta = iact(samples_theta_burnt)[0]
neff_r = (samples_r_burnt.shape[0]/iact_r)
neff_c = np.array([(samples_c_burnt.shape[0]/iact_c[0]), (samples_c_burnt.shape[0]/iact_c[1]), (samples_c_burnt.shape[0]/iact_c[2])])
neff_sigma = (samples_sigma_burnt.shape[0]/iact_sigma)
neff_theta = (samples_theta_burnt.shape[0]/iact_theta)
print('radius iact: ' + str(iact_r) ) 
print('center iact: ' + str(iact_c) ) 
print('sigma  iact: ' + str(iact_sigma))
print('theta iact: ' + str(iact_theta))
print('radius n_eff: ' + str(neff_r) ) 
print('center n_eff: ' + str(neff_c) ) 
print('sigma  n_eff: ' + str(neff_sigma))
print('theta n_eff: ' + str(neff_theta))
#%% credibility intervals
percent = 95
lb = (100-percent)/2
up = 100-lb
lo_conf_r, up_conf_r = np.percentile(r_samples, [lb, up], axis=0)
lo_conf_cx, up_conf_cx = np.percentile(cx_samples, [lb, up], axis=0)
lo_conf_cy, up_conf_cy = np.percentile(cy_samples, [lb, up], axis=0)
lo_conf_cz, up_conf_cz = np.percentile(cz_samples, [lb, up], axis=0)
lo_conf_sigma, up_conf_sigma = np.percentile(sigma_samples, [lb, up], axis=0)
lo_conf_theta, up_conf_theta = np.percentile(theta_samples, [lb, up], axis=0)
print('95% credibility interval r: [' + str(lo_conf_r[0]) + ', ' + str(up_conf_r[0])+']')
print('95% credibility interval cx: [' + str(lo_conf_cx) + ', ' + str(up_conf_cx)+']')
print('95% credibility interval cy: [' + str(lo_conf_cy) + ', ' + str(up_conf_cy)+']')
print('95% credibility interval cz: [' + str(lo_conf_cz) + ', ' + str(up_conf_cz)+']')
print('95% credibility interval sigma: [' + str(lo_conf_sigma[0]) + ', ' + str(up_conf_sigma[0])+']')
print('95% credibility interval theta: [' + str(lo_conf_theta[0]) + ', ' + str(up_conf_theta[0])+']')
#%% sample means after modifying chains
print('r mean:  ', r_samples.mean(axis=0), '   r true:  ', r)
print('cx mean: ', cx_samples.mean(), '    cx true: ', c[0])
print('cy mean: ', cy_samples.mean(), '   cy true: ', c[1])
print('cz mean: ', cz_samples.mean(), '   cz true: ', c[2])
print('sigma mean: ', sigma_samples.mean(), '  sigma true: ', sigma1)
print('theta mean: ', theta_samples.mean(), '  theta true: ', theta)
#%% visualize the uncertainty in an image
imsize = 64
MM = samples_c_burnt.shape[0]
ims = np.ones((imsize,imsize,imsize,MM), dtype='float16')
x,y,z = np.meshgrid(np.linspace(0, 1, imsize), np.linspace(0, 1, imsize), np.linspace(0, 1, imsize))

for i in range(MM):
    ims[(((x-samples_c_burnt[i,0])*np.cos(samples_theta_burnt[i])+(y-samples_c_burnt[i,1])*np.sin(samples_theta_burnt[i]))**2/samples_r_burnt[i,0]**2\
        +((y-samples_c_burnt[i,1])*np.cos(samples_theta_burnt[i])-(x-samples_c_burnt[i,0])*np.sin(samples_theta_burnt[i]))**2/samples_r_burnt[i,1]**2\
            + (z-samples_c_burnt[i,2])**2/samples_r_burnt[i,2]**2) <=1,i] = samples_sigma_burnt[i]

k = 5

plt.figure()
plt.imshow(ims.mean(axis=3)[:,:,k], origin='lower')
plt.title('sample mean for slice ' + str(k) + '/' + str(16))
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(ims.std(axis=3)[:,:,k], origin='lower')
plt.title('standard deviation for slice ' + str(k) + '/' + str(16))
plt.colorbar()
plt.show()

im_tru = np.ones((imsize,imsize,imsize))
img = np.ones((imsize,imsize,imsize))
im_tru[(((x-c[0])*np.cos(theta)+(y-c[1])*np.sin(theta))**2/r[0]**2\
    +((y-c[1])*np.cos(theta)-(x-c[0])*np.sin(theta))**2/r[1]**2)\
       +(z-c[2])**2/r[2]**2 <=1] = sigma1

img[(((x-samples_c_burnt.mean(axis=0)[0])*np.cos(samples_theta_burnt.mean(axis=0)[0])+(y-samples_c_burnt.mean(axis=0)[1])*np.sin(samples_theta_burnt.mean(axis=0)[0]))**2/samples_r_burnt.mean(axis=0)[0]**2\
    +((y-samples_c_burnt.mean(axis=0)[1])*np.cos(samples_theta_burnt.mean(axis=0)[0])-(x-samples_c_burnt.mean(axis=0)[0])*np.sin(samples_theta_burnt.mean(axis=0)[0]))**2/samples_r_burnt.mean(axis=0)[1]**2\
        + (z-samples_c_burnt.mean(axis=0)[2])**2/samples_r_burnt.mean(axis=0)[2]**2)<=1] = samples_sigma_burnt.mean(axis=0)[0]


k = 30
plt.figure()
plt.imshow(im_tru[:,:,k], origin='lower')
plt.title('true sigma for slice ' + str(k) + '/' + str(imsize))
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(img[:,:,k], origin='lower')
plt.title('mean sigma for slice ' + str(k) + '/' + str(imsize))
plt.colorbar()
plt.show()

#%%
m = imsize
fig, axes = plt.subplots(2,2, figsize=(15,10))
im = axes[0,0].imshow(im_tru[:,:,k], origin='lower', vmin=1, vmax=2.5)
axes[0,0].axis('off')
axes[0,0].set_title('ground truth $\sigma(c,r,\kappa,\delta)$ slice ' + str(k) + '/' + str(m), fontsize=24)
cbar=plt.colorbar(im, ax=axes[0,0], shrink=1)
cbar.ax.tick_params(labelsize=24)
im = axes[0,1].imshow(img[:,:,k], origin='lower', vmin=1, vmax=2.5)
axes[0,1].axis('off')
axes[0,1].set_title('$\sigma(\mathbb{E}[c_k],\mathbb{E}[r_k],\mathbb{E}[\kappa_k],\mathbb{E}[\delta_k])$ slice ' + str(k) + '/' + str(m), fontsize=24)
cbar=plt.colorbar(im, ax=axes[0,1], shrink=1)
cbar.ax.tick_params(labelsize=24)
im = axes[1,0].imshow(ims.mean(axis=3)[:,:,k], origin='lower', vmin=1, vmax=2.5)
axes[1,0].axis('off')
axes[1,0].set_title('$\mathbb{E}[\sigma(c_k,r_k,\kappa_k,\delta_k)]$ slice ' + str(k) + '/' + str(m), fontsize=24)
cbar=plt.colorbar(im, ax=axes[1,0], shrink=1)
cbar.ax.tick_params(labelsize=24)
im = axes[1,1].imshow(ims.std(axis=3)[:,:,k], origin='lower', vmin=0, vmax=1.0)
axes[1,1].axis('off')
axes[1,1].set_title('$\sqrt{\mathbb{V} [\sigma(c_k,r_k,\kappa_k,\delta_k)]}$ slice ' + str(k) + '/' + str(m), fontsize=24)
plt.axis('off')
cbar=plt.colorbar(im, ax=axes[1,1], shrink=1)
cbar.ax.tick_params(labelsize=24)
plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Problem6_rotated_ellipsoid_detection_new_bounds/Data/Figures/problem62_3D_individual_colorbar_slice_' + str(k) + '.png',bbox_inches='tight')
