import numpy as np
import matplotlib.pyplot as plt
from conv_checks import Geweke, iact 
from mpl_toolkits.axes_grid1 import ImageGrid
#%% load samples
r0, c0 = np.array([0.1]), np.array([0.8, 0.1])
beta = np.array([0.03, 0.06])

samples_c = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Samples/final_2D_samples_c_beta_'+str(beta[0])+'_c0_'+str(c0[0])+'_'+str(c0[1])+'.npy')
samples_r = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Samples/final_2D_samples_r_beta_'+str(beta[1])+'_r0_'+str(r0[0])+'.npy')

print(samples_c.shape)
print(samples_r.shape)

#%% print true parameters
np.random.seed(420)
r = np.random.uniform(0.1, 0.4)
c = np.random.uniform(r, 1-r, 2)
print(r)
print(c)

#%% plot chains
plt.figure()
plt.plot(samples_r)
plt.title('$r$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Figures/toy1_2D_trace_r.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_c[:,0])
plt.title('$c_x$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Figures/toy1_2D_trace_cx.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(samples_c[:,1])
plt.title('$c_y$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('(cm)', fontsize=16)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Figures/toy1_2D_trace_cy.png',bbox_inches='tight')
plt.show()

#%% applying burn-in and thinning
burn = 1000
# apply burn-in
samples_r_burnt = samples_r[burn:]
samples_c_burnt = samples_c[burn:,:]
# apply thinning according to the iact
r_samples = samples_r_burnt
cx_samples = samples_c_burnt[:,0]
cy_samples = samples_c_burnt[:,1]

#%% MCMC chain diagnostics
iact_r = iact(samples_r_burnt)[0]
iact_c = iact(samples_c_burnt)
neff_r = (samples_r_burnt.shape[0]/iact_r)
neff_c = np.array([(samples_c_burnt.shape[0]/iact_c[0]), (samples_c_burnt.shape[0]/iact_c[1])])
print('radius iact: ' + str(iact_r) ) 
print('center iact: ' + str(iact_c) ) 
print('radius n_eff: ' + str(neff_r) ) 
print('center n_eff: ' + str(neff_c) ) 

#%% credibility intervals
percent = 95
lb = (100-percent)/2
up = 100-lb
lo_conf_r, up_conf_r = np.percentile(r_samples, [lb, up], axis=0)
lo_conf_cx, up_conf_cx = np.percentile(cx_samples, [lb, up], axis=0)
lo_conf_cy, up_conf_cy = np.percentile(cy_samples, [lb, up], axis=0)
print('95% credibility interval r: [' + str(lo_conf_r[0]) + ', ' + str(up_conf_r[0])+']')
print('95% credibility interval cx: [' + str(lo_conf_cx) + ', ' + str(up_conf_cx)+']')
print('95% credibility interval cy: [' + str(lo_conf_cy) + ', ' + str(up_conf_cy)+']')

#%% sample means after modifying chains
print('r mean:  ', r_samples.mean(), '    r true:  ', r)
print('cx mean: ', cx_samples.mean(), '    cx true: ', c[0])
print('cy mean: ', cy_samples.mean(), '    cy true: ', c[1])

#%% visualize the uncertainty in an image
MM = samples_c.shape[0] - burn

ims = np.ones((64,64,MM))
x,y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))

for i in range(MM):
    ims[(x-samples_c_burnt[i,0])**2 + (y-samples_c_burnt[i,1])**2 <= samples_r_burnt[i]**2,i] = 2

plt.figure()
plt.imshow(ims.mean(axis=2), origin='lower')
plt.title('sample mean')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(ims.std(axis=2), origin='lower')
plt.title('standard deviation')
plt.colorbar()
plt.show()

img = np.ones((64,64))
im_tru = np.ones((64,64))

img[(x-samples_c_burnt.mean(axis=0)[0])**2 + (y-samples_c_burnt.mean(axis=0)[1])**2 <= samples_r_burnt.mean(axis=0)**2+1e-16] = 2
im_tru[(x-c[0])**2+ (y-c[1])**2 <= r**2 +1e-16] = 2

plt.figure()
plt.imshow(im_tru, origin='lower')
plt.title('true data')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(img, origin='lower')
plt.title('parameter sample mean')
plt.colorbar()
plt.show()

#%% plots for slides and thesis

fig, axes = plt.subplots(2,2, figsize=(15,10))
im = axes[0,0].imshow(im_tru, origin='lower', vmin=1, vmax=2)
axes[0,0].axis('off')
axes[0,0].set_title('ground truth $\sigma(c,r)$', fontsize=24)
cbar=plt.colorbar(im, ax=axes[0,0], shrink=1)
cbar.ax.tick_params(labelsize=24)
cbar.ax.set_ylabel('(S/cm)', fontsize=24)

im = axes[0,1].imshow(img, origin='lower', vmin=1, vmax=2)
axes[0,1].axis('off')
axes[0,1].set_title('$\sigma(\mathbb{E}[c_k],\mathbb{E}[r_k])$', fontsize=24)
cbar=plt.colorbar(im, ax=axes[0,1], shrink=1)
cbar.ax.tick_params(labelsize=24)
cbar.ax.set_ylabel('(S/cm)', fontsize=24)

im = axes[1,0].imshow(ims.mean(axis=2), origin='lower', vmin=1, vmax=2)
axes[1,0].axis('off')
axes[1,0].set_title('$\mathbb{E}[\sigma(c_k,r_k)]$', fontsize=24)
cbar=plt.colorbar(im, ax=axes[1,0], shrink=1)
cbar.ax.tick_params(labelsize=24)
cbar.ax.set_ylabel('(S/cm)', fontsize=24)

im = axes[1,1].imshow(ims.std(axis=2), origin='lower', vmin=0, vmax=0.5)
axes[1,1].axis('off')
axes[1,1].set_title('$\sqrt{\mathbb{V} [\sigma(c_k,r_k)]}$', fontsize=24)
plt.axis('off')
cbar=plt.colorbar(im, ax=axes[1,1], shrink=1)
cbar.ax.tick_params(labelsize=24)
cbar.ax.set_ylabel('(S/cm)', fontsize=24)

#plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Figures/toy1_2D_individual_colorbar_box.png',bbox_inches='tight')
