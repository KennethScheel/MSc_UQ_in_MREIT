import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#%% load samples
r0, c0 = np.array([0.1]), np.array([0.5, 0.5, 0.5])
beta = np.array([0.03, 0.04])

samples_c = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Samples/final_3D_samples_c_beta_'+str(beta[0])+'_c0_'+str(c0[0])+'_'+str(c0[1])+'_'+str(c0[2])+'.npy')
samples_r = np.load('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Samples/final_3D_samples_r_beta_'+str(beta[1])+'_r0_'+str(r0[0])+'.npy')

print(samples_c.shape)
print(samples_r.shape)
#%% print true parameters
np.random.seed(420)
r = np.random.uniform(0.1, 0.4)
c = np.random.uniform(r, 1-r, 3)
print(r)
print(c)

#%% applying burn-in
burn = 1000
# apply burn-in
samples_r_burnt = samples_r[burn:,:]
samples_c_burnt = samples_c[burn:,:]
r1_samples = samples_r_burnt[:,0]
cx_samples = samples_c_burnt[:,0]
cy_samples = samples_c_burnt[:,1]
cz_samples = samples_c_burnt[:,2]
print('r1 mean:  ', r1_samples.mean(), '    r1 true:  ', r)
print('cx mean: ', cx_samples.mean(axis=0), '     cx true: ', c[0])
print('cy mean: ', cy_samples.mean(axis=0), '    cy true: ', c[1])
print('cz mean: ', cz_samples.mean(axis=0), '    cz true: ', c[2])

# credibility intervals without thinning
percent = 95
lb = (100-percent)/2
up = 100-lb
lo_conf_r1, up_conf_r1 = np.percentile(r1_samples, [lb, up], axis=0)
lo_conf_cx, up_conf_cx = np.percentile(cx_samples, [lb, up], axis=0)
lo_conf_cy, up_conf_cy = np.percentile(cy_samples, [lb, up], axis=0)
lo_conf_cz, up_conf_cz = np.percentile(cz_samples, [lb, up], axis=0)
print('95% credibility interval r1: [' + str(lo_conf_r1) + ', ' + str(up_conf_r1)+']')
print('95% credibility interval cx: [' + str(lo_conf_cx) + ', ' + str(up_conf_cx)+']')
print('95% credibility interval cy: [' + str(lo_conf_cy) + ', ' + str(up_conf_cy)+']')
print('95% credibility interval cz: [' + str(lo_conf_cz) + ', ' + str(up_conf_cz)+']')
#%%

plt.figure()
# vertical lines for cred. intervals
plt.plot(np.ones((5)), np.linspace(lo_conf_r1, up_conf_r1,5), '-r', label='95% cred.', linewidth=3)
plt.plot(np.ones((5))*2, np.linspace(lo_conf_cx, up_conf_cx,5), '-r', linewidth=3)
plt.plot(np.ones((5))*3, np.linspace(lo_conf_cy, up_conf_cy,5), '-r', linewidth=3)
plt.plot(np.ones((5))*4, np.linspace(lo_conf_cz, up_conf_cz,5), '-r', linewidth=3)
# horizontal lines for cred. intervals
plt.plot(np.linspace(-0.1,0.1,5)+1, np.ones(5)*lo_conf_r1, '-r', linewidth=3)
plt.plot(np.linspace(-0.1,0.1,5)+1, np.ones(5)*up_conf_r1, '-r', linewidth=3)
plt.plot(np.linspace(-0.1,0.1,5)+2, np.ones(5)*lo_conf_cx, '-r', linewidth=3)
plt.plot(np.linspace(-0.1,0.1,5)+2, np.ones(5)*up_conf_cx, '-r', linewidth=3)
plt.plot(np.linspace(-0.1,0.1,5)+3, np.ones(5)*lo_conf_cy, '-r', linewidth=3)
plt.plot(np.linspace(-0.1,0.1,5)+3, np.ones(5)*up_conf_cy, '-r', linewidth=3)
plt.plot(np.linspace(-0.1,0.1,5)+4, np.ones(5)*lo_conf_cz, '-r', linewidth=3)
plt.plot(np.linspace(-0.1,0.1,5)+4, np.ones(5)*up_conf_cz, '-r', linewidth=3)
# true values and estimates
plt.plot(np.array([1,2,3,4]), np.array([r,c[0],c[1],c[2]]), 'xk', label='true', markersize=10)
plt.scatter(np.array([1,2,3,4]), (np.array([r1_samples.mean(),cx_samples.mean(axis=0)\
                                            ,cy_samples.mean(axis=0), cz_samples.mean(axis=0)])), 60,facecolors='none', edgecolors='k',label='estimate')
# layout
plt.xticks(np.array([1,2,3,4]), ['$r$', '$c_x$', '$c_y$', '$c_z$'], fontsize=20)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f cm'))
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.ylim([0.1,0.6])
plt.title('parameter UQ', fontsize=21)
plt.savefig('/zhome/ad/7/127239/Desktop/Kandidatspeciale/Toy_problem_one_ball/Data/Figures/toy1_3D_parameter_UQ.png',bbox_inches='tight')






