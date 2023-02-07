import scipy.stats as scp
import numpy as np

# Metropolis-within-Gibbs sampler for center, radius and conductivity value of a single ball

class Metropolis_in_Gibbs_burn_in:
    def __init__(self, pi_like, x01, x02, x03, x04, bounds1, bounds2, bounds3, bounds4, beta, within_loop_size, burn_in_flag):
        self.pi_like = pi_like
        self.x01 = x01
        self.x02 = x02
        self.x03 = x03
        self.x04 = x04
        self.ndim1 = self.x01.shape[0]
        self.ndim2 = self.x02.shape[0]
        self.ndim3 = self.x03.shape[0]
        self.ndim4 = self.x04.shape[0]
        self.Ndim = bounds1.shape[0]

        self.target_last1 = pi_like(x01,x02,x03,x04)
        self.target_last2 = pi_like(x01,x02,x03,x04)
        self.target_last3 = pi_like(x01,x02,x03,x04)
        self.target_last4 = pi_like(x01,x02,x03,x04)

        self.sample_last1 = self.x01
        self.sample_last2 = self.x02
        self.sample_last3 = self.x03
        self.sample_last4 = self.x04

        self.all_samples1 = [ self.x01 ]
        self.all_targets1 = [ self.target_last1 ]
        self.all_accepts1 = [ 1 ]

        self.all_samples2 = [ self.x02 ]
        self.all_targets2 = [ self.target_last2 ]
        self.all_accepts2 = [ 1 ]
        
        self.all_samples3 = [ self.x03 ]
        self.all_targets3 = [ self.target_last3 ]
        self.all_accepts3 = [ 1 ]
        
        self.all_samples4 = [ self.x04 ]
        self.all_targets4 = [ self.target_last4 ]
        self.all_accepts4 = [ 1 ]

        pi1 = lambda c: self.pi_like( c, self.sample_last2, self.sample_last3, self.sample_last4)
        if self.Ndim == 2:
          constraint1 = lambda c: (c[0]>=bounds1[0,0])and(c[0]<=bounds1[0,1])and(c[1]>=bounds1[1,0])and(c[1]<=bounds1[1,1]) 
        elif self.Ndim == 3:
          constraint1 = lambda c: (c[0]>=bounds1[0,0])and(c[0]<=bounds1[0,1])and(c[1]>=bounds1[1,0])and(c[1]<=bounds1[1,1])and(c[2]>=bounds1[2,0])and(c[2]<=bounds1[2,1]) 
        self.update1 = Metropolis_update( pi1, self.sample_last1, constraint1)

        pi2 = lambda r: self.pi_like( self.sample_last1, r, self.sample_last3, self.sample_last4)
        constraint = lambda r: (r[0] < r[1]) and np.array([(r[i]>=bounds2[0])and(r[i]<=bounds2[1]) for i in range(self.ndim2)], dtype=bool).all()
        self.update2 = Metropolis_update( pi2, self.sample_last2, constraint)
        
        pi3 = lambda sigma1: self.pi_like( self.sample_last1, self.sample_last2, sigma1, self.sample_last4)
        constraint2 = lambda sigma1: np.array([(sigma1[i]>=bounds3[0])and(sigma1[i]<=bounds3[1]) for i in range(self.ndim3)], dtype=bool).all()
        self.update3 = Metropolis_update( pi3, self.sample_last3, constraint2)
        
        pi4 = lambda theta: self.pi_like( self.sample_last1, self.sample_last2, self.sample_last3, theta)
        constraint3 = lambda theta: np.array([(theta[i]>=bounds4[0])and(theta[i]<=bounds4[1]) for i in range(self.ndim4)], dtype=bool).all()
        self.update4 = Metropolis_update( pi4, self.sample_last4, constraint3)

        self.in_loop_size1 = within_loop_size        # size of the within-Gibbs Metropolis loop (use larger size for higher quality samples)
        self.in_loop_size2 = within_loop_size
        self.in_loop_size3 = within_loop_size
        self.in_loop_size4 = within_loop_size

        self.star_acc = 0.234

        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.beta3 = beta[2]
        self.beta4 = beta[3]
        
        self.burn_in_flag = burn_in_flag

    def sample(self, Ns, Nb):
        if self.burn_in_flag:
            print(self.beta1)
            print(self.beta2)
            print(self.beta3)
            print(self.beta4)
            print('burn-in period:')
            for s in range(Nb):
                self.burn_in_period()
            
        print(self.beta1)
        print(self.beta2)
        print(self.beta3)
        print(self.beta4)

        print('sampling period:')
        for s in range(Ns-1):
            self.single_step_joint()

    def set_inner_loop_size(self, s1, s2, s3, s4):
        self.in_loop_size1 = s1
        self.in_loop_size2 = s2
        self.in_loop_size3 = s3
        self.in_loop_size4 = s4

    def burn_in_period(self):
        pi1 = lambda c: self.pi_like( c, self.sample_last2, self.sample_last3, self.sample_last4)
        self.update1.set_pi_target(pi1)

        local_acc1 = [1]
        num_adapt1 = 0
        for i in range(1,200):
            sample1, target1, acc1 = self.update1.step(self.beta1)
            local_acc1.append( acc1 )

            if(i%10 == 0):
                av_acc = np.mean( local_acc1[10:] )
                zeta = 1/np.sqrt(num_adapt1+1)
                self.beta1 = np.exp(np.log(self.beta1) + zeta*(av_acc-self.star_acc))
                num_adapt1 += 1
        print(np.mean(local_acc1))


        self.sample_last1 = sample1
        self.target_last1 = target1

        self.all_samples1.append( sample1 )
        self.all_targets1.append( target1 )
        self.all_accepts1.append( 1 )


        pi2 = lambda r: self.pi_like( self.sample_last1, r, self.sample_last3, self.sample_last4 )
        self.update2.set_pi_target(pi2)

        local_acc2 = [1]
        num_adapt2 = 0
        for i in range(1,200):
            sample2, target2, acc2 = self.update2.step(self.beta2)
            local_acc2.append( acc2 )

            if(i%10 == 0):
                av_acc = np.mean( local_acc2[10:] )
                zeta = 1/np.sqrt(num_adapt2+1)
                self.beta2 = np.exp(np.log(self.beta2) + zeta*(av_acc-self.star_acc))
                num_adapt2 += 1
        print(np.mean(local_acc2))

        self.sample_last2 = sample2
        self.target_last2 = target2

        self.all_samples2.append( sample2 )
        self.all_targets2.append( target2 )
        self.all_accepts2.append( 1 )
        
        pi3 = lambda sigma1: self.pi_like( self.sample_last1, self.sample_last2, sigma1, self.sample_last4 )
        self.update3.set_pi_target(pi3)

        local_acc3 = [1]
        num_adapt3 = 0
        for i in range(1,200):
            sample3, target3, acc3 = self.update3.step(self.beta3)
            local_acc3.append( acc3 )

            if(i%10 == 0):
                av_acc = np.mean( local_acc3[10:] )
                zeta = 1/np.sqrt(num_adapt3+1)
                self.beta3 = np.exp(np.log(self.beta3) + zeta*(av_acc-self.star_acc))
                num_adapt3 += 1
        print(np.mean(local_acc3))

        self.sample_last3 = sample3
        self.target_last3 = target3

        self.all_samples3.append( sample3 )
        self.all_targets3.append( target3 )
        self.all_accepts3.append( 1 )
        
        pi4 = lambda theta: self.pi_like( self.sample_last1, self.sample_last2, self.sample_last3, theta )
        self.update4.set_pi_target(pi4)

        local_acc4 = [1]
        num_adapt4 = 0
        for i in range(1,200):
            sample4, target4, acc4 = self.update4.step(self.beta4)
            local_acc4.append( acc4 )

            if(i%10 == 0):
                av_acc = np.mean( local_acc4[10:] )
                zeta = 1/np.sqrt(num_adapt4+1)
                self.beta4 = np.exp(np.log(self.beta4) + zeta*(av_acc-self.star_acc))
                num_adapt4 += 1
        print(np.mean(local_acc4))

        self.sample_last4 = sample4
        self.target_last4 = target4

        self.all_samples4.append( sample4 )
        self.all_targets4.append( target4 )
        self.all_accepts4.append( 1 )

    def single_step_joint(self):
        pi1 = lambda c: self.pi_like( c, self.sample_last2, self.sample_last3, self.sample_last4 )
        self.update1.set_pi_target(pi1)

        for i in range(self.in_loop_size1):
            sample1, target1, acc1 = self.update1.step(self.beta1)

        self.sample_last1 = sample1
        self.target_last1 = target1

        self.all_samples1.append( sample1 )
        self.all_targets1.append( target1 )
        self.all_accepts1.append( acc1 )


        pi2 = lambda r: self.pi_like( self.sample_last1, r, self.sample_last3, self.sample_last4 )
        self.update2.set_pi_target(pi2)

        for i in range(self.in_loop_size2):
            sample2, target2, acc2 = self.update2.step(self.beta2)

        self.sample_last2 = sample2
        self.target_last2 = target2

        self.all_samples2.append( sample2 )
        self.all_targets2.append( target2 )
        self.all_accepts2.append( acc2 )
        
        
        pi3 = lambda sigma_1: self.pi_like( self.sample_last1, self.sample_last2, sigma_1, self.sample_last4 )
        self.update3.set_pi_target(pi3)

        for i in range(self.in_loop_size3):
            sample3, target3, acc3 = self.update3.step(self.beta3)

        self.sample_last3 = sample3
        self.target_last3 = target3

        self.all_samples3.append( sample3 )
        self.all_targets3.append( target3 )
        self.all_accepts3.append( acc3 )
        
        pi4 = lambda theta: self.pi_like( self.sample_last1, self.sample_last2, self.sample_last3, theta )
        self.update4.set_pi_target(pi4)

        for i in range(self.in_loop_size4):
            sample4, target4, acc4 = self.update4.step(self.beta4)

        self.sample_last4 = sample4
        self.target_last4 = target4

        self.all_samples4.append( sample4 )
        self.all_targets4.append( target4 )
        self.all_accepts4.append( acc4 )

    def print_stat(self):
        print( np.array( self.all_accepts1 ).mean() )
        print( np.array( self.all_accepts2 ).mean() )
        print( np.array( self.all_accepts3 ).mean() )
        print( np.array( self.all_accepts4 ).mean() )

    def give_stats(self):
        return np.array(self.all_samples1), np.array(self.all_samples2), np.array(self.all_samples3), np.array(self.all_samples4)

class Metropolis_update:
    def __init__(self, pi_target,x_old,constraint):
        self.pi_target = pi_target
        self.x_old = x_old
        self.dim = self.x_old.shape[0]
        self.constraint = constraint
        self.target_old = self.pi_target( self.x_old )

    def set_pi_target(self, pi_target):
        self.pi_target

    def step(self, beta):
        flag = False
        while( flag == False ):
            x = self.x_old + beta*np.random.standard_normal(self.dim)
            flag = self.constraint(x)

        target = self.pi_target( x )
        ratio = np.exp(target - self.target_old)
        alpha = min(1., ratio)
        uu = np.random.uniform(0,1)
        if (uu <= alpha):
            acc = 1
            self.x_old = x
            self.target_old = target
        else:
            acc = 0

        return self.x_old, self.target_old, acc


