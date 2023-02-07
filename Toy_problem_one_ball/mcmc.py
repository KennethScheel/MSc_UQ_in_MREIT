import scipy.stats as scp
import numpy as np

class Metropolis_in_Gibbs_burn_in:
    def __init__(self, pi_like, x01, x02, bounds1, bounds2, beta, within_loop_size, burn_in_flag):
        self.pi_like = pi_like
        self.x01 = x01
        self.x02 = x02
        self.ndim1 = self.x01.shape[0]
        self.ndim2 = self.x02.shape[0]
        self.Ndim = bounds1.shape[0]

        self.target_last1 = pi_like(x01,x02)
        self.target_last2 = pi_like(x01,x02) 

        self.sample_last1 = self.x01
        self.sample_last2 = self.x02

        self.all_samples1 = [ self.x01 ]
        self.all_targets1 = [ self.target_last1 ]
        self.all_accepts1 = [ 1 ]

        self.all_samples2 = [ self.x02 ]
        self.all_targets2 = [ self.target_last2 ]
        self.all_accepts2 = [ 1 ]

        pi1 = lambda c: self.pi_like( c, self.sample_last2 )
        if self.Ndim == 2:
           constraint1 = lambda c: (c[0]>=bounds1[0,0])and(c[0]<=bounds1[0,1])and(c[1]>=bounds1[1,0])and(c[1]<=bounds1[1,1])
        elif self.Ndim == 3:
           constraint1 = lambda c: (c[0]>=bounds1[0,0])and(c[0]<=bounds1[0,1])and(c[1]>=bounds1[1,0])and(c[1]<=bounds1[1,1])and(c[2]>=bounds1[2,0])and(c[2]<=bounds1[2,1])
        self.update1 = Metropolis_update( pi1, self.sample_last1, constraint1)

        pi2 = lambda r: self.pi_like( self.sample_last1, r )
        constraint = lambda r: np.array([(r[i]>=bounds2[0])and(r[i]<=bounds2[1]) for i in range(self.ndim2)], dtype=bool).all()
        self.update2 = Metropolis_update( pi2, self.sample_last2, constraint)

        self.in_loop_size1 = within_loop_size        # size of the within-Gibbs Metropolis loop (use larger size for higher quality samples)
        self.in_loop_size2 = within_loop_size

        self.star_acc = 0.234

        self.beta1 = beta[0]
        self.beta2 = beta[1]
        
        self.burn_in_flag = burn_in_flag

    def sample(self, Ns, Nb):
        if self.burn_in_flag:
            print(self.beta1)
            print(self.beta2)
            print('burn-in period:')
            for s in range(Nb):
                self.burn_in_period()
            
        print(self.beta1)
        print(self.beta2)

        print('sampling period:')
        for s in range(Ns-1):
            self.single_step_joint()

    def set_inner_loop_size(self, s1, s2):
        self.in_loop_size1 = s1
        self.in_loop_size2 = s2

    def burn_in_period(self):
        pi1 = lambda c: self.pi_like( c, self.sample_last2 )
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


        pi2 = lambda r: self.pi_like( self.sample_last1, r )
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

    def single_step_joint(self):
        pi1 = lambda c: self.pi_like( c, self.sample_last2 )
        self.update1.set_pi_target(pi1)

        for i in range(self.in_loop_size1):
            sample1, target1, acc1 = self.update1.step(self.beta1)

        self.sample_last1 = sample1
        self.target_last1 = target1

        self.all_samples1.append( sample1 )
        self.all_targets1.append( target1 )
        self.all_accepts1.append( acc1 )


        pi2 = lambda r: self.pi_like( self.sample_last1, r )
        self.update2.set_pi_target(pi2)

        for i in range(self.in_loop_size2):
            sample2, target2, acc2 = self.update2.step(self.beta2)

        self.sample_last2 = sample2
        self.target_last2 = target2

        self.all_samples2.append( sample2 )
        self.all_targets2.append( target2 )
        self.all_accepts2.append( acc2 )

    def print_stat(self):
        print( np.array( self.all_accepts1 ).mean() )
        print( np.array( self.all_accepts2 ).mean() )

    def give_stats(self):
        return np.array(self.all_samples1), np.array(self.all_samples2)

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


