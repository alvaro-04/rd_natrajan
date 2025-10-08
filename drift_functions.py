import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sympy import *
import matplotlib.pyplot as plt



def learner(U,X, Z, inh, err_thresh):
    Nx = np.shape(U)[1]
    Ny = np.shape(U)[0]
    Nz = np.shape(Z)[0]
    P = np.shape(X)[1]
    input_data = torch.Tensor(X.T) 
    target_data = torch.Tensor(Z.T)
    model = df.MLP(Nx, Ny, Nz,  U, W, inh)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    all_errors = []
    found_sol = False
    successfully_found = False
    while found_sol == False:
        output = model(input_data)
        loss = criterion(output, target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        all_errors.append(loss.item())
        if err_thresh>abs(all_errors[-1]):
            found_sol = True
            successfully_found = True
    
        elif len(all_errors)>10 and np.isclose(all_errors[-10], all_errors[-1]):
            found_sol = True
    print(all_errors[-1])
    return(found_sol, (model.l1.weight.detach()).numpy())



class ReLU2(nn.Module):
    def __init__(self):
        super(ReLU2, self).__init__()

    def forward(self, x, hmin=0, hmax=5):
        return torch.clamp(nn.functional.relu(x), min = hmin, max=hmax)


class MLP(nn.Module):
  def __init__(self,Nx, Ny, Nz, U_allocated, W, inh=0):
    super().__init__()
    
    self.l1 = nn.Linear(Nx, Ny, bias = True)
    self.l1.bias.data = torch.from_numpy(inh*np.ones(Ny, dtype = 'float32'))
    self.l1.bias.requires_grad = False
    self.l1.weight.data = torch.from_numpy(np.array(U_allocated, dtype = 'float32'))
    self.l2 = ReLU2()
    self.l3 = nn.Linear(Ny, Nz, bias = False)
    self.l3.weight.data = torch.from_numpy(np.array(W, dtype = 'float32'))

    for param in self.l3.parameters():
        param.requires_grad = False
      
    self.layers = nn.Sequential(self.l1, self.l2, self.l3)
    

  def forward(self, x):
    
    x = self.l1(x)
    x = self.l2(x)
    x = self.l3(x)
    ##print(x)
    return x




def U_norm_satisfier_new(U, X_ext, del_eta, eta, d_all, Umax_all, Nx, Ny, Nz,inh):
    redo = False
    eta = eta-inh 
    eta_hat = eta + del_eta
    X_in = np.linalg.inv(X_ext)
    U_new = (eta_hat)@X_in
    eta_new = np.zeros((Ny,Nx))
    all_alphas = []
    crossed=[]
    problems = []
    for i in range(0, Ny):

        d = d_all[i]
        Umax = Umax_all[i]
        Umax_new = np.sum(U_new[i,:]**2)
        if Umax_new> Umax:
            crossed.append('crossed')
            a = np.sum(((del_eta[i, ~d])@(X_in[~d,:]))**2)

            b1 = 2*(eta[i,~d]@(X_in[~d,:]))*(del_eta[i,~d]@(X_in[~d,:]))
            b2 = 2*(eta_hat[i,d]@X_in[d,])*(del_eta[i,~d]@X_in[~d,])
            b = np.sum(b1+b2)

            c1 = (eta_hat[i,d]@X_in[d,:])**2
            c2 = ((eta[i,~d])@(X_in[~d,:]))**2
            c3 = 2*(eta_hat[i,d]@X_in[d,:])*(eta[i,~d]@X_in[~d,:])
            c = np.sum(c1+c2+c3)-Umax

            if (b**2)-(4*a*c)>0:
                ##print(i, 'ok')
                var('x')
                alphas = solve(a*(x**2)+b*x+c, x)
                ##print(alphas)
                if alphas[0]>0 and alphas[0]<1:
                    alpha = alphas[0]
                    all_alphas.append(alpha)
                    problems.append(0)
                elif alphas[1]>0 and alphas[1]<1:
                    alpha = alphas[1]
                    all_alphas.append(alpha)
                    problems.append(0)
                else:
                    all_alphas.append('not between 0 and 1')
                    redo = True
                    alpha =1 
                    problems.append(1)

                del_eta_prop = np.copy(del_eta[i,:])
                del_eta_prop[~d] = alphas[1]*del_eta[i,~d]
                eta_new[i,:] = eta[i,:] + del_eta_prop

            else:
                eta_new[i,:] = eta[i,:] + del_eta[i,:]
                all_alphas.append('imaginary')
                redo = True
                problems.append(1)
        else:
            crossed.append('not crossed')
            eta_new[i,:] = eta[i,:] + del_eta[i,:]
            all_alphas.append(1)
            problems.append(0)
        
    return(eta_new, redo, problems)





def representational_drift_fixed_Unorm (U, X,W,  num_runs, h, h_up, Umax_fixed_all, Wmax_fixed, W_changes_allowed, eta_clip= False, eta_min=0, eta_max=0, lr=1, inh=0):
    save_every = 5
    error_all =[]
    h_prime = 10**-20
    U_diff = []
    Y_diff = []
    all_problems =[]

    U_initial = np.copy(U)

    P = np.shape(X)[1]
    Nx = np.shape(X)[0]
    Ny = np.shape(W)[1]
    Nz = np.shape(W)[0]
    
    

    if Nx> P:
        X_ex = np.concatenate((X, scipy.linalg.null_space(X.T)), axis = 1)
    else:
        X_ex = X
    
    eta = np.matmul(U,X_ex) + inh
    
    Y = np.clip(np.matmul(U, X) + inh, h, h_up)
    Z_given_store = np.matmul(W, Y)
    print('Z given', Z_given_store)

    Active_all = {}
    Inactive_all = {}
    Saturated_all = {}
    first_time_active = {}
    first_time_active_sat = {}
    first_time_inactive = {}
    first_time_saturated = {}
    for mu in range(0,P):
        Y_given = np.matmul(U, X)[:,mu] + inh
        Active_all[mu] =  np.where(np.logical_and((Y_given>(h-h_prime)),Y_given< (h_up-h_prime)))[0]
        Inactive_all[mu] = np.where(Y_given<=h-h_prime)[0]
        Saturated_all[mu] = np.where(Y_given>=h_up+h_prime)[0]
        first_time_active[mu] = []
        first_time_inactive[mu] = []
        first_time_saturated[mu] = []
        first_time_active_sat[mu] = []

    num_saturated = []
    num_inactive = []

    U_sum_all= []
    num_saturated = []
    num_inactive = []
    W_sum_all= []
    sigma_mu = 0.1*lr
    ##h = 0 ##10**-10 ##10**-10 ##2*10**-11
    ##h_up = 5
    hprime = 10**-20

    ##all_errors = []
    all_Ys= []
    all_Us = []
    all_Y_nums = np.arange(0, Ny)
    W_plus = np.linalg.pinv(W)
    pi_w = np.matmul(W_plus, W)## projection matrix

    redos = np.zeros(num_runs)
    for t in range (0, num_runs):
        
        W_plus = np.linalg.pinv(W)
        pi_w = np.matmul(W_plus, W)## projection matrix
        
        
        
        redo = True
        Active_all_prev = Active_all.copy()
        Inactive_all_prev = Inactive_all.copy()
        Saturated_all_prev = Saturated_all.copy()
        first_time_active_prev = first_time_active.copy()
        first_time_active_sat_prev = first_time_active_sat.copy()
        first_time_inactive_prev = first_time_inactive.copy()
        first_time_saturated_prev = first_time_saturated.copy()
        while redo== True:
            ##print('doing')
            Active_all = Active_all_prev.copy()
            Inactive_all = Inactive_all_prev.copy()
            Saturated_all = Saturated_all_prev.copy()
            first_time_active = first_time_active_prev.copy()
            first_time_active_sat = first_time_active_sat_prev.copy()
            first_time_inactive = first_time_inactive_prev.copy()
            first_time_saturated = first_time_saturated_prev.copy()

            U_norm_satisfied = False
            del_eta_con = np.zeros((Ny, Nx))


            for mu in range(0, Nx):
                if mu < P:
                    
                    Y_given = Y[:,mu]
                    Active = Active_all[mu]
                    Inactive = Inactive_all[mu]
                    Saturated = Saturated_all[mu]
                    Y_tilda_saturated =  Y_given[Saturated]
                    Y_tilda_active =  Y_given[Active]
                    Y_tilda_inactive = Y_given[Inactive]
                    W_saturated = W[:,Saturated]
                    W_active = W[:,Active]
                    W_inactive = W[:,Inactive]

                    try:  ## adding fails when all neurons are disengaged
                        k_w = scipy.linalg.null_space(W[:,Active])
                        no_engaged = False
                    except:
                        no_engaged = True
                        print('no engaged')
                        print(Active)

                    found_first_active_nonneg = False
                    found_first_active_below = False
                    while (found_first_active_nonneg == False or found_first_active_below == False) and no_engaged == False:
                        den2 = np.sum(np.linalg.inv(X_ex)[mu,:]**2)
                        den3 = np.sum(k_w**2, axis = 1)

                        del_r = np.random.normal(0, (sigma_mu)/np.sqrt(np.sum(den2)), np.shape(k_w)[1])
                        
                        del_eta_cons_potential = np.matmul(k_w,del_r) 
                        
                        if len(first_time_active[mu])!=0:
                            neur_check = []
                            for neur_check_ind in range(len( first_time_active[mu])):
                                neur_check.append(np.where(Active_all[mu]==first_time_active[mu][neur_check_ind])[0][0])
                            if np.all(del_eta_cons_potential[neur_check]>0):
                                found_first_active_nonneg = True
                        else:
                            found_first_active_nonneg = True


                        if len(first_time_active_sat[mu])!= 0:
                            neur_check = []
                            for neur_check_ind in range(len( first_time_active_sat[mu])):
                                neur_check.append(np.where(Active_all[mu]==first_time_active_sat[mu][neur_check_ind])[0][0])

                            if np.all(del_eta_cons_potential[neur_check]< 0):
                                found_first_active_below = True

                        else:
                            found_first_active_below = True
                            
                        

                    first_time_active[mu] = [] ## reset
                    first_time_active_sat[mu] = []
                    Y_changing_cons = eta[Active, mu] +del_eta_cons_potential
                    mult = 1
                    mult_sat = 1

                    inactivated_neuron = False
                    saturated_neuron = False
                    if len(np.where(Y_changing_cons<=h)[0])>0 or len(np.where(Y_changing_cons>=h_up)[0])>0:

                        problematic = [Y_changing_cons<=h][0]  
                        if np.sum(problematic)!=0:
                            min_to_be_changed = np.min((abs(h-eta[Active, mu]))[problematic]/abs(del_eta_cons_potential[problematic]))
                            to_be_changed1 = np.where((abs(h-eta[Active, mu]))/abs(del_eta_cons_potential) == min_to_be_changed)[0][0]
                            mult = abs(h-eta[Active][to_be_changed1, mu])/abs(del_eta_cons_potential[to_be_changed1])

                        problematic_sat = [Y_changing_cons>=h_up][0]
                        if np.sum(problematic_sat)!=0:

                            min_to_be_changed_act_sat = np.min((h_up-eta[Active, mu][problematic_sat])/abs(del_eta_cons_potential[problematic_sat]))
                            to_be_changed_act_sat = np.where((h_up-eta[Active, mu])/abs(del_eta_cons_potential) == min_to_be_changed_act_sat)[0][0]
                            mult_sat = (h_up-eta[Active][to_be_changed_act_sat, mu])/del_eta_cons_potential[to_be_changed_act_sat]
  
                        
                        if mult>1:
                            print(t, mu, 'mult', Y_changing_cons,mult, min_to_be_changed, h,  eta[Active][to_be_changed1, mu], abs(h-eta[Active][to_be_changed1, mu]),abs(del_eta_cons_potential[to_be_changed1]))
                        if mult_sat>1:
                            print(t, mu,'mult_sat', Y_changing_cons,mult_sat, min_to_be_changed_act_sat, h_up, eta[Active][to_be_changed_act_sat, mu], h_up-eta[Active][to_be_changed_act_sat, mu], del_eta_cons_potential[to_be_changed_act_sat])
                        
                        if mult_sat<mult:
                            
                            saturated_neuron = True

                            
                            to_be_changed_act_sat_val = Active[to_be_changed_act_sat]
                            
                            Active_all[mu]= np.sort(np.setdiff1d(Active_all[mu],to_be_changed_act_sat_val))

                            Saturated_all[mu] = np.sort(np.append(Saturated_all[mu],to_be_changed_act_sat_val))
                            del_eta_con[Active,mu] = del_eta_cons_potential*mult_sat
 

                        else:
                            inactivated_neuron = True
                            
                            to_be_changed1_val = Active[to_be_changed1]
                            
                            Active_all[mu]= np.sort(np.setdiff1d(Active_all[mu],to_be_changed1_val))
                            Inactive_all[mu] = np.sort(np.append(Inactive_all[mu],to_be_changed1_val))
                            del_eta_con[Active,mu] = del_eta_cons_potential*mult 
                            
                            


                    else:
                        del_eta_con[Active,mu] = del_eta_cons_potential


                    sigma_inactive = sigma_mu/np.sqrt(np.sum(np.linalg.inv(X_ex)[mu,:]**2))
                    found_first_active_nonneg = False
                    while found_first_active_nonneg == False:

                        new_change = np.random.normal(0,sigma_inactive, len(Inactive))
                        if len(first_time_inactive[mu])!=0:
                            if new_change[np.where(Inactive==first_time_inactive[mu][0])[0][0]]<0:
                                found_first_active_nonneg = True
                        else:
                            found_first_active_nonneg = True

                    Y_changing = eta[Inactive, mu] + new_change


                    del_eta_con[Inactive,mu] = np.minimum(Y_changing, h)- eta[Inactive, mu] ## maybe problem

                    if len(np.where(Y_changing>=h)[0])>0:
                        ##print('first time active')
                        to_be_changed =Inactive[np.where(Y_changing>=h)[0]]
                        Active_all[mu]= np.sort(np.concatenate((Active_all[mu],to_be_changed)))

                        Inactive_all[mu] = np.setdiff1d(Inactive_all[mu],to_be_changed)

                        first_time_active[mu]= to_be_changed
                    if len(np.intersect1d(Active_all[mu], Inactive_all[mu]))!= 0:
                        print('something wrong, check, inactive, active')

                    first_time_inactive[mu]= [] ## rest
                    if inactivated_neuron == True:
                        first_time_inactive[mu]= [to_be_changed1_val]



                    ## saturated
                    sigma_saturated = sigma_inactive

                    found_first_active_below = False
                    while found_first_active_below == False:
                        new_change = np.random.normal(0,sigma_saturated, len(Saturated))
                        if len(first_time_saturated[mu])!=0:
                            if new_change[np.where(Saturated==first_time_saturated[mu][0])[0][0]]>0:
                                found_first_active_below = True

                        else:
                            found_first_active_below = True

                    Y_changing = eta[Saturated, mu] + new_change


                    del_eta_con[Saturated,mu] = np.maximum(Y_changing, h_up)- eta[Saturated, mu]


                    if len(np.where(Y_changing<=h_up)[0])>0:
                        to_be_changed_sat =Saturated[np.where(Y_changing<h_up)[0]]
                        Active_all[mu]= np.sort(np.concatenate((Active_all[mu],to_be_changed_sat)))

                        Saturated_all[mu] = np.setdiff1d(Saturated_all[mu],to_be_changed_sat)
                        first_time_active_sat[mu]= to_be_changed_sat 

                    if len(np.intersect1d(Active_all[mu], Saturated_all[mu]))!= 0:
                        print('something wrong, check, active, saturated')

                    first_time_saturated[mu]=[]
                    if saturated_neuron == True:
                        first_time_saturated[mu] = [to_be_changed_act_sat_val]


                else:
                    del_eta_con[:,mu] =np.random.normal(0, sigma_mu/np.sqrt(np.sum(np.linalg.inv(X_ex)[mu,:]**2)), Ny)

            
            d_all = np.full((Ny, Nx), False)
            for mu in range(len(Active_all)):
                d_all[Active_all[mu], mu]= True
            eta_new, redo, problems = U_norm_satisfier_new(U, X_ex, del_eta_con, eta, d_all,Umax_fixed_all, Nx, Ny, Nz, inh)
            
            
            all_problems.append(problems)
            if redo == True:
                redos[t] = redos[t]+1
            else:

                eta = eta_new + inh ## U_norm_satisfier_new uses effective eta


        U_prev = np.copy(U)
        Y_prev = np.copy(Y)
        
        if eta_clip == True:
            eta = np.clip(eta, eta_min, eta_max)
        U = np.matmul(eta-inh, np.linalg.inv(X_ex))
        Y = np.clip(np.matmul(U,X)+ inh, h, h_up)
        Z = W@Y
  

        if t%save_every == 0:
            
            U_sum_all.append(np.sum(U**2, axis = 1))
            num_saturated.append(np.sum(eta[:,0:P]>=h_up))
            num_inactive.append(np.sum(eta[:,0:P]<=h))
            all_Us.append(U)
            
        if t%100 ==0:
            print(t, np.sum(U**2, axis = 1), Z)
    return(U, num_saturated, num_inactive, U_sum_all, W, W_sum_all, all_Us, error_all, U_diff, Y_diff, redos, all_problems)



def robustness_plotter(new_U, U_drift, T_all, h_min, h_max,W, W_new, which_add):
    fig, axs = plt.subplots(1,3, figsize = (20,5))
    counter = 0
    mean_diff_before= []
    mean_sd_before = []
    mean_diff_after= []
    mean_sd_after = []
    ##for noise_sd in [0.01, 0.1, 1]:
    for noise_sd in [0.1]:
        U_curr_before = new_U.copy()
        U_curr_after = U_drift.copy()

        before_errors = []
        after_errors = []

        for noises in range(0, 1000):
            actual_output = np.matmul(W, np.clip(np.matmul(U_curr_before, T_all), h_min, h_max))[0]
            if which_add == 'U':
                add_noise = np.random.normal(0, noise_sd, np.shape(U_curr_before))
                add_noise_W = np.zeros_like(W)
            elif which_add == 'W':
                add_noise = np.zeros_like(U_curr_before)
                add_noise_W = np.random.normal(0, noise_sd, np.shape(W))
            elif which_add == 'UW':
                add_noise = np.random.normal(0, noise_sd, np.shape(U_curr_before))
                add_noise_W = np.random.normal(0, noise_sd, np.shape(W))
            new_output_before = np.matmul(W + add_noise_W, np.clip(np.matmul(U_curr_before + add_noise, T_all), h_min, h_max))[0]
            new_output_after = np.matmul(W_new + add_noise_W, np.clip(np.matmul(U_curr_after + add_noise, T_all), h_min, h_max))[0]
            before_errors = np.concatenate((before_errors,actual_output-new_output_before))
            after_errors = np.concatenate((after_errors,actual_output-new_output_after))

        bins=np.histogram(np.hstack((before_errors,after_errors)), bins=100)[1] #get the bin edges


        axs[counter].hist(before_errors, bins = bins, label= 'errors before drift', alpha = 0.5)

        axs[counter].hist(after_errors, bins = bins, label = 'errors after drift', alpha = 0.5)
        axs[counter].legend()
        axs[counter].set_title(str(np.shape(T_all)[1]) + ' patterns, noise sd = ' + str(noise_sd))
        axs[counter].axvline(0)
        
        counter = counter + 1
        mean_diff_before.append(np.mean(before_errors))
        mean_sd_before.append(np.std(before_errors))
        mean_diff_after.append(np.mean(after_errors))
        mean_sd_after.append(np.std(after_errors))
        
    
        
    before_errors = np.array(before_errors)
    after_errors = np.array(after_errors)
    mean_all = [mean_diff_before, mean_diff_after, mean_sd_before, mean_sd_after]
    return(mean_all, np.sum(before_errors**2)/len(before_errors), np.sum(after_errors**2)/len(after_errors))
