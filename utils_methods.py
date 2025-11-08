import copy
import math
import os.path

import numpy as np
import pandas as pd
import torch

from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *
from tensorboardX import SummaryWriter
from collections import Counter
from train_gan_fn import *


def compute_ndvar(clnt_params_list, cld_mdl_param, selected_clnts):

    # 保证 cld_mdl_param 是 Tensor
    if isinstance(cld_mdl_param, np.ndarray):
        cld_mdl_param = torch.tensor(cld_mdl_param)

    grads = []
    for idx in selected_clnts:
        param_i = clnt_params_list[idx]
        if isinstance(param_i, np.ndarray):
            param_i = torch.tensor(param_i)

        grad_i = param_i - cld_mdl_param
        grads.append(grad_i)

    grads_stack = torch.stack(grads)
    mean_grad = grads_stack.mean(dim=0)

    var_sum = 0.0
    for grad in grads:
        diff = grad - mean_grad
        var_sum += torch.sum(diff ** 2)

    ndvar = var_sum / len(selected_clnts)
    return ndvar.item()
##
### Methods
def train_FedAvg(data_obj, act_prob ,learning_rate, batch_size, epoch,
                                     com_amount, print_per, weight_decay,
                                     model_func, init_model, sch_step, sch_gamma,
                                     save_period, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedAvg_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed

    n_clnt=data_obj.n_clnt

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y


    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-New/%s/%s' %(data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model_avg(clnt_models[clnt], trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel




def train_FedProx(data_obj, act_prob ,learning_rate, batch_size, epoch,
                                     com_amount, print_per, weight_decay,
                                     model_func, init_model, alpha_coef, sch_step, sch_gamma,
                                     save_period, weight_uniform = False, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedProx_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay,alpha_coef)

    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed
    suffix += '_WU%s' %(weight_uniform)

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y


    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))


    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))


    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par

    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedProx/%s/%s' %(data_path, data_obj.name, suffix[:26]))


    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model





                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
            cld_mdl_param = get_mdl_params([avg_model], n_par)[0]


        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness

            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False
                # Add regulariser during training
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #local train
                clnt_models[clnt] = train_model_prox(clnt_models[clnt],cld_mdl_param_tensor, alpha_coef, trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i),
                                                batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            cld_mdl_param= np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0)
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))


            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel
def train_SCAFFOLD(data_obj, act_prob ,learning_rate, batch_size, n_minibatch,
                                     com_amount, print_per, weight_decay,
                                     model_func, init_model, sch_step, sch_gamma,
                                     save_period, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1, global_learning_rate=1):
    suffix = 'Scaffold-Adan_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)

    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed

    n_clnt=data_obj.n_clnt

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y


    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it

    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))


    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params_diffs = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER/%s/%s' %(data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model


                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):

                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))

                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))                    # Get state_params_diffs
                    state_params_diffs = np.load('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))

        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models

            clnt_models = list(range(n_clnt))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]



                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True

                # Scale down c
                state_params_diff_curr = torch.tensor(-state_params_diffs[clnt] + state_params_diffs[-1]/weight_list[clnt], dtype=torch.float32, device=device)

                clnt_models[clnt] = train_scaffold_mdl(clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y,
                    learning_rate * (lr_decay_per_round ** i), batch_size, n_minibatch, print_per,
                    weight_decay, data_obj.dataset, sch_step, sch_gamma)

                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_params_diffs[clnt] - state_params_diffs[-1] + 1/n_minibatch/learning_rate * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_params_diffs[clnt])*weight_list[clnt]
                state_params_diffs[clnt] = new_c

                clnt_params_list[clnt] = curr_model_param


            avg_model_params = global_learning_rate*np.mean(clnt_params_list[selected_clnts], axis = 0) + (1-global_learning_rate)*prev_params

            avg_model = set_client_from_params(model_func().to(device), avg_model_params)

            state_params_diffs[-1] += 1 / n_clnt * delta_c_sum


            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])


                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, (i+1)), state_params_diffs)


                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_FedDyn(data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, print_per,
                  weight_decay,  model_func, init_model, alpha_coef,
                  sch_step, sch_gamma, save_period,
                  suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix  = 'FedDyn_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' %rand_seed
    suffix += '_lrdecay%f' %lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y


    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances)) # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DP/%s/%s' %(data_path, data_obj.name, suffix[:45]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                               %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' %(data_path, data_obj.name, suffix, i+1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr//save_period] = fed_cld

                if os.path.exists('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1))):

                    tst_cur_cld_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))



    if (trial) or (not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' %(data_path, data_obj.name, suffix, com_amount))):


        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]


        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedDyn(model, model_func, alpha_coef_adpt,
                                                     cld_mdl_param_tensor, hist_params_diffs_curr,
                                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                     batch_size, epoch, print_per, weight_decay,
                                                     data_obj.dataset, sch_step, sch_gamma)
                # 训练后应用DP
                w_per = get_mdl_params([clnt_models[clnt]], n_par)[0]  # 获取全局模型参数
                w_global = get_mdl_params([cur_cld_model], n_par)[0]

                # 计算参数更新并添加DP噪声
                w_per_flat = np.concatenate([p.flatten() for p in w_per])  # 将参数展平为一维数组
                w_global_flat = np.concatenate([p.flatten() for p in w_global])  # 将全局参数展平

                # 计算参数更新
                nabala_flat = w_per_flat - w_global_flat

                # 梯度裁剪与噪声注入
                noise_scale = 0.95
                clip_norm = 0.2
                client_num = len(selected_clnts)

                # 计算原始梯度范数
                norm = np.linalg.norm(nabala_flat, 2)

                # 梯度裁剪
                clip_coef = min(1, clip_norm / (norm + 1e-8))
                nabala_flat_clipped = nabala_flat * clip_coef

                # 生成高斯噪声 (σ·C/√m)
                noise = np.random.normal(0, noise_scale * clip_norm / np.sqrt(client_num), nabala_flat.shape)

                # 噪声注入
                nabala_flat_dp = nabala_flat_clipped + noise

                # 应用DP保护后的更新
                w_per_flat_dp = w_global_flat + nabala_flat_dp

                # 恢复参数形状并加载回模型
                start_idx = 0
                for i, param in enumerate(clnt_models[clnt].parameters()):
                    param_size = np.prod(param.shape)
                    param_data = w_per_flat_dp[start_idx:start_idx + param_size].reshape(param.shape)
                    param.data = torch.from_numpy(param_data).to(param.device)
                    start_idx += param_size
                # Scale with weights
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis = 0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]
            print(i)


            writer.add_scalars('Loss/test',
                   {
                       'Current cloud':tst_cur_cld_perf[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Current cloud':tst_cur_cld_perf[i][1]
                   }, i
                  )

            if (not trial) and ((i+1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_cur_cld_perf[:i+1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, (i+1)), hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)


                if (i+1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))



            if ((i+1) % save_period == 0):
                avg_cld_mdls[i//save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf


def train_FedDC(data_obj, act_prob, n_minibatch,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func, init_model, alpha_coef,
                sch_step, sch_gamma, save_period,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedDC-Adan_' + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)

    avg_cld_mdls = list(range(n_save_instances))

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    ###
    state_gadient_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32')  # including cloud state

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i


                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_clnt))
            delta_g_sum = np.zeros(n_par)

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                local_update_last = state_gadient_diffs[clnt]  # delta theta_i
                global_update_last = state_gadient_diffs[-1] / weight_list[clnt]  # delta theta
                alpha = alpha_coef / weight_list[clnt]
                hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                alpha=alpha*(1/math.sqrt(saved_itr+2))
                clnt_models[clnt] = train_model_DPFedDC(model, model_func, alpha, local_update_last, global_update_last,
                                                      global_mdl, hist_i,
                                                      trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                      batch_size, epoch, print_per, weight_decay, data_obj.dataset,
                                                      sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                parameter_drifts[clnt] += delta_param_curr
                beta = 1 / n_minibatch / learning_rate

                state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
                delta_g_cur = (state_g - state_gadient_diffs[clnt]) * weight_list[clnt]
                delta_g_sum += delta_g_cur
                state_gadient_diffs[clnt] = state_g
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            delta_g_cur = 1 / n_clnt * delta_g_sum
            state_gadient_diffs[-1] += delta_g_cur

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####


            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved array
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf



def train_FedDisco(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedDisco_0.5_0.1_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedDisco/%s/%s' % (data_path, data_obj.name, suffix[:28]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedDyn(model, model_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift
                clnt_params_list[clnt] = curr_model_par

            total_data_points = 0
            for clnt in selected_clnts:
                total_data_points += data_obj.clnt_y[clnt].shape[0]

            fed_avg_freqs = [data_obj.clnt_y[clnt].shape[0] / total_data_points for clnt in selected_clnts]

            traindata_cls_counts = np.asarray(
                [np.mean(data_obj.clnt_y[clnt] == cls) for clnt in range(data_obj.n_client) for cls in
                 range(data_obj.n_cls)]).reshape(-1, 10)
            for clnt in range(data_obj.n_client):
                traindata_cls_counts[clnt] = np.multiply(traindata_cls_counts[clnt], data_obj.clnt_y[clnt].shape[0])

            global_dist = np.ones(traindata_cls_counts.shape[1]) / traindata_cls_counts.shape[1]

            # avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            distribution_difference = get_distribution_difference(traindata_cls_counts, selected_clnts, 'kl', global_dist)
            fed_avg_freqs = disco_weight_adjusting(fed_avg_freqs,distribution_difference,0.5,0.1)
            avg_mdl_param_sel = np.dot(np.array(fed_avg_freqs).reshape(1, -1), clnt_params_list[selected_clnts])
            avg_mdl_param_sel = np.squeeze(avg_mdl_param_sel)
            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0) #1/N

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)



            #####



            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf


def train_FedSpeed(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,rho,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedSpeed_' + str(epoch) + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedSpeed/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_Fedspeed(model, model_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                # print('d is {}'.format(d))
                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0) #1/N

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)



            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf

def train_DPFedSpeed(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,rho,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedSpeed0.2_' + str(epoch) + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_DPFedspeed(model, model_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                # print('d is {}'.format(d))
                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0) #1/N

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)



            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf


def train_FedAvr(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedVRA_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y


    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)

    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    lamda = torch.from_numpy(np.zeros((n_clnt, n_par)).astype('float32')).to(device)
    lamda_num = torch.from_numpy(np.zeros(n_clnt).astype('float32')).to(device)

    corr_palty = 0.1
    a_step_size = 7

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedVRA/%s/%s' % (data_path, data_obj.name, suffix[:24]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                # update lamda
                lamda[clnt] += corr_palty * a_step_size * torch.from_numpy(hist_params_diffs[clnt]).to(device)
                lamda_num[clnt]+=1
                lamda[clnt]=lamda[clnt]/lamda_num[clnt]


                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedAvr(model, model_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma,lamda[clnt])
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabmla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par


            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)




            #####


            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf



def train_FedSMOO(data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, print_per,
                  weight_decay, model_func, init_model, alpha_coef, rho,
                  sch_step, sch_gamma, save_period,
                  suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedSMOO_' + str(act_prob) + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_client = data_obj.n_clnt
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    ndvar_value_list = np.zeros((com_amount, 1))

    hist_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    ndvar_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                  -1)  # n_client X n_par
    mu_params_list = torch.zeros(n_client, n_par)
    global_dynamic_dual = torch.zeros(n_par)
    clnt_models = list(range(n_client))
    # Dynamic_dual_list = np.zeros(n_client,n_par).astype('float32')
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedSGAM_KL/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
            not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        all_clnts = torch.tensor([i for i in range(n_client)])
        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_client))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                Dynamic_dual = get_params_list_with_shape(model, mu_params_list[clnt], device)
                Dynamic_dual_correction = get_params_list_with_shape(model, global_dynamic_dual, device)
                clnt_models[clnt], dynamic_dual = train_model_Fedsmoo(model, model_func, alpha_coef_adpt, Dynamic_dual,
                                                                      Dynamic_dual_correction,
                                                                      cld_mdl_param_tensor, hist_params_diffs_curr,
                                                                      trn_x, trn_y,
                                                                      learning_rate * (lr_decay_per_round ** i), rho,
                                                                      batch_size, epoch, print_per, weight_decay,
                                                                      data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift

                ndvar_params_diffs[clnt] = (curr_model_par - cld_mdl_param)
                local_dynamical_dual = dynamic_dual

                clnt_params_list[clnt] = curr_model_par

                mu = []
                for _mu_ in local_dynamical_dual:
                    mu.append(_mu_.clone().detach().cpu().reshape(-1))
                mu_params_list[clnt] = torch.cat(mu)
            #
            # h = 0
            # for clnt in all_clnts:
            #     h += np.sum((clnt_params_list[clnt] - cld_mdl_param) * (clnt_params_list[clnt] - cld_mdl_param))
            # h /= n_client
            # print('h is {}'.format(h))

            variance_value = compute_ndvar(clnt_params_list, cld_mdl_param, selected_clnts)
            print(f"variance_value: {variance_value}")
            ndvar_value_list[i] = variance_value

            avg_dynamic_dual = torch.mean(mu_params_list[selected_clnts], dim=0)
            _l2_ = torch.norm(avg_dynamic_dual, p=2, dim=0) + 1e-7
            global_dynamic_dual = avg_dynamic_dual / _l2_ * rho

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)  # 1/N

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):

                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

            df = pd.DataFrame(ndvar_value_list)
            df.to_csv("ndvar_values_FedSMOO_iid.csv", index=False)

    return avg_cld_mdls, tst_cur_cld_perf

def train_FedSGAM_KL(data_obj, act_prob,
                     learning_rate, batch_size, epoch, com_amount, print_per,
                     weight_decay, model_func, init_model, alpha_coef, rho,
                     sch_step, sch_gamma, save_period,
                     suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedSGAM_KL_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round



    n_client = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_ins_mdls = list(range(n_save_instances))  # Avg active clients
    avg_all_mdls = list(range(n_save_instances))  # Avg all clients
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))

    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))

    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    ndvar_value_list= np.zeros((com_amount, 1))

    hist_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    ndvar_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_client X n_par
    clnt_models = list(range(n_client))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedSGAM_KL/%s/%s' % (data_path, data_obj.name, suffix[:12]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_ins = model_func()
                fed_ins.load_state_dict(
                    torch.load('%sModel/%s/%s/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_ins.eval()
                fed_ins = fed_ins.to(device)

                # Freeze model
                for params in fed_ins.parameters():
                    params.requires_grad = False

                avg_ins_mdls[saved_itr // save_period] = fed_ins

                ####
                fed_all = model_func()
                fed_all.load_state_dict(
                    torch.load('%sModel/%s/%s/all_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_all.eval()
                fed_all = fed_all.to(device)

                # Freeze model
                for params in fed_all.parameters():
                    params.requires_grad = False

                avg_all_mdls[saved_itr // save_period] = fed_all

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_client))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                p = sinp(com_amount,i)
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedSGAM_KL(cur_cld_model,model, model_func, alpha_coef_adpt,
                                                     cld_mdl_param_tensor,
                                                           hist_params_diffs_curr,
                                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                     batch_size, epoch, print_per, weight_decay,
                                                     data_obj.dataset, sch_step, sch_gamma)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                ndvar_params_diffs[clnt] = (curr_model_par - cld_mdl_param)
                clnt_params_list[clnt] = curr_model_par



            variance_value = compute_ndvar(clnt_params_list,cld_mdl_param ,selected_clnts)
            print(f"variance_value: {variance_value}")
            ndvar_value_list[i] = variance_value

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))


                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays

                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

            df = pd.DataFrame(ndvar_value_list)
            df.to_csv("ndvar_values_FedSGAM_KL_iid.csv", index=False)

    return avg_cld_mdls, tst_cur_cld_perf

def train_DPFedSGAM_KL(data_obj, act_prob,
                 learning_rate, batch_size, rho, epoch, n_minibatch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedSGAM_KL_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round



    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round



    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))
    # if not os.path.exists('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8])):
    #     os.mkdir('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8]))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))
    ndvar_value_list=np.zeros((com_amount, 1))
    n_par = len(get_mdl_params([model_func()])[0])

    client_momentum = np.zeros(n_par).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sDPRuns_FedSGAM_KL/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            round_idx=i
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                delta_list = torch.tensor(client_momentum).to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedSGAM_KL(round_idx,clnt_num,cur_cld_model,model, model_func, alpha_coef_adpt,
                                                    delta_list,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)


                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                clnt_params_list[clnt] = curr_model_par




            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            avg_update = avg_mdl_param_sel - cld_mdl_param
            client_momentum = avg_update / n_minibatch / learning_rate * lr_decay_per_round * -1.

            cld_mdl_param = avg_mdl_param_sel

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))


                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays

                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model
    # last_100_acc_tst = [tst_cur_cld_perf[i][1] for i in range(com_amount - 100, com_amount)]
    # mean_acc_tst = np.mean(last_100_acc_tst) * 100
    # var_acc_tst = np.var(last_100_acc_tst) * 100
    # with open('%sAccuracy/%s%s/results.txt' % (data_path, data_obj.name,suffix[:8]), 'w') as f:
    #     f.write(f"{mean_acc_tst} ± {var_acc_tst}\n")


    return avg_cld_mdls, tst_cur_cld_perf,tst_cur_cld_perf


def train_FedTOGA(data_obj, act_prob,
                 learning_rate, batch_size, epoch,n_minibatch, com_amount, print_per,
                 weight_decay,  model_func, init_model, alpha_coef,beta,
                 sch_step, sch_gamma, save_period,
                 suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix  = 'FedTOGA_' + str(alpha_coef)+ suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par),dtype='float32')
    delta_params_diffs = np.zeros((n_clnt,n_par),dtype='float32')
    delta = np.zeros(n_par).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedTOGA/%s/%s' % (data_path, data_obj.name, suffix[:24]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_Fedtoga(model, model_func, alpha_coef_adpt,beta,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,delta,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                delta_params_diffs[clnt] = curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            delta = -1. / n_minibatch * np.mean(delta_params_diffs[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####



            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf


def train_FedASK(data_obj, act_prob,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func, init_model, alpha_coef,beta,
                sch_step, sch_gamma, save_period,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedASK_' + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    parameter_drifts_for_means = np.zeros((n_clnt,n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_ASK/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i


                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cur_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
            for clnt in clnt_models: # 0 - n_clnt
                print('pre-training for k-means {}'.format(clnt))
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]

                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters()))) #use cloud model paramter to init local moedel
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_pre(model, model_func,
                                                              trn_x,
                                                              trn_y,
                                                              global_mdl,
                                                              learning_rate,
                                                              batch_size, epoch,
                                                              print_per,
                                                              weight_decay,
                                                              data_obj.dataset,
                                                              sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par
                parameter_drifts_for_means[clnt] = delta_param_curr #更新的参数值，一个高维向量
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10)
            reduced_updates = pca.fit_transform(parameter_drifts_for_means)

            # 直接在特征向量空间中进行聚类
            kmeans = KMeans(n_clusters=3, random_state=42).fit(reduced_updates)
            cluster_labels_m = kmeans.labels_

            print("Cluster labels for each client:")
            print(cluster_labels_m)
            unique, counts = np.unique(cluster_labels_m, return_counts=True)

            # 打印结果
            for label, count in zip(unique, counts):
                print(f"Cluster label {label}: {count}")
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(reduced_updates, cluster_labels_m)
            print(f"Silhouette Score: {silhouette_avg}")

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                selected_clusters_labels = cluster_labels_m[selected_clnts]
                count_per_cluster = Counter(selected_clusters_labels)
                print('count per cluster is {}'.format(count_per_cluster))
                all_categories = [0, 1, 2]
                for category in all_categories:
                    if category not in count_per_cluster:
                        count_per_cluster[category] = 1
                sorted_counts = sorted(count_per_cluster.items())
                print('sorted_counts is {}'.format(sorted_counts))
                count_values = [count for _, count in sorted_counts]
                count_array = np.array(count_values)
                enlarge = counts / count_array * beta
                enlarge = np.clip(enlarge, 1, 10)
                print('enlarge is {}'.format(enlarge))
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                d = enlarge[cluster_labels_m[clnt]]
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                alpha = alpha_coef / weight_list[clnt]
                parameter_drifts_curr = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                clnt_models[clnt] = train_model_FedASK(model, model_func, alpha,
                                                      global_mdl, parameter_drifts_curr,
                                                      trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                      batch_size, epoch, print_per, weight_decay, data_obj.dataset,
                                                      sch_step, sch_gamma)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                parameter_drifts[clnt] += d * delta_param_curr
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved array
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_parameter_drifts.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf


# def train_FedLESAM_D(data_obj, act_prob,
#                  learning_rate, batch_size, epoch, com_amount, print_per,
#                  weight_decay, model_func, init_model, alpha_coef,rho,
#                  sch_step, sch_gamma, save_period,
#                  suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
#     suffix = 'FedLESAM_D' + suffix
#     suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
#     save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
#     suffix += '_seed%d' % rand_seed
#     suffix += '_lrdecay%f' % lr_decay_per_round
#
#     n_clnt = data_obj.n_client
#     clnt_x = data_obj.clnt_x;
#     clnt_y = data_obj.clnt_y
#
#
#
#     weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
#     weight_list = weight_list / np.sum(weight_list) * n_clnt
#     if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
#         os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))
#
#     n_save_instances = int(com_amount / save_period)
#     avg_cld_mdls = list(range(n_save_instances))  # Cloud models
#
#     tst_cur_cld_perf = np.zeros((com_amount, 2))
#
#     n_par = len(get_mdl_params([model_func()])[0])
#
#     hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
#     cld_update_param = np.zeros(n_par).astype('float32')
#     g_update = None
#
#     init_par_list = get_mdl_params([init_model], n_par)[0]
#     clnt_update_list = np.zeros(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)
#     clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
#     clnt_models = list(range(n_clnt))
#     saved_itr = -1
#
#     # writer object is for tensorboard visualization, comment out if not needed
#     writer = SummaryWriter('%sRuns_FedLESAM_D/%s/%s' % (data_path, data_obj.name, suffix[:26]))
#
#     if not trial:
#         # Check if there are past saved iterates
#         for i in range(com_amount):
#             if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
#                               % (data_path, data_obj.name, suffix, i + 1)):
#                 saved_itr = i
#
#                 ####
#                 fed_cld = model_func()
#                 fed_cld.load_state_dict(
#                     torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
#                 fed_cld.eval()
#                 fed_cld = fed_cld.to(device)
#
#                 # Freeze model
#                 for params in fed_cld.parameters():
#                     params.requires_grad = False
#
#                 avg_cld_mdls[saved_itr // save_period] = fed_cld
#
#                 if os.path.exists(
#                         '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
#
#                     tst_cur_cld_perf[:i + 1] = np.load(
#                         '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
#
#                     # Get hist_params_diffs
#                     hist_params_diffs = np.load(
#                         '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
#                     clnt_params_list = np.load(
#                         '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))
#
#     if (trial) or (
#     not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):
#
#         if saved_itr == -1:
#
#             cur_cld_model = model_func().to(device)
#             cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
#             cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
#
#         else:
#             cur_cld_model = model_func().to(device)
#             cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
#             cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
#
#         for i in range(saved_itr + 1, com_amount):
#             # Train if doesn't exist
#             ### Fix randomness
#             inc_seed = 0
#             while (True):
#                 np.random.seed(i + rand_seed + inc_seed)
#                 act_list = np.random.uniform(size=n_clnt)
#                 act_clients = act_list <= act_prob
#                 selected_clnts = np.sort(np.where(act_clients)[0])
#                 unselected_clnts = np.sort(np.where(act_clients == False)[0])
#                 inc_seed += 1
#                 # Choose at least one client in each synch
#                 if len(selected_clnts) != 0:
#                     break
#
#             print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
#             cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
#
#             del clnt_models
#             clnt_models = list(range(n_clnt))
#
#             for clnt in selected_clnts:
#                 # Train locally
#                 print('---- Training client %d' % clnt)
#                 trn_x = clnt_x[clnt]
#                 trn_y = clnt_y[clnt]
#
#                 clnt_models[clnt] = model_func().to(device)
#
#                 model = clnt_models[clnt]
#                 # Warm start from current avg model
#                 model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
#                 for params in model.parameters():
#                     params.requires_grad = True
#                 # Scale down
#                 alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
#                 hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
#                 clnt_models[clnt] = train_model_FedLESAM_D(model, model_func, alpha_coef_adpt,
#                                                     cld_mdl_param_tensor, hist_params_diffs_curr,g_update,
#                                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
#                                                     batch_size, epoch, print_per, weight_decay,
#                                                     data_obj.dataset, sch_step, sch_gamma)
#                 curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
#                 # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
#                 # print('d is {}'.format(d))
#                 hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift
#                 clnt_params_list[clnt] = curr_model_par
#                 clnt_update_list[clnt] = (curr_model_par - cld_mdl_param)
#
#
#             avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
#
#             cld_update_param = np.mean(clnt_update_list[selected_clnts],axis=0) + np.mean(hist_params_diffs,axis=0)
#             cld_update_model = set_client_from_params(model_func().to(device),cld_update_param)
#             g_update = get_mdl_params(cld_update_model)
#
#             cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0) #1/N
#
#             cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)
#
#             #####
#
#
#
#
#             #####
#
#             loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
#                                              cur_cld_model, data_obj.dataset, 0)
#             print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
#                   % (i + 1, acc_tst, loss_tst))
#             tst_cur_cld_perf[i] = [loss_tst, acc_tst]
#
#             writer.add_scalars('Loss/test',
#                                {
#                                    'Current cloud': tst_cur_cld_perf[i][0]
#                                }, i
#                                )
#
#             writer.add_scalars('Accuracy/test',
#                                {
#                                    'Current cloud': tst_cur_cld_perf[i][1]
#                                }, i
#                                )
#
#             if (not trial) and ((i + 1) % save_period == 0):
#                 torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
#                            % (data_path, data_obj.name, suffix, (i + 1)))
#
#
#
#                 np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
#                         tst_cur_cld_perf[:i + 1])
#
#                 # save hist_params_diffs
#
#                 np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
#                         hist_params_diffs)
#                 np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
#                         clnt_params_list)
#
#                 if (i + 1) > save_period:
#                     # Delete the previous saved arrays
#
#                     os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
#                     data_path, data_obj.name, suffix, i + 1 - save_period))
#
#                     os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
#                     data_path, data_obj.name, suffix, i + 1 - save_period))
#                     os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
#                     data_path, data_obj.name, suffix, i + 1 - save_period))
#
#             if ((i + 1) % save_period == 0):
#                 avg_cld_mdls[i // save_period] = cur_cld_model
#
#     return avg_cld_mdls, tst_cur_cld_perf
#
#
#
# def train_FedLESAM(data_obj, act_prob,
#                  learning_rate, batch_size, epoch, com_amount, print_per,
#                  weight_decay, model_func, init_model, alpha_coef,rho,
#                  sch_step, sch_gamma, save_period,
#                  suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
#     suffix = 'FedLESAM' + suffix
#     suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
#     save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
#     suffix += '_seed%d' % rand_seed
#     suffix += '_lrdecay%f' % lr_decay_per_round
#
#     n_clnt = data_obj.n_client
#     clnt_x = data_obj.clnt_x;
#     clnt_y = data_obj.clnt_y
#

#
#     weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
#     weight_list = weight_list / np.sum(weight_list) * n_clnt
#     if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
#         os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))
#
#     n_save_instances = int(com_amount / save_period)
#     avg_cld_mdls = list(range(n_save_instances))  # Cloud models
#
#     tst_cur_cld_perf = np.zeros((com_amount, 2))
#
#     n_par = len(get_mdl_params([model_func()])[0])
#
#     cld_update_param = np.zeros(n_par).astype('float32')
#     g_update = None
#
#     init_par_list = get_mdl_params([init_model], n_par)[0]
#     clnt_update_list = np.zeros(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)
#     clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
#     clnt_models = list(range(n_clnt))
#     saved_itr = -1
#
#     # writer object is for tensorboard visualization, comment out if not needed
#     writer = SummaryWriter('%sRuns_FedLESAM/%s/%s' % (data_path, data_obj.name, suffix[:26]))
#
#     if not trial:
#         # Check if there are past saved iterates
#         for i in range(com_amount):
#             if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
#                               % (data_path, data_obj.name, suffix, i + 1)):
#                 saved_itr = i
#
#                 ####
#                 fed_cld = model_func()
#                 fed_cld.load_state_dict(
#                     torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
#                 fed_cld.eval()
#                 fed_cld = fed_cld.to(device)
#
#                 # Freeze model
#                 for params in fed_cld.parameters():
#                     params.requires_grad = False
#
#                 avg_cld_mdls[saved_itr // save_period] = fed_cld
#
#                 if os.path.exists(
#                         '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
#
#                     tst_cur_cld_perf[:i + 1] = np.load(
#                         '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
#
#                     # Get hist_params_diffs
#                     hist_params_diffs = np.load(
#                         '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
#                     clnt_params_list = np.load(
#                         '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))
#
#     if (trial) or (
#     not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):
#
#         if saved_itr == -1:
#
#             cur_cld_model = model_func().to(device)
#             cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
#             cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
#
#         else:
#             cur_cld_model = model_func().to(device)
#             cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
#             cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
#
#         for i in range(saved_itr + 1, com_amount):
#             # Train if doesn't exist
#             ### Fix randomness
#             inc_seed = 0
#             while (True):
#                 np.random.seed(i + rand_seed + inc_seed)
#                 act_list = np.random.uniform(size=n_clnt)
#                 act_clients = act_list <= act_prob
#                 selected_clnts = np.sort(np.where(act_clients)[0])
#                 unselected_clnts = np.sort(np.where(act_clients == False)[0])
#                 inc_seed += 1
#                 # Choose at least one client in each synch
#                 if len(selected_clnts) != 0:
#                     break
#
#             print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
#             cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
#
#             del clnt_models
#             clnt_models = list(range(n_clnt))
#
#             for clnt in selected_clnts:
#                 # Train locally
#                 print('---- Training client %d' % clnt)
#                 trn_x = clnt_x[clnt]
#                 trn_y = clnt_y[clnt]
#
#                 clnt_models[clnt] = model_func().to(device)
#
#                 model = clnt_models[clnt]
#                 # Warm start from current avg model
#                 model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
#                 for params in model.parameters():
#                     params.requires_grad = True
#                 # Scale down
#                 alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
#                 clnt_models[clnt] = train_model_FedLESAM(model, model_func, alpha_coef_adpt,
#                                                     g_update,
#                                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
#                                                     batch_size, epoch, print_per, weight_decay,
#                                                     data_obj.dataset, sch_step, sch_gamma)
#                 curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
#                 # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
#                 # print('d is {}'.format(d))
#
#                 clnt_params_list[clnt] = curr_model_par
#                 clnt_update_list[clnt] = (curr_model_par - cld_mdl_param)
#
#             avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
#
#             cld_update_param = np.mean(clnt_update_list[selected_clnts],axis=0)
#             cld_update_model = set_client_from_params(model_func().to(device),cld_update_param)
#             g_update = get_mdl_params(cld_update_model)
#
#             cld_mdl_param = avg_mdl_param_sel #1/N
#
#             cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)
#
#             #####
#
#
#
#
#             #####
#
#             loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
#                                              cur_cld_model, data_obj.dataset, 0)
#             print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
#                   % (i + 1, acc_tst, loss_tst))
#             tst_cur_cld_perf[i] = [loss_tst, acc_tst]
#
#             writer.add_scalars('Loss/test',
#                                {
#                                    'Current cloud': tst_cur_cld_perf[i][0]
#                                }, i
#                                )
#
#             writer.add_scalars('Accuracy/test',
#                                {
#                                    'Current cloud': tst_cur_cld_perf[i][1]
#                                }, i
#                                )
#
#             if (not trial) and ((i + 1) % save_period == 0):
#                 torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
#                            % (data_path, data_obj.name, suffix, (i + 1)))
#
#
#
#                 np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
#                         tst_cur_cld_perf[:i + 1])
#
#                 # save hist_params_diffs
#
#                 np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
#                         hist_params_diffs)
#                 np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
#                         clnt_params_list)
#
#                 if (i + 1) > save_period:
#                     # Delete the previous saved arrays
#
#                     os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
#                     data_path, data_obj.name, suffix, i + 1 - save_period))
#
#                     os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
#                     data_path, data_obj.name, suffix, i + 1 - save_period))
#                     os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
#                     data_path, data_obj.name, suffix, i + 1 - save_period))
#
#             if ((i + 1) % save_period == 0):
#                 avg_cld_mdls[i // save_period] = cur_cld_model
#
#     return avg_cld_mdls, tst_cur_cld_perf





def train_FedALS(data_obj, act_prob,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func, model_func1,init_model,init_model1, alpha_coef,beta,
                sch_step, sch_gamma, save_period,T,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedALS_' + str(epoch) + str(alpha_coef) + str(beta) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y




    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    n_par1 = len(get_mdl_params([model_func1()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par),dtype='float32')
    parameter_drifts_s = np.zeros((n_clnt,n_par),dtype='float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_ALS/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cur_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
            cur_cld_model1 = model_func1().to(device)
            cur_cld_model1.load_state_dict(copy.deepcopy(dict(init_model1.named_parameters())))
            cld_mdl_pre_param = get_mdl_params([cur_cld_model1],n_par1)[0]
            client_gradients = []
            normalized_sums = 0.0
            g_gradient = np.zeros_like(cld_mdl_pre_param)
            for clnt in clnt_models:  # 0 - n_clnt
                print('pre-training for gradient diversity {}'.format(clnt))
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func1().to(device)
                model = clnt_models[clnt]

                model.load_state_dict(copy.deepcopy(
                    dict(cur_cld_model1.named_parameters())))  # use cloud model paramter to init local moedel
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_gradient = compute_client_gradient(model, trn_x, trn_y, batch_size, dataset_name=data_obj.dataset)
                client_gradients.append(clnt_gradient)
                if len(client_gradients) == 50:
                    for i, grad in enumerate(client_gradients):
                        g_gradient += 1/ n_clnt * client_gradients[i]
                        grad_norm_squared = np.linalg.norm(grad, 2) ** 2
                        normalized_sums += 1/ n_clnt * weight_list[i] * grad_norm_squared
                    client_gradients.clear()

            total_gradient_norm = np.linalg.norm(g_gradient, 2) ** 2

            diversity = np.sqrt(normalized_sums / total_gradient_norm)
            print('diversity is {}'.format(diversity))
            del client_gradients

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                # clnt_models[clnt] = model_func().to(device)
                # model = clnt_models[clnt]
                model = model_func().to(device)
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                alpha = alpha_coef / weight_list[clnt]
                parameter_drifts_curr = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                if i > ( com_amount / 2 ) and T:
                    kd_T = T
                    clnt_models[clnt] = train_model_FedALS(model, model_func, cur_cld_model,alpha,
                                                      global_mdl, parameter_drifts_curr,
                                                      trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                      batch_size, epoch, kd_T,print_per, weight_decay, data_obj.dataset,
                                                      sch_step, sch_gamma)
                else:
                    clnt_models[clnt] = train_model_FedALS_wo_T(model, model_func, alpha,
                                                           global_mdl, parameter_drifts_curr,
                                                           trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                           batch_size, epoch, print_per, weight_decay,
                                                           data_obj.dataset,
                                                           sch_step, sch_gamma)
                del parameter_drifts_curr
                del model
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                parameter_drifts[clnt] += diversity * delta_param_curr
                clnt_params_list[clnt] = curr_model_par
                parameter_drifts_s[clnt] += 1 / act_prob * beta * delta_param_curr
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts_s, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)



            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )
            torch.cuda.empty_cache()

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_parameter_drifts.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf



def train_FedVARP(data_obj, act_prob, learning_rate, batch_size, n_minibatch,
                  com_amount, print_per, weight_decay,
                  model_func, init_model, sch_step, sch_gamma,
                  save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                  global_learning_rate=1):
    suffix = 'FedVARP_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_client

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y




    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt  # normalize it

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params = np.zeros((n_clnt + 1, n_par)).astype('float32')
    delta_params = np.zeros((n_clnt,n_par)).astype('float32')
    # state_params_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32')  # including cloud state
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedVARP/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):


                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1))  # Get state_params_diffs
                    state_params = np.load(
                        '%sModel/%s/%s/%d_state_params.npy' % (data_path, data_obj.name, suffix, i + 1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            del clnt_models

            clnt_models = list(range(n_clnt))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0] # global params
            global_mdl = torch.tensor(prev_params, dtype=torch.float32, device=device)  # Theta

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True

                clnt_models[clnt] = train_fedvarp_mdl(clnt_models[clnt], model_func, trn_x,
                                                       trn_y,
                                                       learning_rate * (lr_decay_per_round ** i), batch_size,
                                                       n_minibatch, print_per,
                                                       weight_decay, data_obj.dataset, sch_step, sch_gamma)

                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]  #w_i^(t,tau)
                delta_params[clnt] = (prev_params - curr_model_param) / n_minibatch / learning_rate
                clnt_params_list[clnt] = curr_model_param
            for clnt in unselected_clnts:
                delta_params[clnt] = state_params[clnt]
            delta_global_params = state_params[-1] + np.mean((delta_params[selected_clnts] - state_params[selected_clnts]), axis=0)
            avg_model_params = prev_params - learning_rate * n_minibatch * delta_global_params
            avg_model = set_client_from_params(model_func().to(device), avg_model_params)

            state_params[-1] += np.mean((delta_params - state_params[:-1]),axis=0)
            state_params[:-1] = delta_params[:]


            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        state_params)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period)):
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%d_state_params.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_A_FedPD(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,rho,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'A_FedPD_SGD_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_A_FedPD/%s/%s' % (data_path, data_obj.name, suffix[:42]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedDyn(model, model_func, alpha_coef_adpt,
                                                       cld_mdl_param_tensor, hist_params_diffs_curr,
                                                       trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, epoch, print_per, weight_decay,
                                                       data_obj.dataset, sch_step, sch_gamma)
                # clnt_models[clnt] = train_model_Fedspeed(model, model_func, alpha_coef_adpt,
                #                                     cld_mdl_param_tensor, hist_params_diffs_curr,
                #                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                #                                     batch_size, epoch, print_per, weight_decay,
                #                                     data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)   # \overline{\theta^{t+1}} = \frac{1}{P}\sum_{i \in P_t} \theta_i^{t+1}
            for clnt in unselected_clnts:
                hist_params_diffs[clnt] += avg_mdl_param_sel - cld_mdl_param
            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf





def train_FedAKL(data_obj, act_prob,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func,init_model,init_guide_model, alpha_coef,beta,
                sch_step, sch_gamma, save_period,T,feat_s_pre,feat_t_pre,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedAKL_EAM_1e-4' + str(epoch) + str(alpha_coef) + str(beta) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y




    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    # n_pr_par = len(get_mdl_params([pr_model_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par),dtype='float32')
    parameter_drifts_s = np.zeros((n_clnt,n_par),dtype='float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)# n_clnt X n_par


    feats_s_pre = [feat_s_pre for _ in range(n_clnt)]
    feats_t_pre = [feat_t_pre for _ in range(n_clnt)]

    clnt_models = list(range(n_clnt))
    clnt_guide_models = [init_guide_model for _ in range(n_clnt)]
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_AKL/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cur_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_clnt))
            clnt_guide_models = [init_guide_model for _ in range(n_clnt)]

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                # clnt_models[clnt] = model_func().to(device)
                # model = clnt_models[clnt]
                model = model_func().to(device)
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True

                alpha = alpha_coef / weight_list[clnt]
                parameter_drifts_curr = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                kd_T = T
                # tuning_modules_model = set_client_from_params(pr_model_func().to(device), clnt_pr_params_list[clnt])
                clnt_guide_models[clnt].global_model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                guide_model = clnt_guide_models[clnt]
                feat_s_pre = feats_s_pre[clnt]
                feat_t_pre = feats_t_pre[clnt] #是个字典
                aa = eAM(i,1e-4)
                bb = eAM(i,1e-4)
                print(f'aa is {aa}')
                clnt_models[clnt],clnt_guide_models[clnt],feat_s_pre,feat_t_pre = train_model_FedAKL(model, model_func, cur_cld_model,alpha,
                                                  global_mdl, parameter_drifts_curr,
                                                  trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                  batch_size, epoch, aa, bb, kd_T,print_per, weight_decay, data_obj.dataset,
                                                  sch_step, sch_gamma,feat_s_pre,feat_t_pre,guide_model)

                del parameter_drifts_curr
                del model
                del guide_model
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                parameter_drifts[clnt] += delta_param_curr
                clnt_params_list[clnt] = curr_model_par
                # parameter_drifts_s[clnt] += 1 / act_prob * beta * delta_param_curr
                feats_s_pre[clnt] = feat_s_pre
                feats_t_pre[clnt] = feat_t_pre
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)



            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )
            torch.cuda.empty_cache()

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_parameter_drifts.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_parameter_drifts.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf
def train_DPFedSMOO(data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, print_per,
                  weight_decay, model_func, init_model, alpha_coef, rho,
                  sch_step, sch_gamma, save_period,
                  suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedSMOO3_' + str(act_prob) + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_client = data_obj.n_clnt
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    ndvar_value_list= np.zeros((com_amount, 1))

    hist_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    ndvar_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_client X n_par
    mu_params_list = torch.zeros(n_client, n_par)
    global_dynamic_dual = torch.zeros(n_par)
    clnt_models = list(range(n_client))
    # Dynamic_dual_list = np.zeros(n_client,n_par).astype('float32')
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i


                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
            not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        all_clnts = torch.tensor([i for i in range(n_client)])
        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_client))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                Dynamic_dual = get_params_list_with_shape(model, mu_params_list[clnt], device)
                Dynamic_dual_correction = get_params_list_with_shape(model, global_dynamic_dual, device)
                clnt_models[clnt],dynamic_dual = train_model_DPFedsmoo(model, model_func, alpha_coef_adpt, Dynamic_dual,
                                                        Dynamic_dual_correction,
                                                        cld_mdl_param_tensor, hist_params_diffs_curr,
                                                        trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), rho,
                                                        batch_size, epoch, print_per, weight_decay,
                                                        data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift

                ndvar_params_diffs[clnt] = (curr_model_par - cld_mdl_param)
                local_dynamical_dual = dynamic_dual

                clnt_params_list[clnt] = curr_model_par

                mu = []
                for _mu_ in local_dynamical_dual:
                        mu.append(_mu_.clone().detach().cpu().reshape(-1))
                mu_params_list[clnt] = torch.cat(mu)
            #
            # h = 0
            # for clnt in all_clnts:
            #     h += np.sum((clnt_params_list[clnt] - cld_mdl_param) * (clnt_params_list[clnt] - cld_mdl_param))
            # h /= n_client
            # print('h is {}'.format(h))



            avg_dynamic_dual = torch.mean(mu_params_list[selected_clnts], dim=0)
            _l2_ = torch.norm(avg_dynamic_dual, p=2, dim=0) + 1e-7
            global_dynamic_dual = avg_dynamic_dual / _l2_ * rho

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)  # 1/N

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####




            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):

                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):

                avg_cld_mdls[i // save_period] = cur_cld_model



    return avg_cld_mdls,  tst_cur_cld_perf
def train_DPFedAvg(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedAvg_' + str(epoch) + suffix
    suffix += '_c%d_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    com_amount,save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    ndvar_value_list= np.zeros((com_amount, 1))
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_deltas = np.zeros((n_clnt, n_par))  # 存储 delta_i
    clnt_mu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 ut (mu_t)
    clnt_nu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 vt (nu_t)

    # 初始化全局动量/方差变量（ut/vt）
    mu_global = torch.zeros(n_par).to(device)
    nu_global = torch.ones(n_par).to(device) * 1e-10

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-New/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                cld_mdl_param=avg_model.named_parameters()

                clnt_models[clnt].load_state_dict(dict(cld_mdl_param))

                cld_mdl_param = get_mdl_params([avg_model], n_par)[0]

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPavg(clnt_models[clnt], clnt_num,trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
  
    return fed_mdls_sel, tst_perf_sel


def train_DPFedGloss(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                 lamb=1e-3, rho_l=0.1, T_s=100, esam_rho=0.5):
    """
    DP-FedGloss under DPFed framework.
    - Local training: ESAM + linear correction (FedGloss)
    - DP on uploads: DPFedAvg-style per-layer clipping/noise
    - Server update: w_{t+1} = w_t + avg(delta) + mean(h)
    - Perturbed broadcast (FedGloss): w_comm = w_t + epsilon (epsilon uses last avg update as pseudo-grad)
    """
    suffix = 'DPFedGloss_' + str(epoch) + suffix
    suffix += '_c%d_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
        com_amount, save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)
    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)]).reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    tst_perf_sel = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)

    # Dual variables per client (FedGloss)
    h_params_list = np.zeros((n_clnt, n_par), dtype='float32')

    # Server pseudo-grad state for perturbation
    delta_w_tilde_vec = None  # numpy vector (n_par,)

    saved_itr = -1
    writer = SummaryWriter('%sRuns_DPFedGloss/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    # Load resume state if any
    if not trial:
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval(); fed_model = fed_model.to(device)
                for p in fed_model.parameters(): p.requires_grad = False
                fed_mdls_sel[saved_itr // save_period] = fed_model
                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_perf_sel[:i + 1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))
                    # Optional: load h_params_list if saved
                    h_path = '%sModel/%s/%s/%d_h_params_list.npy' % (data_path, data_obj.name, suffix, i + 1)
                    if os.path.exists(h_path):
                        h_params_list = np.load(h_path)

    # Training rounds
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Select clients
            inc_seed = 0
            while True:
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            # Compute broadcast params: w_comm = w_t + epsilon
            w_t_vec = get_mdl_params([avg_model], n_par)[0]
            if delta_w_tilde_vec is not None:
                # rho schedule
                rho = 0.001 + min(i + 1, T_s) * ((rho_l - 0.001) / max(1, T_s))
                denom = np.linalg.norm(delta_w_tilde_vec)
                if denom > 0:
                    eps_vec = (rho / denom) * delta_w_tilde_vec
                else:
                    eps_vec = np.zeros_like(w_t_vec)
            else:
                eps_vec = np.zeros_like(w_t_vec)

            w_comm_vec = w_t_vec + eps_vec

            # Train selected clients
            clnt_models = list(range(n_clnt))
            # cache local updates for selected clients
            local_updates = {}

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]; trn_y = clnt_y[clnt]

                # Init client model from w_comm
                clnt_models[clnt] = set_client_from_params(model_func().to(device), w_comm_vec)
                for p in clnt_models[clnt].parameters(): p.requires_grad = True

                # Local dual correction vector: h_i - w_comm
                corr_vec = h_params_list[clnt] - w_comm_vec

                # Train with ESAM + correction, then DP protect updates
                # clnt_num = len(selected_clnts)
                clnt_num = 50
                clnt_models[clnt] = train_model_DPgloss(
                    clnt_models[clnt], clnt_num, trn_x, trn_y, False, False,
                    learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                    weight_decay, data_obj.dataset, sch_step, sch_gamma,
                    local_dual_correction_vec=corr_vec, lamb=lamb, esam_rho=esam_rho)

                # Save client params and update
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                local_updates[clnt] = clnt_params_list[clnt] - w_comm_vec

            # Compute averaged model and update server
            # Weighted average of client models (same as DPFedAvg)
            avg_local_model_vec = np.sum(clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]), axis=0)
            mean_h = np.mean(h_params_list, axis=0)

            # w_{t+1} = avg_local_model - epsilon + mean(h)
            new_w_vec = avg_local_model_vec - eps_vec + mean_h
            avg_model = set_client_from_params(model_func(), new_w_vec)

            # Pseudo-grad for next round's epsilon: averaged update
            if len(local_updates) > 0:
                # Equal average of local updates
                upd_mat = np.stack([local_updates[c] for c in local_updates.keys()], axis=0)
                delta_w_tilde_vec = np.mean(upd_mat, axis=0)
            else:
                delta_w_tilde_vec = np.zeros_like(w_t_vec)

            # Update dual variables for selected clients
            for clnt in selected_clnts:
                h_params_list[clnt] += local_updates[clnt]

            # Evaluate
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/test', {'Sel clients': tst_perf_sel[i][0]}, i)
            writer.add_scalars('Accuracy/test', {'Sel clients': tst_perf_sel[i][1]}, i)

            # Freeze and optionally save
            for p in avg_model.parameters(): p.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, (i + 1)))
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)), tst_perf_sel[:i + 1])
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)), clnt_params_list)
                np.save('%sModel/%s/%s/%d_h_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)), h_params_list)
                if (i + 1) > save_period:
                    for name in ['com_tst_perf_sel', 'clnt_params_list', 'h_params_list']:
                        old_path = '%sModel/%s/%s/%d%s.npy' % (data_path, data_obj.name, suffix, i + 1 - save_period, '_' + name)
                        if os.path.exists(old_path):
                            os.remove(old_path)

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_DPFedDyn(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                 lamb=1e-2):
    """
    DP-FedDyn under DPFed framework.
    - Local: FedDyn linear correction with coefficient lamb
    - DP: DPFedAvg-style per-layer clipping/noise on model deltas
    - Server: w_{t+1} = Averaged_model + mean(h)
    - Dual update: h_i += delta_i
    """
    suffix = 'DPFedDyn_' + str(epoch) + suffix
    suffix += '_c%d_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
        com_amount, save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)
    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)]).reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    tst_perf_sel = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)

    # Dual variables per client
    h_params_list = np.zeros((n_clnt, n_par), dtype='float32')

    saved_itr = -1
    writer = SummaryWriter('%sRuns_DPFedDyn/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval(); fed_model = fed_model.to(device)
                for p in fed_model.parameters(): p.requires_grad = False
                fed_mdls_sel[saved_itr // save_period] = fed_model
                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_perf_sel[:i + 1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))
                    h_path = '%sModel/%s/%s/%d_h_params_list.npy' % (data_path, data_obj.name, suffix, i + 1)
                    if os.path.exists(h_path):
                        h_params_list = np.load(h_path)

    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Select clients
            inc_seed = 0
            while True:
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            # Broadcast current global params (no perturbation)
            w_t_vec = get_mdl_params([avg_model], n_par)[0]

            clnt_models = list(range(n_clnt))
            local_updates = {}

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]; trn_y = clnt_y[clnt]

                # Init client model from w_t
                clnt_models[clnt] = set_client_from_params(model_func().to(device), w_t_vec)
                for p in clnt_models[clnt].parameters(): p.requires_grad = True

                # Local dual correction vector: h_i - w_t
                corr_vec = h_params_list[clnt] - w_t_vec

                # Train then DP-protect
                clnt_num = len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedDyn(
                    clnt_models[clnt], clnt_num, trn_x, trn_y, False, False,
                    learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                    weight_decay, data_obj.dataset, sch_step, sch_gamma,
                    local_dual_correction_vec=corr_vec, lamb=lamb)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                local_updates[clnt] = clnt_params_list[clnt] - w_t_vec

            # Averaged model
            avg_local_model_vec = np.sum(clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]), axis=0)
            mean_h = np.mean(h_params_list, axis=0)
            new_w_vec = avg_local_model_vec + mean_h
            avg_model = set_client_from_params(model_func(), new_w_vec)

            # Update duals
            for clnt in selected_clnts:
                h_params_list[clnt] += local_updates[clnt]

            # Evaluate
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" % (i + 1, acc_tst, loss_tst))
            writer.add_scalars('Loss/test', {'Sel clients': tst_perf_sel[i][0]}, i)
            writer.add_scalars('Accuracy/test', {'Sel clients': tst_perf_sel[i][1]}, i)

            # Save
            for p in avg_model.parameters(): p.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, (i + 1)))
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)), tst_perf_sel[:i + 1])
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)), clnt_params_list)
                np.save('%sModel/%s/%s/%d_h_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)), h_params_list)
                if (i + 1) > save_period:
                    for name in ['com_tst_perf_sel', 'clnt_params_list', 'h_params_list']:
                        old_path = '%sModel/%s/%s/%d%s.npy' % (data_path, data_obj.name, suffix, i + 1 - save_period, '_' + name)
                        if os.path.exists(old_path):
                            os.remove(old_path)

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel
def train_DPFedSMP_topk(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedSMP_topk_' + str(epoch) + '_'+str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_deltas = np.zeros((n_clnt, n_par))  # 存储 delta_i
    clnt_mu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 ut (mu_t)
    clnt_nu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 vt (nu_t)

    # 初始化全局动量/方差变量（ut/vt）
    mu_global = torch.zeros(n_par).to(device)
    nu_global = torch.ones(n_par).to(device) * 1e-10

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sDPRuns_DPFedFER-NEW/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                cld_mdl_param=avg_model.named_parameters()

                clnt_models[clnt].load_state_dict(dict(cld_mdl_param))

                cld_mdl_param = get_mdl_params([avg_model], n_par)[0]

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedSMP_topk(clnt_models[clnt], clnt_num,trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model


    return fed_mdls_sel, tst_perf_sel


def train_DPFedAvg_blurs(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedAvg_blur_130' + str(epoch) +'_'+str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_deltas = np.zeros((n_clnt, n_par))  # 存储 delta_i
    clnt_mu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 ut (mu_t)
    clnt_nu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 vt (nu_t)

    # 初始化全局动量/方差变量（ut/vt）
    mu_global = torch.zeros(n_par).to(device)
    nu_global = torch.ones(n_par).to(device) * 1e-10

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-blur/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                cld_mdl_param=avg_model.named_parameters()

                clnt_models[clnt].load_state_dict(dict(cld_mdl_param))

                cld_mdl_param = get_mdl_params([avg_model], n_par)[0]

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPavg_blurs(clnt_models[clnt], clnt_num,trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel

def train_DPFedAvg_blur(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedAvg_blur_' + str(epoch) +'_'+ str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_deltas = np.zeros((n_clnt, n_par))  # 存储 delta_i
    clnt_mu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 ut (mu_t)
    clnt_nu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 vt (nu_t)

    # 初始化全局动量/方差变量（ut/vt）
    mu_global = torch.zeros(n_par).to(device)
    nu_global = torch.ones(n_par).to(device) * 1e-10

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-New/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                cld_mdl_param=avg_model.named_parameters()

                clnt_models[clnt].load_state_dict(dict(cld_mdl_param))

                cld_mdl_param = get_mdl_params([avg_model], n_par)[0]

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPavg_blur(clnt_models[clnt], clnt_num,trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_DPSCAFFOLD(data_obj, act_prob, learning_rate, batch_size, n_minibatch,
                   com_amount, print_per, weight_decay,
                   model_func, init_model, sch_step, sch_gamma,
                   save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                   global_learning_rate=1):
    suffix = 'DPScaffold_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_client = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client  # normalize it

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    epsilon_sel=np.zeros((n_client, 1))
    epsilon_list = np.zeros((com_amount, 1))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params_diffs = np.zeros((n_client + 1, n_par)).astype('float32')  # including cloud state
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                  -1)  # n_client X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-NEW/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1))  # Get state_params_diffs
                    state_params_diffs = np.load(
                        '%sModel/%s/%s/%d_state_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_client))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            del clnt_models

            clnt_models = list(range(n_client))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True

                # Scale down c
                state_params_diff_curr = torch.tensor(
                    -state_params_diffs[clnt] + state_params_diffs[-1] / weight_list[clnt], dtype=torch.float32,
                    device=device)

                clnt_models[clnt] = train_DPscaffold_mdl(clnt_models[clnt],model_func, state_params_diff_curr, trn_x,
                                                       trn_y,
                                                       learning_rate * (lr_decay_per_round ** i), batch_size,
                                                       n_minibatch, print_per,
                                                       weight_decay, data_obj.dataset, sch_step, sch_gamma)

                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_params_diffs[clnt] - state_params_diffs[-1] + 1 / n_minibatch / learning_rate * (
                            prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_params_diffs[clnt]) * weight_list[clnt]
                state_params_diffs[clnt] = new_c

                clnt_params_list[clnt] = curr_model_param

            epsilon_list[i]=np.mean(epsilon_sel[selected_clnts])

            avg_model_params = global_learning_rate * np.mean(clnt_params_list[selected_clnts], axis=0) + (
                        1 - global_learning_rate) * prev_params

            avg_model = set_client_from_params(model_func().to(device), avg_model_params)

            state_params_diffs[-1] += 1 / n_client * delta_c_sum

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1]
                               }, i
                               )


            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])



                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        state_params_diffs)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period)):
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%d_state_params_diffs.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_global_model_cgan_bs_weight_iloop_fz_dis(model_func,s_model, g_model, t_model, client_params,
                                                   clnt_cls_num, glb_lr, gen_lr,
                                                   batch_size, print_per, weight_decay,
                                                   dataset_name, trn_x, trn_y, tst_x, tst_y, i, suffix, data_obj,
                                                   data_path,
                                                   ewc_lambda, his_lambda, hist_gan_diffs):
    t_model.eval()
    s_model.to(device)
    t_model.to(device)
    g_model.to(device)

    init_model = g_model.to(device)

    num_clients, num_classes = clnt_cls_num.shape
    optimizer_D = torch.optim.SGD(s_model.parameters(), lr=glb_lr, weight_decay=weight_decay)
    optimizer_G = torch.optim.Adam(g_model.parameters(), lr=gen_lr)
    iterations = 10
    inner_round_g = 1
    inner_round_d = 5
    nz = 100

    for params in s_model.parameters():
        params.requires_grad = True

    cls_num = np.sum(clnt_cls_num, axis=0)
    cls_clnt_weight = clnt_cls_num / (np.tile(cls_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    cls_clnt_weight = cls_clnt_weight.transpose()
    labels_all = generate_labels(iterations * batch_size, cls_num)

    for e in range(iterations):

        labels = labels_all[e * batch_size:(e * batch_size + batch_size)]
        batch_weight = torch.Tensor(get_batch_weight(labels, cls_clnt_weight)).to(device)
        onehot = np.zeros((batch_size, num_classes))
        onehot[np.arange(batch_size), labels] = 1
        y_onehot = torch.Tensor(onehot).to(device)
        z = torch.randn((batch_size, nz, 1, 1)).to(device)

        ############## train generator ##############
        s_model.eval()
        g_model.train()
        loss_G = 0
        loss_tran_total = 0
        loss_cls_total = 0
        loss_div_total = 0
        ewc_value_total = 0
        for _ in range(inner_round_g):
            for client in range(num_clients):
                optimizer_G.zero_grad()
                t_model = set_client_from_params(t_model, client_params[client])
                loss, loss_tran, loss_cls, loss_div, ewc_value = train_generator(z, y_onehot, labels, g_model,
                                                                                 s_model, t_model,
                                                                                 batch_weight[:, client], num_clients,
                                                                                 i, suffix, data_obj, data_path,
                                                                                 ewc_lambda)
                loss_G += loss
                loss_tran_total += loss_tran
                loss_cls_total += loss_cls
                loss_div_total += loss_div
                ewc_value_total += ewc_value
                optimizer_G.step()

        if i > 1:
            g_model_pre = init_model
            g_model_pre.load_state_dict(torch.load('%sModel/%s/%s/%dcom_G.pt'
                                                   % (data_path, data_obj.name, suffix, i - 1)))
            idx_nonbn_gan = get_mdl_nonbn_idx([init_model])[0]
            n_par_gan = len(get_mdl_params([init_model])[0])
            gan_pre_param = get_mdl_params([g_model_pre], n_par_gan)[0]

            g_param = get_mdl_params([g_model], n_par_gan)[0]
            hist_gan_diffs += g_param - gan_pre_param
            # g_param = update_cloud_model(g_param, hist_gan_diffs, model_func)
            g_param[idx_nonbn_gan] = g_param[idx_nonbn_gan] + hist_gan_diffs[idx_nonbn_gan]
            # g_param= g_param+ hist_gan_diffs
            g_model_his = init_model
            g_model_his = set_client_from_params(g_model_his, g_param)

        ############## train student model ##############
        s_model.train()
        g_model.eval()

        if i > 1:
            s_model.train()
            g_model.eval()

            for _ in range(inner_round_d):
                optimizer_D.zero_grad()
                fake = g_model(z, y_onehot).detach()
                fake_his = g_model_his(z, y_onehot).detach()
                s_logit = s_model(fake)
                s_logit_his = s_model(fake_his)
                t_logit_merge = 0
                t_logit_merge_his = 0
                for client in range(num_clients):
                    t_model = set_client_from_params(t_model, client_params[client])
                    t_logit = t_model(fake).detach()
                    t_logit_his = t_model(fake_his).detach()
                    t_logit_merge += F.softmax(t_logit, dim=1) * batch_weight[:, client][:, np.newaxis].repeat(1,
                                                                                                               num_classes)
                    t_logit_merge_his += F.softmax(t_logit_his, dim=1) * batch_weight[:, client][:, np.newaxis].repeat(
                        1, num_classes)
                loss_kd = torch.mean(-F.log_softmax(s_logit, dim=1) * t_logit_merge) + his_lambda * torch.mean(
                    -F.log_softmax(s_logit_his, dim=1) * t_logit_merge_his)
                loss_kd.backward()
                optimizer_D.step()
        else:
            for _ in range(inner_round_d):
                optimizer_D.zero_grad()
                fake = g_model(z, y_onehot).detach()
                s_logit = s_model(fake)
                t_logit_merge = 0
                for client in range(num_clients):
                    t_model = set_client_from_params(t_model, client_params[client])
                    t_logit = t_model(fake).detach()
                    t_logit_merge += F.softmax(t_logit, dim=1) * batch_weight[:, client][:, np.newaxis].repeat(1,
                                                                                                               num_classes)

                loss_ks = torch.mean(-F.log_softmax(s_logit, dim=1) * t_logit_merge)
                loss_ks.backward()
                optimizer_D.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss_FedFLD(trn_x, trn_y, s_model, dataset_name)
            loss_tst, acc_tst = get_acc_loss_FedFLD(tst_x, tst_y, s_model, dataset_name)
            s_model.train()

    # Freeze model
    for params in s_model.parameters():
        params.requires_grad = False
    s_model.eval()

    return s_model, hist_gan_diffs


def update_cloud_model(cld_mdl_param, hist_avg_diffs, model_func):
    model = model_func()
    idx_nonbn = get_non_bn_parameter_indices(model)
    avg_diffs = np.mean(hist_avg_diffs, axis=0)

    # 确保索引有效
    valid_indices = idx_nonbn[(idx_nonbn < len(cld_mdl_param)) & (idx_nonbn < len(avg_diffs))]

    if len(valid_indices) > 0:
        cld_mdl_param[valid_indices] += avg_diffs[valid_indices]

    return cld_mdl_param
def get_non_bn_parameter_indices(model):
    """获取非BN层参数的全局索引"""
    indices = []
    current_idx = 0

    state_dict = model.state_dict()

    for name, param in state_dict.items():
        param_size = param.numel()

        # 判断是否为BN层参数
        is_bn_param = (
                'bn' in name or
                'batchnorm' in name or
                ('downsample' in name and '1.' in name) or
                'running_mean' in name or
                'running_var' in name or
                'num_batches_tracked' in name
        )

        # 只包含非BN层的可训练参数
        if not is_bn_param and param.requires_grad:
            indices.extend(range(current_idx, current_idx + param_size))

        current_idx += param_size

    return np.array(indices)
def get_mdl_nonbn_idx(model_list, name_par=None):
    if name_par is None:
        exp_mdl = model_list[0]
        name_par = []
        for name, param in exp_mdl.named_parameters():
            name_par.append(name)

    idx_list = [[]] * len(model_list)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            if name in name_par:
                idx_list[i].extend(list(range(idx, idx + len(temp))))
            idx += len(temp)
    return np.array(idx_list)


def train_FedGAN_common(data_obj, act_prob, learning_rate, batch_size, epoch,
                        com_amount, print_per, weight_decay,
                        model_func, init_model, init_g_model, sch_step, sch_gamma,
                        save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                        ewc_lambda=1.0, his_lambda=0.0):
    glb_model_lr = 0.1
    gen_model_lr = 0.01
    init_g_model = init_g_model.to(device)
    clnt_cls_num = get_class_number(data_obj)

    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_e%f_h%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, ewc_lambda,
        his_lambda)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))
    fed_mdls_ft = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2))
    trn_perf_all = np.zeros((com_amount, 2))
    trn_perf_ft = np.zeros((com_amount, 2))

    tst_perf_sel = np.zeros((com_amount, 2))
    tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_ft = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    # 获取非BN层索引
    idx_nonbn = get_mdl_nonbn_idx([model_func()])[0]

    # idx_nonbn = get_non_bn_parameter_indices(model_func())



    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    hist_avg_diffs = np.zeros((n_clnt, n_par)).astype('float32')

    n_par_gan = len(get_mdl_params([init_g_model])[0])

    hist_gan_diffs = np.zeros((n_par_gan,)).astype('float32')

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_ft[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_ft[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

                    if i > 2:
                        hist_avg_diffs = np.load(
                            '%sModel/%s/%s/%d_hist_avg_diffs.npy' % (data_path, data_obj.name, suffix, i))
                        hist_gan_diffs = np.load(
                            '%sModel/%s/%s/%d_hist_gan_diffs.npy' % (data_path, data_obj.name, suffix, i))

    if trial or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        # clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            avg_model_ft = model_func().to(device)
            avg_model_ft.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
            ft_mdl_param = get_mdl_params([avg_model_ft], n_par)[0]

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            avg_model_ft = model_func().to(device)
            avg_model_ft.load_state_dict(torch.load('%sModel/%s/%s/%dcom_ft.pt'
                                                    % (data_path, data_obj.name, suffix, (saved_itr + 1))))
            ft_mdl_param = get_mdl_params([avg_model_ft], n_par)[0]

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_model.state_dict())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            # Fix randomness
            inc_seed = 0
            while True:
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            # del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model_ft.state_dict())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model_FedFLD(clnt_models[clnt], trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                hist_avg_diffs[clnt] += curr_model_par - ft_mdl_param
                clnt_params_list[clnt] = curr_model_par

            # Scale with weights

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel.copy()
            cld_mdl_param[idx_nonbn] = cld_mdl_param[idx_nonbn] + np.mean(hist_avg_diffs, axis=0)[idx_nonbn]  # 更新
            # cld_mdl_param = cld_mdl_param + np.mean(hist_avg_diffs, axis=0) # 更新
            # cld_mdl_param = update_cloud_model(cld_mdl_param, hist_avg_diffs, model_func)
            avg_model = set_client_from_params(model_func(), np.sum(
                clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                axis=0))
            all_model = set_client_from_params(model_func(),
                                               np.sum(clnt_params_list * weight_list / np.sum(weight_list), axis=0))

            trn_x_sel = np.concatenate(clnt_x[selected_clnts], axis=0)
            trn_y_sel = np.concatenate(clnt_y[selected_clnts], axis=0)
            avg_model_ft = set_client_from_params(model_func().to(device), cld_mdl_param)
            avg_model_ft, hist_gan_diffs = train_global_model_cgan_bs_weight_iloop_fz_dis(model_func,avg_model_ft, init_g_model,
                                                                                          model_func(),
                                                                                          clnt_params_list[
                                                                                              selected_clnts],
                                                                                          clnt_cls_num[selected_clnts],
                                                                                          glb_model_lr * (
                                                                                                      lr_decay_per_round ** i),
                                                                                          gen_model_lr * (
                                                                                                      lr_decay_per_round ** i),
                                                                                          batch_size, print_per,
                                                                                          weight_decay,
                                                                                          data_obj.dataset,
                                                                                          trn_x_sel, trn_y_sel,
                                                                                          data_obj.tst_x,
                                                                                          data_obj.tst_y, i, suffix,
                                                                                          data_obj, data_path,
                                                                                          ewc_lambda, his_lambda,
                                                                                          hist_gan_diffs)

            ft_mdl_param = get_mdl_params([avg_model_ft], n_par)[0]
            loss_tst, acc_tst = get_acc_loss_FedFLD(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]


            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )


            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(avg_model_ft.state_dict(), '%sModel/%s/%s/%dcom_ft.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                torch.save(init_g_model.state_dict(), '%sModel/%s/%s/%dcom_G.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_sel[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_all[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_all[:i + 1])

                np.save('%sModel/%s/%s/%dcom_trn_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_ft[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_ft[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)
                if i > 1:
                    np.save('%sModel/%s/%s/%d_hist_avg_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                            hist_avg_diffs)
                    np.save('%sModel/%s/%s/%d_hist_gan_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                            hist_gan_diffs)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%dcom_trn_perf_ft.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_ft.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        if i > 3:
                            os.remove('%sModel/%s/%s/%dcom_sel.pt'
                                      % (data_path, data_obj.name, suffix, i - save_period))
                            os.remove('%sModel/%s/%s/%dcom_all.pt'
                                      % (data_path, data_obj.name, suffix, i - save_period))
                            os.remove('%sModel/%s/%s/%dcom_ft.pt'
                                      % (data_path, data_obj.name, suffix, i - save_period))
                            os.remove('%sModel/%s/%s/%dcom_G.pt'
                                      % (data_path, data_obj.name, suffix, i - save_period))
                            os.remove('%sModel/%s/%s/%d_hist_avg_diffs.npy' % (
                                data_path, data_obj.name, suffix, i - save_period))
                            os.remove('%sModel/%s/%s/%d_hist_gan_diffs.npy' % (
                                data_path, data_obj.name, suffix, i - save_period))

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
                fed_mdls_all[i // save_period] = all_model
                fed_mdls_ft[i // save_period] = avg_model_ft

            torch.cuda.empty_cache()

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, \
        fed_mdls_all, trn_perf_all, tst_perf_all, \
        fed_mdls_ft, trn_perf_ft, tst_perf_ft


def train_FedAdan_topk(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedAdan_topk_' + str(epoch) + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par


    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-New/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedAdan_topk(clnt_models[clnt], clnt_num,trn_x, trn_y,
                                                      tst_x, tst_y,
                                                      learning_rate * (lr_decay_per_round ** i), batch_size, epoch,
                                                      print_per,
                                                      weight_decay,
                                                      data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_FedAdan(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedAdan_' +str(epoch)+  str(learning_rate)+'_'+ str(act_prob) + suffix
    suffix += '_c%d_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    com_amount,save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par


    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedAdan_epoch/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedAdan(clnt_models[clnt], clnt_num,trn_x, trn_y,
                                                      tst_x, tst_y,
                                                      learning_rate * (lr_decay_per_round ** i), batch_size, epoch,
                                                      print_per,
                                                      weight_decay,
                                                      data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel
def train_FedAdan_SWAT(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedAdan_SWAT_' + str(act_prob) + suffix
    suffix += '_c%d_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    com_amount,save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par


    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-SWAT/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))


    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False


                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedAdan_SWAT(clnt_models[clnt], clnt_num,trn_x, trn_y,
                                                      tst_x, tst_y,
                                                      learning_rate * (lr_decay_per_round ** i), batch_size, epoch,
                                                      print_per,
                                                      weight_decay,
                                                      data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_tst, loss_tst))



            writer.add_scalars('Loss/test',
                   {
                       'Sel clients':tst_perf_sel[i][0]
                   }, i
                  )

            writer.add_scalars('Accuracy/test',
                   {
                       'Sel clients':tst_perf_sel[i][1]
                   }, i
                  )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                               %(data_path, data_obj.name, suffix, (i+1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)

                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel


def train_FedGKD(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedGKD-adan_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedGkd(model, model_func, cur_cld_model,alpha_coef_adpt,
                                                       cld_mdl_param_tensor, hist_params_diffs_curr,
                                                       trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, epoch, print_per, weight_decay,
                                                       data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par



            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf
def train_DPFER(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFER_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_DPFER(model, model_func, cur_cld_model,alpha_coef_adpt,
                                                       cld_mdl_param_tensor, hist_params_diffs_curr,
                                                       trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                       batch_size, epoch, print_per, weight_decay,
                                                       data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par - cld_mdl_param
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls, tst_cur_cld_perf

def train_FedFC(data_obj, act_prob,
                 learning_rate, batch_size, epoch, beta,com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,rho,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'rFedFC_b=0.05_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y




    dual_steps = np.asarray([1 for _ in range(n_clnt)])

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%srRuns_FedFC/%s/%s' % (data_path, data_obj.name, suffix[:42]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            print(f"current dual step list is {dual_steps}")
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                dual_steps[unselected_clnts] += 1
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                # clnt_models[clnt] = train_model_FedFC(model, model_func, alpha_coef_adpt,
                #                                        cld_mdl_param_tensor, hist_params_diffs_curr,
                #                                        trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                #                                        batch_size, epoch, print_per, weight_decay,
                #                                        data_obj.dataset, sch_step, sch_gamma)
                clnt_models[clnt] = train_model_FedALS_wo_T(model, model_func, alpha_coef_adpt,
                                                         cld_mdl_param_tensor, hist_params_diffs_curr,
                                                         trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                         batch_size, epoch, print_per, weight_decay,
                                                         data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                print(f'before step is {dual_steps[clnt]}')
                if dual_steps[clnt] <= (1 / act_prob):
                    hist_params_diffs[clnt] += (1 + beta * dual_steps[clnt]) * (curr_model_par - cld_mdl_param)
                    dual_steps[clnt] = 1
                else:
                    hist_params_diffs[clnt] += (1 + 1 / act_prob * beta) * (curr_model_par - cld_mdl_param)
                    dual_steps[clnt] -= (1 / act_prob)
                clnt_params_list[clnt] = curr_model_par
                print(f'after step is {dual_steps[clnt]}')
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model


    return avg_cld_mdls,  tst_cur_cld_perf




def train_FedCM(data_obj, act_prob,
                 learning_rate, batch_size, epoch, n_minibatch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedCM_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round



    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    if not os.path.exists('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8])):
        os.mkdir('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8]))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models


    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    client_momentum = np.zeros(n_par).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedCM/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                delta_list = torch.tensor(client_momentum).to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                clnt_models[clnt] = train_model_FedCM(model, model_func, alpha_coef_adpt,
                                                    delta_list,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            avg_update = avg_mdl_param_sel - cld_mdl_param
            client_momentum = avg_update / n_minibatch / learning_rate * lr_decay_per_round * -1.

            cld_mdl_param = avg_mdl_param_sel

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays

                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model
    last_100_acc_tst = [tst_cur_cld_perf[i][1] for i in range(com_amount - 100, com_amount)]
    mean_acc_tst = np.mean(last_100_acc_tst) * 100
    var_acc_tst = np.var(last_100_acc_tst) * 100
    with open('%sAccuracy/%s%s/results.txt' % (data_path,data_obj.name,suffix[:8]), 'w') as f:
        f.write(f"{mean_acc_tst} ± {var_acc_tst}\n")
    return avg_cld_mdls,  tst_cur_cld_perf




def train_FedGamma(data_obj, act_prob, learning_rate, rho,batch_size, n_minibatch,
                   com_amount, print_per, weight_decay,
                   model_func, init_model, sch_step, sch_gamma,
                   save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                   global_learning_rate=1):
    suffix = 'FedGamma_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_client = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client  # normalize it

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))
    # if not os.path.exists('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8])):
    #     os.mkdir('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8]))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    ndvar_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    ndvar_value_list= np.zeros((com_amount, 1))
    state_params_diffs = np.zeros((n_client + 1, n_par)).astype('float32')  # including cloud state
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_client X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedSGAM_KL/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1))  # Get state_params_diffs
                    state_params_diffs = np.load(
                        '%sModel/%s/%s/%d_state_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_client))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))
            cld_mdl_param = get_mdl_params([avg_model], n_par)[0]
        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            del clnt_models

            clnt_models = list(range(n_client))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True

                # Scale down c
                state_params_diff_curr = torch.tensor(
                    -state_params_diffs[clnt] + state_params_diffs[-1] / weight_list[clnt], dtype=torch.float32,
                    device=device)

                clnt_models[clnt] = train_model_Fedgamma(clnt_models[clnt], model_func, state_params_diff_curr, trn_x,
                                                       trn_y,
                                                       learning_rate * (lr_decay_per_round ** i),rho, batch_size,
                                                       n_minibatch, print_per,
                                                       weight_decay, data_obj.dataset, sch_step, sch_gamma)

                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_params_diffs[clnt] - state_params_diffs[-1] + 1 / n_minibatch / learning_rate * (
                            prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_params_diffs[clnt]) * weight_list[clnt]
                state_params_diffs[clnt] = new_c

                ndvar_params_diffs[clnt] = (curr_model_param - cld_mdl_param)

                clnt_params_list[clnt] = curr_model_param

            variance_value = compute_ndvar(clnt_params_list, cld_mdl_param, selected_clnts)
            print(f"variance_value: {variance_value}")
            ndvar_value_list[i] = variance_value

            avg_model_params = global_learning_rate * np.mean(clnt_params_list[selected_clnts], axis=0) + (
                        1 - global_learning_rate) * prev_params

            cld_mdl_param = avg_model_params

            avg_model = set_client_from_params(model_func().to(device), avg_model_params)

            state_params_diffs[-1] += 1 / n_client * delta_c_sum

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        state_params_diffs)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period)):
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%d_state_params_diffs.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))
            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
    last_100_acc_tst = [tst_perf_sel[i][1] for i in range(com_amount - 100, com_amount)]
    mean_acc_tst = np.mean(last_100_acc_tst) * 100
    var_acc_tst = np.var(last_100_acc_tst) * 100
    # with open('%sAccuracy/%s%s/results.txt' % (data_path, data_obj.name,suffix[:8]), 'w') as f:
    #     f.write(f"{mean_acc_tst} ± {var_acc_tst}\n")

    df = pd.DataFrame(ndvar_value_list)
    df.to_csv("ndvar_values_FedGamma_iid.csv", index=False)

    return fed_mdls_sel, tst_perf_sel



def subtract(model1, model2):
    """
    计算两个模型参数的差值 (model1 - model2)
    """
    # 获取模型参数的状态字典
    params1 = model1.state_dict()
    params2 = model2.state_dict()

    # 计算差值
    delta = {}
    for key in params1:
        delta[key] = params1[key] - params2[key]

    return delta


def add(params_a, params_b):
    """
    计算两个模型/参数字典的和 (params_a + params_b)
    支持:
        - 两个都是模型的 state_dict (字典)
        - 两个都是 PyTorch 模型对象
        - 混合输入（一个模型对象，一个字典）
    """
    # 统一转换为参数字典
    if not isinstance(params_a, dict):  # 如果 params_a 是模型对象
        params_a = params_a.state_dict()
    if not isinstance(params_b, dict):  # 如果 params_b 是模型对象
        params_b = params_b.state_dict()

    # 计算和
    result = {}
    for k in params_a.keys():
        result[k] = params_a[k] + params_b[k]
    return result



def aggregate_models(model2):
    """
    聚合多个本地模型参数（加权平均）

    参数:
        model1: 主模型（将被更新）
        model2: 包含多个本地模型的列表，每个元素是 (样本数, 模型参数字典)

    返回:
        聚合后的全局参数字典
    """
    # 计算总样本数
    training_num = 0
    for idx in range(len(model2)):
        (sample_num, _) = model2[idx]
        training_num += sample_num

    # 初始化全局参数
    w_global = {}
    (_, averaged_params) = model2[0]  # 获取第一个模型的参数结构

    # 加权聚合
    for k in averaged_params.keys():
        for i in range(len(model2)):
            local_sample_number, local_model_params = model2[i]
            weight = local_sample_number / training_num  # 计算权重

            if i == 0:
                w_global[k] = local_model_params[k] * weight
            else:
                w_global[k] += local_model_params[k] * weight

    return w_global
def top_p_sparsify(tensor, p):
    """保留前 p 百分比的元素，其余置零"""
    flat = tensor.view(-1)
    k = int(p * flat.numel())
    if k == 0:
        return torch.zeros_like(tensor)
    topk, indices = torch.topk(torch.abs(flat), k)
    mask = torch.zeros_like(flat)
    mask[indices] = 1.0
    return (tensor * mask.view(tensor.shape))

def train_MoFedSAM(data_obj, act_prob,
                 learning_rate, batch_size, rho, epoch, n_minibatch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'MoFedSAM_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round



    n_client = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))
    ndvar_value_list=np.zeros((com_amount, 1))
    sam_grads_list = {}

    n_par = len(get_mdl_params([model_func()])[0])
    hist_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    ndvar_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    client_momentum = np.zeros(n_par).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_client X n_par
    clnt_models = list(range(n_client))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedSGAM_KL/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    trial=True
    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # # Get hist_params_diffs
                    # hist_params_diffs = np.load(
                    #     '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    # clnt_params_list = np.load(
                    #     '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]



        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_client)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_client))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                delta_list = torch.tensor(client_momentum).to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                clnt_models[clnt] = train_model_MoFedSAM(model, model_func, alpha_coef_adpt,
                                                    delta_list,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                clnt_params_list[clnt] = curr_model_par
                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)
                ndvar_params_diffs[clnt] = (curr_model_par - cld_mdl_param)
                # 应该是diff的值求全局平局
                # intro 三个算法的方差图

            variance_value = compute_ndvar(clnt_params_list, cld_mdl_param, selected_clnts)
            print(f"variance_value: {variance_value}")
            ndvar_value_list[i] = variance_value


            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            avg_update = avg_mdl_param_sel - cld_mdl_param
            client_momentum = avg_update / n_minibatch / learning_rate * lr_decay_per_round * -1.

            cld_mdl_param = avg_mdl_param_sel

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))


                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays

                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):

                avg_cld_mdls[i // save_period] = cur_cld_model

            df = pd.DataFrame(ndvar_value_list)
            df.to_csv("ndvar_values_FedSAM_iid.csv", index=False)

    return avg_cld_mdls, tst_cur_cld_perf


def aggregate_weights(weighted_params):
    """
    加权平均聚合客户端参数

    Args:
        weighted_params: list of (weight, params) tuples
                         params is a list of numpy arrays

    Returns:
        averaged_params: list of averaged numpy arrays
    """
    # 初始化累加器
    total_weight = 0.0
    averaged_params = [np.zeros_like(p) for p in weighted_params[0][1]]

    # 加权累加
    for weight, params in weighted_params:
        for i, p in enumerate(params):
            averaged_params[i] += weight * p
        total_weight += weight

    # 计算平均值
    for i in range(len(averaged_params)):
        averaged_params[i] /= total_weight

    return averaged_params



def train_AdDPNFL(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):

    suffix = 'AdDPNFL_' + str(act_prob) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_clnt

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))

    tst_perf_sel = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_deltas = np.zeros((n_clnt, n_par))  # 存储 delta_i
    clnt_mu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 ut (mu_t)
    clnt_nu_list = np.zeros((n_clnt, n_par))  # 存储每个客户端的 vt (nu_t)

    # 初始化全局动量/方差变量（ut/vt）
    mu_global = torch.zeros(n_par).to(device)
    nu_global = torch.ones(n_par).to(device) * 1e-10

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-NEW/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))
        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt], delta_i = train_model_AdDPNFL(clnt_models[clnt],avg_model, trn_x, trn_y,
                                                    tst_x, tst_y,
                                                    learning_rate * (lr_decay_per_round ** i), batch_size, epoch,
                                                    print_per,
                                                    weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

                clnt_deltas[clnt] = delta_i

            # # Scale with weights
            b1=0.9
            b2=0.99

            delta_t =torch.tensor(np.sum(clnt_deltas[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                    axis=0)).to(device)
            mu_global = b1*mu_global+(1-b1)*delta_t
            nu_global = b2*nu_global+(1-b2)*delta_t**2
            pi = 1e-3
            # # 确保 mu_global 和 nu_global 是 torch.Tensor
            if isinstance(mu_global, np.ndarray):
                mu_global = torch.tensor(mu_global, dtype=torch.float32).to(device)
            if isinstance(nu_global, np.ndarray):
                nu_global = torch.tensor(nu_global, dtype=torch.float32).to(device)
            #
            w_global = get_mdl_params([avg_model], n_par)[0]  # shape [n_par]
            lr_g=0.005
            w_global_updated = w_global + lr_g*(1/math.sqrt(lr_g)) * mu_global.cpu().numpy() / (np.sqrt(nu_global.cpu().numpy()) + pi)


            avg_model = set_client_from_params(model_func(), w_global_updated)


            # avg_model = set_client_from_params(model_func(), np.sum(
            #     clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
            #     axis=0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                        data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model

    return fed_mdls_sel, tst_perf_sel

def train_DPFedSAM(data_obj, act_prob,
                 learning_rate, batch_size, rho, epoch, n_minibatch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedSAM_new_' +str(epoch)+ suffix
    suffix += '_c%d_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    com_amount,save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)


    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round



    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))
    # if not os.path.exists('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8])):
    #     os.mkdir('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8]))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))
    ndvar_value_list=np.zeros((com_amount, 1))
    n_par = len(get_mdl_params([model_func()])[0])

    client_momentum = np.zeros(n_par).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sDPRuns_DPFedFER-NEW/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            round_idx=i
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                delta_list = torch.tensor(client_momentum).to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedSAM(round_idx,clnt_num,model, model_func, alpha_coef_adpt,
                                                    delta_list,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)


                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                clnt_params_list[clnt] = curr_model_par




            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            avg_update = avg_mdl_param_sel - cld_mdl_param
            client_momentum = avg_update / n_minibatch / learning_rate * lr_decay_per_round * -1.

            cld_mdl_param = avg_mdl_param_sel

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))


                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays

                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model
    # last_100_acc_tst = [tst_cur_cld_perf[i][1] for i in range(com_amount - 100, com_amount)]
    # mean_acc_tst = np.mean(last_100_acc_tst) * 100
    # var_acc_tst = np.var(last_100_acc_tst) * 100
    # with open('%sAccuracy/%s%s/results.txt' % (data_path, data_obj.name,suffix[:8]), 'w') as f:
    #     f.write(f"{mean_acc_tst} ± {var_acc_tst}\n")

    return avg_cld_mdls, tst_cur_cld_perf,tst_cur_cld_perf

def train_DPFedGAM(data_obj, act_prob,
                 learning_rate, batch_size, rho, epoch, n_minibatch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'DPFedGAM_' +str(com_amount)+ suffix
    suffix += '_c%d_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    com_amount,save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)


    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round



    n_clnt = data_obj.n_clnt
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y



    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))
    # if not os.path.exists('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8])):
    #     os.mkdir('%sAccuracy/%s%s' % (data_path,data_obj.name,suffix[:8]))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    client_momentum = np.zeros(n_par).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_DPFedFER-NEW/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            round_idx=i
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                delta_list = torch.tensor(client_momentum).to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                clnt_num=len(selected_clnts)
                clnt_models[clnt] = train_model_DPFedGAM(round_idx,clnt_num,model, model_func, alpha_coef_adpt,
                                                    delta_list,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)


                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            avg_update = avg_mdl_param_sel - cld_mdl_param
            client_momentum = avg_update / n_minibatch / learning_rate * lr_decay_per_round * -1.

            cld_mdl_param = avg_mdl_param_sel

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))


                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays

                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model
    # last_100_acc_tst = [tst_cur_cld_perf[i][1] for i in range(com_amount - 100, com_amount)]
    # mean_acc_tst = np.mean(last_100_acc_tst) * 100
    # var_acc_tst = np.var(last_100_acc_tst) * 100
    # with open('%sAccuracy/%s%s/results.txt' % (data_path, data_obj.name,suffix[:8]), 'w') as f:
    #     f.write(f"{mean_acc_tst} ± {var_acc_tst}\n")

    return avg_cld_mdls, tst_cur_cld_perf,tst_cur_cld_perf



def train_FedA1(data_obj, act_prob,
                 learning_rate, batch_size, epoch, com_amount, print_per,
                 weight_decay, model_func, init_model, alpha_coef,rho,
                 sch_step, sch_gamma, save_period,
                 suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedA1_' + str(epoch) + str(alpha_coef) + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_cld_mdls = list(range(n_save_instances))  # Cloud models

    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns_FedA1/%s/%s' % (data_path, data_obj.name, suffix[:26]))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

            del clnt_models
            clnt_models = list(range(n_clnt))

            for clnt in selected_clnts:
                # Train locally
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)

                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt]  # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_FedA1(model, cur_cld_model,model_func, alpha_coef_adpt,
                                                    cld_mdl_param_tensor, hist_params_diffs_curr,
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),rho,
                                                    batch_size, epoch, print_per, weight_decay,
                                                    data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                # print('d is {}'.format(d))
                hist_params_diffs[clnt] += (curr_model_par - cld_mdl_param)  # local drift
                clnt_params_list[clnt] = curr_model_par

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0) #1/N

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            #####


            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))


                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_cld_mdls,  tst_cur_cld_perf




