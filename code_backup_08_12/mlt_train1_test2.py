import sys
import time
import numpy as np
import pandas as pd
import torch.optim as optim
import os
import torch
from pathlib import Path

import streamlit as st


def train(model, args, train_loader, test_loader):
    # define
    from code_backup_08_12 import mlt_dataprocess
    from code_backup_08_12 import mlt_loss

    latest_iteration_first = st.empty()
    latest_iteration_first.text(f'Iteration of stage Ⅰ : 0')
    bar_first = st.progress(0)
    latest_iteration_second = st.empty()
    latest_iteration_second.text(f'Iteration of stage Ⅱ : 0')
    bar_second = st.progress(0)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    Result = os.path.join(work_dir, 'result')
    train_loss = np.repeat(np.nan, 6, axis=0).tolist()
    test_loss = np.repeat(np.nan, 7, axis=0).tolist()
    test_std_error = np.repeat(np.nan, 5, axis=0).tolist()
    train_std_error = np.repeat(np.nan, 5, axis=0).tolist()
    train_lossfunction1 = mlt_loss.traloss0()
    train_lossfunction2 = mlt_loss.traloss2()
    test_lossfunction1 = mlt_loss.tesloss0()
    test_lossfunction2 = mlt_loss.tesloss2()
    stdfunction = mlt_loss.Std()
    Uncertainty = mlt_loss.bploss(len(args.target_index), args).to(args.device)

    # scheduler
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # training
    since = time.time()
    sys.stdout = mlt_dataprocess.Record(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.logname))
    csvfile = Path(os.path.join('result', args.csvname))
    if csvfile.is_file():
        os.remove(csvfile)

    """ 顺序是 K phi theta p t los """

    list = ['epoch', 'K_mae', 'Phi_mae', 'Theta_mae', 'P_mae', 'T_mae', 'TPR_mae', 'FPR_mae', 'K_std', 'Phi_std',
            'Theta_std', 'P_std', 'T_std']
    data = pd.DataFrame([list])
    data.to_csv(os.path.join(Result, args.csvname), mode='w', header=None, index=False)

    print("*************************第一阶段:********************************* ")
    ################################################
    col1, col2, col3 = st.columns(3)
    info = st.empty()
    a = []
    with col1:
        newline1 = st.line_chart(a)
        newline2 = st.line_chart(a)
    with col2:

        newline3 = st.line_chart(a)
        newline4 = st.line_chart(a)
    with col3:
        newline5 = st.line_chart(a)
        newline6 = st.line_chart(a)
    #########################################################
    for epoch in range(args.epochs_P1):
        P1_epoch_trainloss = torch.Tensor([0]).to(args.device)
        P1_epoch_teststderror = torch.Tensor([0]).to(args.device)
        P1_epoch_testloss = torch.Tensor([0]).to(args.device)
        # optimizer
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': Uncertainty.parameters(), 'lr': args.lr}])

        for Data, Label, mask in (train_loader):
            data = Data.to(args.device)
            label = Label.to(args.device)
            mask = mask.to(args.device)
            """注意、接下来遇到的所有的thephi都是K&Phi。所有的powerration都是theta"""
            phithe, poweratio, power, delay, los = model(data, args)
            # loss1thephi = train_lossfunction1(thephi, label[:, 0:2, :, :], mask)  # c
            loss1poweratio = train_lossfunction1(poweratio, label[:, 0:1, :, :], mask)
            # loss2poweratio = train_lossfunction1(poweratio, label[:, 2:3, :, :], mask)
            loss2phithe = train_lossfunction1(phithe, label[:, 1:3, :, :], mask)
            loss3power = train_lossfunction1(power, label[:, 3:4, :, :], mask)
            loss4delay = train_lossfunction1(delay, label[:, 4:5, :, :], mask)
            lossLOS = train_lossfunction2(los, label[:, [-1], :, :], mask).unsqueeze(0)  # 1
            # loss = torch.cat((loss1thephi, loss2poweratio, loss3power, loss4delay, lossLOS), dim=0)

            """std error plus"""
            # std1thephi = stdfunction(thephi, label[:, 0:2, :, :], mask)
            std1poweratio = stdfunction(poweratio, label[:, 0:1, :, :], mask)
            # std2poweratio = stdfunction(poweratio, label[:, 2:3, :, :], mask)
            std2phithe = stdfunction(phithe, label[:, 1:3, :, :], mask)
            std3power = stdfunction(power, label[:, 3:4, :, :], mask)
            std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)

            """
            loss1thephi = loss1thephi + std1thephi
            loss2poweratio = loss2poweratio + std2poweratio
            loss3power = loss3power + std3power
            loss4delay = loss4delay + std4delay
            """

            loss = torch.cat((loss1poweratio, loss2phithe, loss3power, loss4delay, lossLOS), dim=0)
            Std = torch.cat((std1poweratio, std2phithe, std3power, std4delay), dim=0)

            bploss = Uncertainty(loss, Std)
            optimizer.zero_grad()
            bploss.backward()
            optimizer.step()
            P1_epoch_trainloss = P1_epoch_trainloss + loss / len(train_loader)

        # print(model.state_dict()['net1.Convv.0.bias'])

        with torch.no_grad():
            TPRmean = torch.Tensor([0]).to(args.device)
            FPRmean = torch.Tensor([0]).to(args.device)
            for Data, Label, mask in test_loader:
                data = Data.to(args.device)
                label = Label.to(args.device)
                mask = mask.to(args.device)
                phithe, poweratio, power, delay, los = model(data, args)
                # loss1thephi = test_lossfunction1(phithe, label[:, 0:2, :, :], mask)  # c
                loss1poweratio = test_lossfunction1(poweratio, label[:, 0:1, :, :], mask)
                # loss2poweratio = test_lossfunction1(poweratio, label[:, 2:3, :, :], mask)
                loss2phithe = test_lossfunction1(phithe, label[:, 1:3, :, :], mask)
                loss3power = test_lossfunction1(power, label[:, 3:4, :, :], mask)
                loss4delay = test_lossfunction1(delay, label[:, 4:5, :, :], mask)
                TPR, FPR = test_lossfunction2(los, label[:, [-1], :, :], mask)
                loss = torch.cat((loss1poweratio, loss2phithe, loss3power, loss4delay), dim=0)

                # std1thephi = stdfunction(thephi, label[:, 0:2, :, :], mask)
                std1poweratio = stdfunction(poweratio, label[:, 0:1, :, :], mask)
                # std2poweratio = stdfunction(poweratio, label[:, 2:3, :, :], mask)
                std2phithe = stdfunction(phithe, label[:, 1:3, :, :], mask)
                std3power = stdfunction(power, label[:, 3:4, :, :], mask)
                std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)
                std = torch.cat((std1poweratio, std2phithe, std3power, std4delay), dim=0)

                P1_epoch_testloss = P1_epoch_testloss + loss / len(test_loader)
                P1_epoch_teststderror = P1_epoch_teststderror + std / len(test_loader)
                TPRmean += TPR / len(test_loader)
                FPRmean += FPR / len(test_loader)

        # result recording
        counter = 0
        for index_I in range(5):
            if len([k for k in range(len(args.target_index)) if args.target_index[k] == index_I + 1]) > 0:
                train_loss[index_I] = P1_epoch_trainloss[counter].item()
                test_loss[index_I] = P1_epoch_testloss[counter].item()
                test_std_error[index_I] = P1_epoch_teststderror[counter].item()
                counter = counter + 1
        train_loss[5] = P1_epoch_trainloss[5].item()
        test_loss[5] = TPRmean.item()
        test_loss[6] = FPRmean.item()

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        test_loss = (np.array(test_loss) * [100, 180, 180, 100, 100, 100, 100]).tolist()
        train_loss = (np.array(train_loss) * [100, 180, 180, 100, 100, 1]).tolist()
        test_std_error = (np.array(test_std_error) * [100, 180, 180, 100, 100]).tolist()
        list = [epoch + 1] + test_loss
        list = list + test_std_error
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)

        print(Uncertainty.coefficient.detach().squeeze().cpu().numpy())
        latest_iteration_first.text(f'Iteration of stage Ⅰ : {epoch + 1}')
        bar_first.progress((epoch + 1) / args.epochs_P1)
        print(
            f"Epoch : {epoch + 1} (trainloss) K :{train_loss[0]:.4f}   phi :{train_loss[1]:.4f}   theta :{train_loss[2]:.4f}\
   P :{train_loss[3]:.4f}   T :{train_loss[4]:.4f}   L :{train_loss[5]:.4f}")
        st.write(
            f"Epoch : {epoch + 1} (trainloss) K :{train_loss[0]:.4f}   phi :{train_loss[1]:.4f}   theta :{train_loss[2]:.4f}\
   P :{train_loss[3]:.4f}   T :{train_loss[4]:.4f}   L :{train_loss[5]:.4f}")

        ##########################
        info.success("第一轮第" + str(epoch + 1) + "次已完成")
        newline1.add_rows([train_loss[0]])
        newline2.add_rows([test_loss[5]])
        newline3.add_rows([train_loss[2]])
        newline4.add_rows([train_loss[3]])
        newline5.add_rows([train_loss[4]])
        newline6.add_rows([train_loss[5]])

        ####################################

        print(
            f"Epoch : {epoch + 1} (testloss)  K :{test_loss[0]:.4f}   phi :{test_loss[1]:.4f}   theta :{test_loss[2]:.4f}\
   P :{test_loss[3]:.4f}   T :{test_loss[4]:.4f}   TPR :{test_loss[5]:.4f}   FPR :{test_loss[6]:.4f}")
        st.write(
            f"Epoch : {epoch + 1} (testloss)  K :{test_loss[0]:.4f}   phi :{test_loss[1]:.4f}   theta :{test_loss[2]:.4f}\
   P :{test_loss[3]:.4f}   T :{test_loss[4]:.4f}   TPR :{test_loss[5]:.4f}   FPR :{test_loss[6]:.4f}")

        print(
            f"Epoch : {epoch + 1} (test_std_error)  K :{test_std_error[0]:.4f}   phi :{test_std_error[1]:.4f}   theta :{test_std_error[2]:.4f}\
   P :{test_std_error[3]:.4f}   T :{test_std_error[4]:.4f}\n")
        st.write(
            f"Epoch : {epoch + 1} (test_std_error)  K :{test_std_error[0]:.4f}   phi :{test_std_error[1]:.4f}   theta :{test_std_error[2]:.4f}\
   P :{test_std_error[3]:.4f}   T :{test_std_error[4]:.4f}\n")
        st.write("\n")

    print("*************************第二阶段:********************************* ")
    args.period = 2
    for epoch in range(args.epochs_P2):
        P2_epoch_trainloss = torch.Tensor([0]).to(args.device)
        P2_epoch_teststderror = torch.Tensor([0]).to(args.device)
        P2_epoch_testloss = torch.Tensor([0]).to(args.device)
        P2_epoch_trainstderror = torch.Tensor([0]).to(args.device)
        # optimizer
        optimizer = optim.Adam([
            {'params': model.net2.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}])
        for Data, Label, mask in (train_loader):
            data = Data.to(args.device)
            label = Label.to(args.device)
            mask = mask.to(args.device)
            phithe, poweratio, power, delay, los = model(data, args)
            # loss1thephi = train_lossfunction1(thephi, label[:, 0:2, :, :], mask)  # c
            loss1poweratio = train_lossfunction1(poweratio, label[:, 0:1, :, :], mask)
            # loss2poweratio = train_lossfunction1(poweratio, label[:, 2:3, :, :], mask)
            loss2phithe = train_lossfunction1(phithe, label[:, 1:3, :, :], mask)
            loss3power = train_lossfunction1(power, label[:, 3:4, :, :], mask)
            loss4delay = train_lossfunction1(delay, label[:, 4:5, :, :], mask)
            lossLOS = train_lossfunction2(los, label[:, [-1], :, :], mask).unsqueeze(0)  # 1
            loss = torch.cat((loss1poweratio, loss2phithe, loss3power, loss4delay, lossLOS), dim=0)

            # trstd1thephi = stdfunction(thephi, label[:, 0:2, :, :], mask)
            trstd1poweratio = stdfunction(poweratio, label[:, 0:1, :, :], mask)
            # trstd2poweratio = stdfunction(poweratio, label[:, 2:3, :, :], mask)
            trstd2phithe = stdfunction(phithe, label[:, 1:3, :, :], mask)
            trstd3power = stdfunction(power, label[:, 3:4, :, :], mask)
            trstd4delay = stdfunction(delay, label[:, 4:5, :, :], mask)
            std = torch.cat((trstd1poweratio, trstd2phithe, trstd3power, trstd4delay), dim=0)

            optimizer.zero_grad()
            # torch.sum(loss).backward()
            torch.sum(loss1poweratio).backward()
            torch.sum(loss2phithe).backward()
            torch.sum(loss3power).backward()
            torch.sum(loss4delay).backward()
            torch.sum(lossLOS).backward()
            optimizer.step()

            P2_epoch_trainstderror = P2_epoch_trainstderror + std / len(train_loader)
            P2_epoch_trainloss = P2_epoch_trainloss + loss / len(train_loader)

        # print(model.state_dict()['net2.power.0.bias'])

        with torch.no_grad():
            TPRmean = torch.Tensor([0]).to(args.device)
            FPRmean = torch.Tensor([0]).to(args.device)
            for Data, Label, mask in test_loader:
                data = Data.to(args.device)
                label = Label.to(args.device)
                mask = mask.to(args.device)
                phithe, poweratio, power, delay, los = model(data, args)
                # loss1thephi = test_lossfunction1(thephi, label[:, 0:2, :, :], mask)  # c
                loss1poweratio = test_lossfunction1(poweratio, label[:, 0:1, :, :], mask)
                # loss2poweratio = test_lossfunction1(poweratio, label[:, 2:3, :, :], mask)
                loss2phithe = test_lossfunction1(phithe, label[:, 1:3, :, :], mask)
                loss3power = test_lossfunction1(power, label[:, 3:4, :, :], mask)
                loss4delay = test_lossfunction1(delay, label[:, 4:5, :, :], mask)
                TPR, FPR = test_lossfunction2(los, label[:, [-1], :, :], mask)
                loss = torch.cat((loss1poweratio, loss2phithe, loss3power, loss4delay), dim=0)

                # std1thephi = stdfunction(thephi, label[:, 0:2, :, :], mask)
                std1poweratio = stdfunction(poweratio, label[:, 0:1, :, :], mask)
                # std2poweratio = stdfunction(poweratio, label[:, 2:3, :, :], mask)
                std2phithe = stdfunction(phithe, label[:, 1:3, :, :], mask)
                std3power = stdfunction(power, label[:, 3:4, :, :], mask)
                std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)
                std = torch.cat((std1poweratio, std2phithe, std3power, std4delay), dim=0)

                P2_epoch_testloss = P2_epoch_testloss + loss / len(test_loader)
                P2_epoch_teststderror = P2_epoch_teststderror + std / len(test_loader)
                TPRmean += TPR / len(test_loader)
                FPRmean += FPR / len(test_loader)

        # result recording --
        counter = 0
        for index_I in range(5):
            if len([k for k in range(len(args.target_index)) if args.target_index[k] == index_I + 1]) > 0:
                train_loss[index_I] = P2_epoch_trainloss[counter].item()
                test_loss[index_I] = P2_epoch_testloss[counter].item()
                test_std_error[index_I] = P2_epoch_teststderror[counter].item()
                train_std_error[index_I] = P2_epoch_trainstderror[counter].item()
                counter = counter + 1
        train_loss[5] = P2_epoch_trainloss[5].item()
        test_loss[5] = TPRmean.item()
        test_loss[6] = FPRmean.item()

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        test_loss = (np.array(test_loss) * [100, 180, 180, 100, 100, 100, 100]).tolist()
        train_loss = (np.array(train_loss) * [100, 180, 180, 100, 100, 1]).tolist()
        test_std_error = (np.array(test_std_error) * [100, 180, 180, 100, 100]).tolist()
        train_std_error = (np.array(train_std_error) * [100, 180, 180, 100, 100]).tolist()
        list = [epoch + 1] + test_loss
        list = list + test_std_error
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(Result, args.csvname), mode='a', header=0, index=False)

        latest_iteration_second.text(f'Iteration of stage Ⅱ : {epoch + 1}')
        bar_second.progress((epoch + 1) / args.epochs_P2)



        ##########################
        info.success("第二轮第" + str(epoch + 1) + "次已完成")
        newline1.add_rows([train_loss[0]])
        newline2.add_rows([train_loss[1]])
        newline3.add_rows([train_loss[2]])
        newline4.add_rows([train_loss[3]])
        newline5.add_rows([train_loss[4]])
        newline6.add_rows([train_loss[5]])

        ####################################

        st.write(
            f"Epoch : {epoch + 1} (trainloss) K :{train_loss[0]:.4f}   phi :{train_loss[1]:.4f}   theta :{train_loss[2]:.4f}\
            P :{train_loss[3]:.4f}   T :{train_loss[4]:.4f}   L :{train_loss[5]:.4f}")

        print(
            f"Epoch : {epoch + 1} (trainloss) K :{train_loss[0]:.4f}   phi :{train_loss[1]:.4f}   theta :{train_loss[2]:.4f}\
    P :{train_loss[3]:.4f}   T :{train_loss[4]:.4f}   L :{train_loss[5]:.4f}")
        st.write(
            f"Epoch : {epoch + 1} (train_std_error) K :{train_std_error[0]:.4f}   phi :{train_std_error[1]:.4f}   theta :{train_std_error[2]:.4f}\
            P :{train_std_error[3]:.4f}   T :{train_std_error[4]:.4f}")

        print(
            f"Epoch : {epoch + 1} (train_std_error) K :{train_std_error[0]:.4f}   phi :{train_std_error[1]:.4f}   theta :{train_std_error[2]:.4f}\
    P :{train_std_error[3]:.4f}   T :{train_std_error[4]:.4f}")

        st.write(
            f"Epoch : {epoch + 1} (testloss)  K :{test_loss[0]:.4f}   phi :{test_loss[1]:.4f}   theta :{test_loss[2]:.4f}\
            P :{test_loss[3]:.4f}   T :{test_loss[4]:.4f}   TPR :{test_loss[5]:.4f}   FPR :{test_loss[6]:.4f}")
        print(
            f"Epoch : {epoch + 1} (testloss)  K :{test_loss[0]:.4f}   phi :{test_loss[1]:.4f}   theta :{test_loss[2]:.4f}\
    P :{test_loss[3]:.4f}   T :{test_loss[4]:.4f}   TPR :{test_loss[5]:.4f}   FPR :{test_loss[6]:.4f}")

        st.write(
            f"Epoch : {epoch + 1} (test_std_error)  K :{test_std_error[0]:.4f}   phi :{test_std_error[1]:.4f}   theta :{test_std_error[2]:.4f}\
            P :{test_std_error[3]:.4f}   T :{test_std_error[4]:.4f}\n")

        print(
            f"Epoch : {epoch + 1} (test_std_error)  K :{test_std_error[0]:.4f}   phi :{test_std_error[1]:.4f}   theta :{test_std_error[2]:.4f}\
    P :{test_std_error[3]:.4f}   T :{test_std_error[4]:.4f}\n")

        # save model parameters and weights
        if epoch == args.epochs_P2 - 5:
            last_model_wts = model.state_dict()
            save_path_last = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.wight_lastname)
            torch.save(last_model_wts, save_path_last)
