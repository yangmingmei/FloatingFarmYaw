%% This example generates the essential files for Fast.Farm using matlab-toolbox

addpath(genpath("C:\Users\85164\Desktop\IEEE 浮式风场强化学习调频\high-fidelity Simulink model\matlab-toolbox"))
addpath(genpath("C:\Users\85164\Desktop\IEEE 浮式风场强化学习调频\high-fidelity Simulink model\Utilities"))
addpath(genpath("TurbulentWind_20WT"))

clear,clc

%% Run OpenFast
fast.FAST_directory = "./TurbulentWind_20WT";
fast.FAST_InputFile_1    = 'IEA_15_Semi_Farm.fstf';   % FAST input file (ext=.fst)


%% Read data
% GenPwrs = cell(20,1);
% PtfmSurges = cell(20,1);
% PtfmSways = cell(20,1);
% Yaws = cell(20,1);
% WindSpeeds = cell(20,1);
N_start = 1;
N_dend = 30000;
for Turbine_ID = 1:1:20
    disp(Turbine_ID)
    fast.Fast_out = strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.out');
    fast.Fast_out_binary =  strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.outb');

%     [OutData,OutList] = ReadFASTtext(fast.Fast_out);
%     if isempty(OutData)
%         [OutData,OutList] = ReadFASTbinary(fast.Fast_out_binary);
%     end

    [OutData,OutList] = ReadFASTbinary(fast.Fast_out_binary);
    if isempty(OutData)
        [OutData,OutList] = ReadFASTtext(fast.Fast_out);
    end
    
    for j = 1:length(OutList)
        simout.(OutList{j}) = OutData(:,j);
    end
    GenPwrs(Turbine_ID,N_start:N_dend) = simout.GenPwr(N_start:N_dend);
    PtfmSurges(Turbine_ID,N_start:N_dend) = simout.PtfmSurge(N_start:N_dend);
    PtfmSways(Turbine_ID,N_start:N_dend) = simout.PtfmSway(N_start:N_dend);
    Yaws(Turbine_ID,N_start:N_dend) = simout.NacYaw(N_start:N_dend) + simout.PtfmYaw(N_start:N_dend);
    WindSpeeds(Turbine_ID,N_start:N_dend) = simout.Wind1VelX(N_start:N_dend);
    OoPDefl1s(Turbine_ID,N_start:N_dend)  = simout.OoPDefl1(N_start:N_dend);

end

WFpower = sum(GenPwrs, 1);

%% Visualize
Pl_FastPlots(simout)

figure(1)
plot(WFpower)

figure(2)
plot(PtfmSways(1,:))
hold on
plot(PtfmSways(2,:))
plot(PtfmSways(3,:))
plot(PtfmSways(4,:))
plot(PtfmSways(5,:))
plot(PtfmSways(6,:))
plot(PtfmSways(7,:))
plot(PtfmSways(12,:),'LineWidth',4)
legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7','FOWT12')

figure(4)
plot(PtfmSurges(1,:))
hold on
plot(PtfmSurges(2,:))
plot(PtfmSurges(3,:))
plot(PtfmSurges(4,:))
plot(PtfmSurges(5,:))
plot(PtfmSurges(6,:))
plot(PtfmSurges(7,:),'LineWidth',4)
plot(PtfmSways(12,:),'LineWidth',4)
legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7','FOWT12')

figure(5)
plot(WindSpeeds(1,:))
hold on
plot(WindSpeeds(2,:))
plot(WindSpeeds(3,:))
plot(WindSpeeds(4,:))
plot(WindSpeeds(5,:))
plot(WindSpeeds(6,:))
plot(WindSpeeds(7,:),'LineWidth',4)

legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')

figure(6)
plot(OoPDefl1s(1,:))
hold on
plot(OoPDefl1s(2,:))
plot(OoPDefl1s(3,:))
plot(OoPDefl1s(4,:))
plot(OoPDefl1s(5,:))
plot(OoPDefl1s(6,:))
plot(OoPDefl1s(7,:),'LineWidth',4)

legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')



figure(6)
plot(GenPwrs(1,:))
hold on
plot(GenPwrs(2,:))
plot(GenPwrs(3,:))
plot(GenPwrs(4,:))
plot(GenPwrs(5,:))
plot(GenPwrs(6,:))
plot(GenPwrs(7,:),'LineWidth',4)

legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')






















