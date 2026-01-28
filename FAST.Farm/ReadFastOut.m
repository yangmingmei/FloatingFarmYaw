%% This example generates the essential files for Fast.Farm using matlab-toolbox

addpath(genpath("C:\Users\85164\Desktop\IEEE 浮式风场强化学习调频\high-fidelity Simulink model\matlab-toolbox"))
addpath(genpath("C:\Users\85164\Desktop\IEEE 浮式风场强化学习调频\high-fidelity Simulink model\Utilities"))
addpath(genpath("TurbulentWind_20WT"))

clear,clc

% power generation, sway, yaw

%% 1：model-based DRL scheme 2:Baseline  3: no yaw misalignment
dt = 0.025;
T_start = 0;
T_end = 999;
N_start = T_start/dt +1;
N_end = T_end/dt ;
length_N = N_end - N_start +1;
N = 23;

GenPwrs_1 = zeros(N, length_N);
PtfmSways_1 = zeros(N, length_N);
PtfmYaws_1 = zeros(N, length_N);
NacYaws_1 = zeros(N, length_N);

GenPwrs_2 = zeros(N, length_N);
PtfmSways_2 = zeros(N, length_N);
PtfmYaws_2 = zeros(N, length_N);
NacYaws_2 = zeros(N, length_N);

GenPwrs_3 = zeros(N, length_N);
PtfmSways_3 = zeros(N, length_N);
PtfmYaws_3 = zeros(N, length_N);
NacYaws_3 = zeros(N, length_N);



%% Read data of Simulation 1, model-based DRL scheme 2 baseline 3 convential no yaw
fast.FAST_directory = "./TurbulentWind_30WT";
fast.FAST_InputFile_1    = 'IEA_15_Semi_Farm.fstf';   % FAST input file (ext=.fst)

for Turbine_ID = 1:1:N
    disp(Turbine_ID)
    fast.Fast_out = strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.out');
    fast.Fast_out_binary =  strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.outb');

    [OutData,OutList] = ReadFASTbinary(fast.Fast_out_binary);
    if isempty(OutData)
        [OutData,OutList] = ReadFASTtext(fast.Fast_out);
    end
    
    for j = 1:length(OutList)
        simout.(OutList{j}) = OutData(:,j);
    end
    GenPwrs_1(Turbine_ID,1:length_N) = simout.GenPwr(N_start:N_end);
    PtfmSways_1(Turbine_ID,1:length_N) = simout.PtfmSway(N_start:N_end);
    PtfmYaws_1(Turbine_ID,1:length_N) = simout.PtfmYaw(N_start:N_end);
    NacYaws_1(Turbine_ID,1:length_N) = simout.NacYaw(N_start:N_end);

end
WFpower_1 = sum(GenPwrs_1, 1);

%% Read data
fast.FAST_directory = "D:\FastFarm\23FOWTs_15MW_1000s_1200m_mooring_45deg_yaw1_baseline (TASE)\TurbulentWind_23WT" ;
fast.FAST_InputFile_1    = 'IEA_15_Semi_Farm.fstf';   % FAST input file (ext=.fst)
for Turbine_ID = 1:1:N
    disp(Turbine_ID)
    fast.Fast_out = strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.out');
    fast.Fast_out_binary =  strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.outb');

    [OutData,OutList] = ReadFASTbinary(fast.Fast_out_binary);
    if isempty(OutData)
        [OutData,OutList] = ReadFASTtext(fast.Fast_out);
    end
    
    for j = 1:length(OutList)
        simout.(OutList{j}) = OutData(:,j);
    end
    GenPwrs_2(Turbine_ID,1:length_N) = simout.GenPwr(N_start:N_end);
    PtfmSways_2(Turbine_ID,1:length_N) = simout.PtfmSway(N_start:N_end);
    PtfmYaws_2(Turbine_ID,1:length_N) = simout.PtfmYaw(N_start:N_end);
    NacYaws_2(Turbine_ID,1:length_N) = simout.NacYaw(N_start:N_end);

end
WFpower_2 = sum(GenPwrs_2, 1);

%%
fast.FAST_directory = "D:\FastFarm\23FOWTs_15MW_1000s_1200m_mooring_45deg_no_yaw (TASE)\TurbulentWind_23WT" ;
fast.FAST_InputFile_1    = 'IEA_15_Semi_Farm.fstf';   % FAST input file (ext=.fst)
for Turbine_ID = 1:1:N
    disp(Turbine_ID)
    fast.Fast_out = strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.out');
    fast.Fast_out_binary =  strcat(fast.FAST_directory,filesep,fast.FAST_InputFile_1(1:end-5),'.T',num2str(Turbine_ID),'.outb');

    [OutData,OutList] = ReadFASTbinary(fast.Fast_out_binary);
    if isempty(OutData)
        [OutData,OutList] = ReadFASTtext(fast.Fast_out);
    end
    
    for j = 1:length(OutList)
        simout.(OutList{j}) = OutData(:,j);
    end
    GenPwrs_3(Turbine_ID,1:length_N) = simout.GenPwr(N_start:N_end);
    PtfmSways_3(Turbine_ID,1:length_N) = simout.PtfmSway(N_start:N_end);
    PtfmYaws_3(Turbine_ID,1:length_N) = simout.PtfmYaw(N_start:N_end);
    NacYaws_3(Turbine_ID,1:length_N) = simout.NacYaw(N_start:N_end);

end
WFpower_3 = sum(GenPwrs_3, 1);
Time = simout.Time(N_start:N_end);

%%

save('Results.mat')
%%
figure(1)
width=1440;%宽度，像素数
height=640;%高度
left=200;%距屏幕左下角水平距离
bottom=100;%距屏幕左下角垂直距离
set(gcf,'position',[left,bottom,width,height])%设置图窗大小和位置

plot(Time'-T_start,WFpower_1/1000, 'LineStyle','-','LineWidth',3,'Color',	"#0072BD")
hold on 
plot(Time'-T_start,WFpower_2/1000, 'LineStyle','-.','LineWidth',3,'Color', "#D95319")
hold on 
plot(Time'-T_start,WFpower_3/1000, 'LineStyle','--','LineWidth',3,'Color', 		"#EDB120")
hold on
area(Time'-T_start,WFpower_1/1000, 'FaceColor',"#0072BD",'FaceAlpha',.2,'EdgeAlpha',.1)
hold on; 
area(Time'-T_start,WFpower_2/1000, 'FaceColor',"#D95319",'FaceAlpha',.2,'EdgeAlpha',.1)
hold on 
area(Time'-T_start,WFpower_3/1000, 'FaceColor',	"#EDB120",'FaceAlpha',.2,'EdgeAlpha',.1)
hold on 

legend('Model-based DRL','Baseline','No yaw control')
xlabel('\fontname{Times new roman}Time(s)','FontSize',28);
ylabel('\fontname{Times new roman}Wind farm power\rm\fontname{Times new roman}(MW)','FontSize',28);  %x,y 轴

ax=gca;ax.GridLineStyle = ':';ax.GridColor = 'black';ax.GridAlpha = 0.4;  %网格属性
set(gca,'FontName','Times New Roman','FontSize',28,'LineWid',2);  %图中的字
grid on;

xlim([0 T_end-T_start])
set(gca,'XTick',[0:100:T_end-T_start])
ylim([120 200])
set(gca,'YTick',[120:20:200]);

% figure(2)
% x = linspace(0,10);
% y1 = 4 + sin(x).*exp(0.1*x);
% area(x,y1,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3)
% 
% y2 = 4 + cos(x).*exp(0.1*x);
% hold on
% area(x,y2,'FaceColor','r','FaceAlpha',.3,'EdgeAlpha',.3)
% hold off
% area(Time'-T_start,WFpower_1/1000,'FaceColor','b','FaceAlpha',.3,'EdgeAlpha',.3,'EdgeColor',	"#FF0000","LineWidth",4)

% figure(2)
% plot(PtfmSways_1(1,:))
% hold on
% plot(PtfmSways_1(2,:))
% plot(PtfmSways_1(3,:))
% plot(PtfmSways_1(4,:))
% plot(PtfmSways_1(5,:))
% % plot(PtfmSways(6,:))
% % plot(PtfmSways(7,:))
% legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')
% 
% figure(3)
% plot(PtfmYaws_1(1,:))
% hold on
% plot(PtfmYaws_1(2,:))
% plot(PtfmYaws_1(3,:))
% plot(PtfmYaws_1(4,:))
% plot(PtfmYaws_1(5,:))
% % plot(PtfmYaws(6,:))
% % plot(PtfmYaws(7,:))
% legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')
% 
% figure(4)
% plot(NacYaws_1(1,:))
% hold on
% plot(NacYaws_1(2,:))
% plot(NacYaws_1(3,:))
% plot(NacYaws_1(4,:))
% plot(NacYaws_1(5,:))
% % plot(NacYaws(6,:))
% % plot(NacYaws(7,:))
% legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')
% 
% figure(3)
% plot(PtfmYaws_1(1,:))
% hold on
% plot(PtfmYaws_1(2,:))
% plot(PtfmYaws_1(3,:))
% plot(PtfmYaws_1(4,:))
% plot(PtfmYaws_1(5,:))
% % plot(PtfmYaws(6,:))
% % plot(PtfmYaws(7,:))
% legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')
% 
% figure(5)
% plot(NacYaws_1(1,:)+PtfmYaws_1(1,:))
% hold on
% plot(NacYaws_1(2,:)+PtfmYaws_1(2,:))
% plot(NacYaws_1(3,:)+PtfmYaws_1(3,:))
% plot(NacYaws_1(4,:)+PtfmYaws_1(4,:))
% plot(NacYaws_1(5,:)+PtfmYaws_1(5,:))
% % plot(NacYaws(6,:))
% % plot(NacYaws(7,:))
% legend('FOWT 1','FOWT 2','FOWT 3','FOWT 4','FOWT 5','FOWT 6','FOWT 7')























