%% This example generates the essential files for Fast.Farm using matlab-toolbox

addpath(genpath("C:\Users\85164\Desktop\IEEE 浮式风场强化学习调频\high-fidelity Simulink model\matlab-toolbox"))
addpath(genpath("TurbulentWind_20WT"))

%% read FST, ServoDyn, InflowWind Files
for Turbine_ID = 1:1:20
    WindType = '3';
    HWindSpeed = '10';
    FileName_BTS = strcat('./Wind/InflowWind_High_WT',num2str(Turbine_ID),'.bts');
    FileName_DLL = strcat('rosco_WT',num2str(Turbine_ID),'.dll');
    Yaw = [-8.06248     7.2512822  -3.142085    3.4640763  0  -8.06248     7.2512822  -3.142085    3.4640763  0 -8.06248     7.2512822  -3.142085    3.4640763  0 -8.06248     7.2512822  -3.142085    3.4640763  0] *2;
    
    oldFSTName = "./TurbulentWind_20WT/FOWTs/IEA-15-240-RWT-UMaineSemi_WT.fst";
    
    newFSTName = strcat("./TurbulentWind_20WT\FOWTs\IEA-15-240-RWT-UMaineSemi_WT",num2str(Turbine_ID),".fst");
    newServoName = strcat("./TurbulentWind_20WT\FOWTs\IEA-15-240-RWT-UMaineSemi_ServoDyn_WT",num2str(Turbine_ID),".dat");
    newInflowName = strcat("./TurbulentWind_20WT\FOWTs\InflowWind_WT",num2str(Turbine_ID),".dat");
    newElastroName = strcat("./TurbulentWind_20WT\FOWTs\IEA-15-240-RWT-UMaineSemi_ElastoDyn",num2str(Turbine_ID),".dat");
    
    [~,filenameIW] = fileparts(newInflowName);
    filenameIW = strcat(filenameIW,'.dat');
    
    [~,filenameServo] = fileparts(newServoName);
    filenameServo = strcat(filenameServo,'.dat');

    [~,filenameElastro] = fileparts(newElastroName);
    filenameElastro = strcat(filenameElastro,'.dat');
    
    % Derived Parameters
    [templateDir, baseName, ext ] = fileparts(oldFSTName);
    if strcmp(templateDir, filesep)
        templateDir = ['.' filesep];
    end
    
    %-----------------------------------------------------------------------------------------------------------
    % Read and setup path of new FST files
    FP = FAST2Matlab(oldFSTName, 2); %FP are the FST parameters, specify 2 lines of header
    
    FP_mod = SetFASTPar(FP,'InflowFile',strcat('"', filenameIW ,'"'));
    FP_mod = SetFASTPar(FP_mod,'ServoFile',strcat('"', filenameServo ,'"'));
    FP_mod = SetFASTPar(FP_mod,'EDFile',strcat('"', filenameElastro ,'"'));
    FP_mod = SetFASTPar(FP_mod,'WtrDpth',"400");
    
    Matlab2FAST(FP_mod, oldFSTName, newFSTName, 2); %contains 2 header lines

    %-----------------------------------------------------------------------------------------------------------
    % Read and setup new InflowWind files
    [paramIW, templateFilenameIW] = GetFASTPar_Subfile(FP, 'InflowFile', templateDir, templateDir);
    
    % Modify some parameters in this file 
    paramIW_mod = SetFASTPar(paramIW    ,'WindType'  , WindType);
    paramIW_mod = SetFASTPar(paramIW_mod,'HWindSpeed', HWindSpeed);
    
    % Write the new inflow wind file
    Matlab2FAST(paramIW_mod, templateFilenameIW, newInflowName, 2); %contains 2 header lines
    
    %-----------------------------------------------------------------------------------------------------------
    % Read the Servo file
    [paramServo, templateFilenameServo] = GetFASTPar_Subfile(FP, 'ServoFile', templateDir, templateDir);
    
    % Modify some parameters in this file (setting the wind the a steady wind at 12m/s
    paramServo_mod = SetFASTPar(paramServo    ,'DLL_FileName'  , FileName_DLL);
    paramServo_mod = SetFASTPar(paramServo_mod,'YawNeut', num2str(Yaw(Turbine_ID)));
    paramServo_mod = SetFASTPar(paramServo_mod,'NacYawF', num2str(Yaw(Turbine_ID)));
    
    % Write the new ServoDyn file
    Matlab2FAST(paramServo_mod, templateFilenameServo, newServoName, 2); %contains 2 header lines

    %-----------------------------------------------------------------------------------------------------------
    % Read and setup new Elastrodyn files
    [paramElastro, templateFilenameElastro] = GetFASTPar_Subfile(FP, 'EDFile', templateDir, templateDir);
    
    % Modify some parameters in this file 
    paramElastro_mod = SetFASTPar(paramElastro   ,'NacYaw' , num2str(Yaw(Turbine_ID)));
    
    % Write the new ServoDyn file
    Matlab2FAST(paramElastro_mod, templateFilenameElastro, newElastroName, 2); %contains 2 header lines


end



%% run Turbwind 
clear,clc
get_new_INP_file = false;
get_new_turb_wind = false; % 


if get_new_INP_file == true
    
    INP.Uref = 10;
    INP.IECturbc = 10;
    INP.IECstandard = "1-Ed3";
    INP.IEC_WindType = "NTM" ;
    INP.AnalysisTime = 200;
    INP.UsableTime = 200;

    Inp_filenames = cell(1, 20);
    
    for Turbine_ID = 1:1:20
        INP.RandSeed1 = 2024 + Turbine_ID;
        oldINPName = "./TurbulentWind_20WT/FOWTs/Wind/InflowWind_High_WT.inp";

        newINPName = strcat("./TurbulentWind_20WT/FOWTs/Wind\InflowWind_High_WT",num2str(Turbine_ID),".inp");
        Inp_filenames{1,Turbine_ID} = char(newINPName);
        
        paramINP = FAST2Matlab(oldINPName, 2);
        paramINP_mod = SetFASTPar(paramINP ,'URef',INP.Uref);
        paramINP_mod = SetFASTPar(paramINP_mod ,'TurbModel',['"' "IECVKM" '"']);
        paramINP_mod = SetFASTPar(paramINP_mod ,'UserFile', ['"' "unused" '"']);
        paramINP_mod = SetFASTPar(paramINP_mod ,'ProfileFile', "unused");
        paramINP_mod = SetFASTPar(paramINP_mod ,'RandSeed1',INP.RandSeed1);
        paramINP_mod = SetFASTPar(paramINP_mod ,'IECturbc',INP.IECturbc);
%         paramINP_mod = SetFASTPar(paramINP_mod ,'IECstandard',['"' INP.IECstandard '"']);
%         paramINP_mod = SetFASTPar(paramINP_mod ,'IEC_WindType',['"' INP.IEC_WindType '"']);
        paramINP_mod = SetFASTPar(paramINP_mod ,'AnalysisTime',INP.AnalysisTime);
        paramINP_mod = SetFASTPar(paramINP_mod ,'UsableTime',INP.UsableTime);
        
        Matlab2FAST(paramINP_mod, oldINPName, newINPName, 2); %contains 2 header lines
    end

end

TubSim_exe = '.\TurbSim_x64.exe';
% 14开始
opts.flag = '';
if get_new_turb_wind == true
    commands=cell(1,20);
    for Turbine_ID = 14:1:20
        Inp_file = Inp_filenames{Turbine_ID};
        commands{Turbine_ID} = [TubSim_exe ' ' opts.flag ' ' Inp_file];
        runCommands(commands(Turbine_ID))
    end
    
end






















