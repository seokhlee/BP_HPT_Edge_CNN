%clear;clc;close all; % edge detect__Input_generation


%RandNum_p=importdata('C:\Users\admin\Desktop\Seokhyeong\bP_CTT\ctt_1550_EVOA\test.dat');
%RandNum_n=importdata('G:\My Drive\Lab_Changming\data\09_15_20_GAN_network_test\Generate7_v2\Before Training\Step0\Step0_Pix1_n.txt');
pix1=importdata('D:\SeokhyeongLee\BP_project\BP_CTT_Paper\CNN\Basement\huskyreduced.csv');

Ampq=(0:0.001:1)';
direct="D:\SeokhyeongLee\BP_project\BP_CTT_Paper\CNN\Basement\test3.dat";
Vq1=EVOA_calib(direct,Ampq);

%OutV_1_negative=convert2V(RandNum_n,Ampq,Vq1);
OutV_1_positive=convert2V(pix1./255,Ampq,Vq1);
%Vreset(:,1)=[convert2V(1,Ampq,Vq1)];

%{
dlmwrite('G:\My Drive\Lab_Changming\data\09_15_20_GAN_network_test\Generate7_v2\Before Training\Step0\Step0_V1_p.txt',OutV_1_positive,'delimiter',' ');
dlmwrite('G:\My Drive\Lab_Changming\data\09_15_20_GAN_network_test\Generate7_v2\Before Training\Step0\Step0_V1_n.txt',OutV_1_negative,'delimiter',' ');
%}
savefile=strcat('D:\SeokhyeongLee\BP_project\BP_CTT_Paper\CNN\Basement\huskyreduced_to_V1_p.csv');
csvwrite(savefile,OutV_1_positive);

