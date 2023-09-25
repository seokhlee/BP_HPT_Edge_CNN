%clc;clear;
%%
Datapath2 ='D:\SeokhyeongLee\BP_project\BP_CTT\ctt_v12_IDT_20210703\MNIST_Test_0705';
%X =imread('D:\SeokhyeongLee\BP_project\BP_CTT\ctt_v12_IDT_20210703\MNIST_Test_0705');
for p=33:80
        filename = strcat('test_32_',num2str(p),'00nA_1.csv');
%filename = strcat('alex_d1_p3p4_test_gate_3100mVb.csv');
    sig_data(:,:,p) = load(fullfile(Datapath2,filename));

end   
%% cov layer

K1=[33+8 33+25 33+47;33+0 33+25 33+36;33+14 33+38 33+31];
%F12=[3.5320 3.5640;3.3379 3.5785];
K2=[33+16 33+17 33+25;33+24 33+47 33+16;33+25 33+7 33+17];
b1=cnn.layers{1, 2}.b;
%b1=[-4.3711,-0.0456];
K11=[-4.230340426	0.45243617	6.5125; -6.434	0.45243617	3.482468085; -2.577595745	4.033382979	 2.105180851];
K12=[-0.102261702	-0.095846809	-0.04452766; -0.050942553	0.0966	-0.102261702; -0.04452766	-0.159995745	-0.095846809];
nk=size(K1);
ni=sqrt((length(sig_data(:,1,1))));

for i=1:ni-2
    for j=1:ni-2
        conv1(i,j)=sig_data(i+ni*(j-1),2,K1(1,1))./(K1(1,1)*10^-6)*10*K11(1,1)+sig_data(i+1+ni*(j-1),2,K1(1,2))./(K1(1,2)*10^-6)*10*K11(1,2)+sig_data(i+2+ni*(j-1),2,K1(1,3))./(K1(1,3)*10^-6)*10*K11(1,3)+sig_data(i+ni*(j),2,K1(2,1))./(K1(2,1)*10^-6)*10*K11(2,1)+sig_data(i+1+ni*(j),2,K1(2,2))./(K1(2,2)*10^-6)*10*K11(2,2)+sig_data(i+2+ni*(j),2,K1(2,3))./(K1(2,3)*10^-6)*10*K11(2,3)+sig_data(i+ni*(j+1),2,K1(3,1))./(K1(3,1)*10^-6)*10*K11(3,1)+sig_data(i+1+ni*(j+1),2,K1(3,2))./(K1(3,2)*10^-6)*10*K11(3,2)+sig_data(i+2+ni*(j+1),2,K1(3,3))./(K1(3,3)*10^-6)*10*K11(3,3);
        conv2(i,j)=sig_data(i+ni*(j-1),2,K2(1,1))./(K2(1,1)*10^-6)*10*K12(1,1)+sig_data(i+1+ni*(j-1),2,K2(1,2))./(K2(1,2)*10^-6)*10*K12(1,2)+sig_data(i+2+ni*(j-1),2,K2(1,3))./(K2(1,3)*10^-6)*10*K12(1,3)+sig_data(i+ni*(j),2,K2(2,1))./(K2(2,1)*10^-6)*10*K12(2,1)+sig_data(i+1+ni*(j),2,K2(2,2))./(K2(2,2)*10^-6)*10*K12(2,2)+sig_data(i+2+ni*(j),2,K2(2,3))./(K2(2,3)*10^-6)*10*K12(2,3)+sig_data(i+ni*(j+1),2,K2(3,1))./(K2(3,1)*10^-6)*10*K12(3,1)+sig_data(i+1+ni*(j+1),2,K2(3,2))./(K2(3,2)*10^-6)*10*K12(3,2)+sig_data(i+2+ni*(j+1),2,K2(3,3))./(K2(3,3)*10^-6)*10*K12(3,3);
        
    end
end
%relu
S11=conv1+b1(1);
S12=conv2+b1(2);
%%
figure('Name','imag');
imagesc(transpose(reshape(sig_data(:,2,80),[28,28])));
axis equal;
colormap gray
axis off;

figure('Name','S11');
imagesc(transpose(S11));
axis equal;
colormap gray
axis off;


figure('Name','S12');
imagesc(transpose(S12));
axis equal;
colormap gray
axis off;

%%  average Pooling
S11=max(0,conv1+b1(1));
S12=max(0,conv2+b1(2));

S211=mean(mean(S11(1:13,1:13)));
S212=mean(mean(S11(1:13,14:26)));
S213=mean(mean(S11(14:26,1:13)));
S214=mean(mean(S11(14:26,14:26)));
S221=mean(mean(S12(1:13,1:13)));
S222=mean(mean(S12(1:13,14:26)));
S223=mean(mean(S12(14:26,1:13)));
S224=mean(mean(S12(14:26,14:26)));


S2=transpose([S211 S212 S213 S214 S221 S222 S223 S224]);
%%  FC layer

F3=[0.989198529552177,-6.07181942283334,-6.46918378373735,-0.481582562138981,-0.229711378279665,-0.188951202918283,0.149228394145220,-0.178131398565880;-1.06503019711013,6.11095688752357,6.44038484414863,0.547950747630256,0.202922114696913,0.170438292710118,0.209068634515200,-0.000923457025492860];
b3=cnn.layers{1, 4}.W(:,1);
%b3=[0.9892;-1.0650];
S3=F3*S2+b3;
Output=exp(S3)./(1+exp(S3))
