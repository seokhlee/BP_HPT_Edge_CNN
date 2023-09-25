%%
X =importdata('D:\SeokhyeongLee\BP_project\BP_CTT_Paper\CNN\Basement\cameraman.csv');
imag1=double(X)./255;
%% cov layer
%Y_high=importdata('test_huskyreduced_high.dat');
Y_high=importdata('test1_1590nm_low_cameraman_2.dat');
%Y_high=importdata('test1_1590nm_cameraman_2.dat');
%Yr_h=transpose(reshape(Y_high(:,2),[222,316]));
Yr_h=transpose(reshape(Y_high(:,2),[256,256]));
%Yr_h=[];
%for i=1:316
 %   for j=1:222
        %if (Yr_high(1+(i-1)*2,1+(j-1)*2)) > 0.0009
      %      Yr_h(i,j)=0.0009;
       % else
     %       Yr_h(i,j)=Yr_high(1+(i-1)*2,1+(j-1)*2);
      %  end
    %end
%end
Y_out=Y_high(77:end,:);
%Y=importdata('test_huskyreduced_low.dat');
%Y=importdata('test_camera1_low.dat');
Y=importdata('test1_1590nm_cameraman_2.dat');
%Y=importdata('test1_1590nm_cameraman_2.dat');
Yy=Y(77:end,:);
%Yr=transpose(reshape(Y(:,2),[222,316]));
Yr=transpose(reshape(Yy(:,2),[256,256]));

figure('Name','Yr_h');
imagesc(Yr_h);
axis equal;
colormap gray
axis off;

figure('Name','Yr');
imagesc(Yr);
axis equal;
colormap gray


F11=[-1 1;-1 1];
F12=[1 1;-1 -1];
%F12=[3.5 3.5;3.5 3.5];
b1=[0 0];
F21=[1 -1;1 -1];
F22=[-1 -1;1 1];
%%
nf=size(F11);
n1=size(Yr);
%fac=7.5;
%fac=6.3;
%fac=0.65;
fac=0.59;

test_difference=Yr_h-Yr*fac;
figure('Name','test');
imagesc(test_difference);
axis equal;
colormap gray
axis off;



for i=1:n1(1)-nf(1)+1
    for j=1:n1(2)-nf(1)+1
        Patch11=[Yr(i,j),Yr(i,j+1);Yr(i+1,j),Yr(i,j+1)];
        S11_id(i,j)=sum(sum(Patch11.*F11));
        S12_id(i,j)=sum(sum(Patch11.*F12));
        S21_id(i,j)=sum(sum(Patch11.*F21));
        S22_id(i,j)=sum(sum(Patch11.*F22));
    end
end


for i=1:n1(1)-nf(1)+1
    for j=1:n1(2)-nf(1)+1
        S11(i,j)=Yr(i,j)*(-fac)+Yr_h(i,j+1)+Yr(i+1,j)*(-fac)+Yr_h(i+1,j+1);
        S12(i,j)=Yr_h(i,j)+Yr_h(i,j+1)+Yr(i+1,j)*(-fac)+Yr(i+1,j+1)*(-fac);
        S21(i,j)=Yr_h(i,j)+Yr(i,j+1)*(-fac)+Yr_h(i+1,j)+Yr(i+1,j+1)*(-fac);
        S22(i,j)=Yr(i,j)*(-fac)+Yr(i,j+1)*(-fac)+Yr_h(i+1,j)+Yr_h(i+1,j+1);
    end
end
outputsave=S11;
%%

%%
figure('Name','imag');
imagesc(imag1);
axis equal;
colormap gray
axis off;

figure('Name','S11');
imagesc(S11);
axis equal;
colormap gray
axis off;


figure('Name','S12');
imagesc(S12);
axis equal;
colormap gray
axis off;

figure('Name','S21');
imagesc(S21);
axis equal;
colormap gray
axis off;

figure('Name','S22');
imagesc(S22);
axis equal;
colormap gray
axis off;

flat_S11=transpose(reshape(S11,[1,255*255]));

S11=max(0,S11+b1(1));
S12=max(0,S12+b1(2));
S21=max(0,S21);
S22=max(0,S22);
S2=S11+S12+S21+S22;

figure('Name','S2');
imagesc(S2);
axis equal;
colormap gray
axis off;
%% output rawdata

rawout=transpose(([outputsave(40,:),outputsave(41,:),outputsave(42,:),outputsave(43,:)]));
rawhigh=transpose(([Yr_h(40,:),Yr_h(41,:),Yr_h(42,:),Yr_h(43,:)]));
%%  average Pooling


flat_S11_id=transpose(reshape(S11_id,[1,255*255]));
%S21=mean(mean(S11));
%S22=mean(mean(S12));
%S2=[S21;S22];
%%  FC layer

%%F3=[-0.1415,-4.8810;0.1620,4.8829];
%b3=[7.923;-7.926];

%S3=F3*S2+b3;
%Output=exp(S3)./(1+exp(S3))