
%%
X =imread('D:\SeokhyeongLee\BP_project\BP_CTT_Paper\CNN\MNIST\mnistasjpg\testSet\testSet\img_22.jpg');
imag1=double(X)./255;
%% cov layer

F11=[-0.2192 0.1369;-0.1485 -0.1272];
%F12=[3.5320 3.5640;3.3379 3.5785];
F12=[3.5 3.5;3.5 3.5];
b1=[0 -0.0326];

nf=size(F11);
n1=size(imag1);

for i=1:n1(1)-nf(1)+1
    for j=1:n1(1)-nf(1)+1
        Patch11=[imag1(i,j),imag1(i,j+1);imag1(i+1,j),imag1(i,j+1)];
        S11(i,j)=sum(sum(Patch11.*F11));
        S12(i,j)=sum(sum(Patch11.*F12));
    end
end

S11=max(0,S11+b1(1));
S12=max(0,S12+b1(2));
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

%%  average Pooling

S21=mean(mean(S11));
S22=mean(mean(S12));
S2=[S21;S22];
%%  FC layer

F3=[-0.1415,-4.8810;0.1620,4.8829];
b3=[7.923;-7.926];

S3=F3*S2+b3;
Output=exp(S3)./(1+exp(S3))
