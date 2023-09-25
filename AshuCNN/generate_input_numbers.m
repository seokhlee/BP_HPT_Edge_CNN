clear all;close all;
display 'start....'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Download MNIST dataset from http://yann.lecun.com/exdb/mnist/
%% Store in a folder /directory named MNIST
%% correct the following path
Datapath = 'G:\My Drive\SWG FDTD\GST_SiN_1.55um_test\New folder\wg_2um_t_30nm_w_200nm\with ALD encapsulate\t_30_real_device\correct nk GST\150 nm ALD\patt recog\AshuCNN\MNIST\';

display 'start....'
display 'reading MNIST dataset...'

f=fopen(fullfile(Datapath, 't10k-images.idx3-ubyte'),'r', 'b') ;
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
h=fread(f,1,'int32');
w=fread(f,1,'int32');
test_x1 = uint8(fread(f,h*w*num,'uchar')); %load train images
test_x1 = permute(reshape(test_x1, h, w,num), [2 1 3]);
test_x1 = double(test_x1)./255;
fclose(f) ;

clear y;

f=fopen(fullfile(Datapath, 't10k-labels.idx1-ubyte'),'r', 'b') ;
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
y2 = double(fread(f,num,'uint8')); %load test labels
y2 = (y2)' ;

test_y = zeros([2 length(find(y2==1))+length(find(y2==2))]); % there are 10 labels in MNIST lables
count=0;
for i=1:num
    if y2(i)==1
        count=count+1;
        test_x(:,:,count)=test_x1(:,:,i);
        y(count)=y2(i);
        test_y(1,count)=1;
    end
    if y2(i)==2
        count=count+1;
        test_x(:,:,count)=test_x1(:,:,i);
        y(count)=y2(i);
        test_y(2,count)=1;
    end
end
fclose(f) ;

%%
for i=1:5
    figure('Name',num2str(i));
    imagesc(test_x(:,:,i));
    axis equal;
    colormap gray
    axis off;
end

%% cov layer

F11=[-0.2192 0.1369;-0.1485 -0.1272];
%F12=[3.5320 3.5640;3.3379 3.5785];
F12=[3.5 3.5;3.5 3.5];
b1=[0 -0.0326];

F3=[-0.1415,-4.8810;0.1620,4.8829];
b3=[7.923;-7.926];
nf=size(F11);

num=1000;
start=randi([1 count-num])

for k=1:num 
    imag1=test_x(:,:,start+k);
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

    
    S21=mean(mean(S11));
    S22=mean(mean(S12));
    S2=[S21;S22];
    
    S3=F3*S2+b3;
    Output(:,k)=exp(S3)./(1+exp(S3));
end

for i=1:num
    if Output(1,i)>=Output(2,i)
        label(i)=1;
    else 
        label(i)=2;
    end
end
    
check=label-y(start+1:start+num);
error=sum(abs(check)/num)

count=0;
for k=1:num
    imag=test_x(:,:,start+k); 
    for i=1:27
        for j=1:27
            count=count+1;
            Pix_1(count)=imag(i,j);
            Pix_2(count)=imag(i,j+1);
            Pix_3(count)=imag(i+1,j);
            Pix_4(count)=imag(i+1,j+1);
        end
    end
end
          
%% read the calibration file for each EVOA  (func EVOA_calib)
Ampq=(0:0.0001:1)';

direct="G:\My Drive\Lab_Changming\VIs\Thorlabs EVOA 1550A\EVOA1 I_V Calibration.dat";
Vq1=EVOA_calib(direct,Ampq);
direct="G:\My Drive\Lab_Changming\VIs\Thorlabs EVOA 1550A\EVOA2 I_V Calibration.dat";
Vq2=EVOA_calib(direct,Ampq);
direct="G:\My Drive\Lab_Changming\VIs\Thorlabs EVOA 1550A\EVOA3 I_V Calibration.dat";
Vq3=EVOA_calib(direct,Ampq);
direct="G:\My Drive\Lab_Changming\VIs\Thorlabs EVOA 1550A\EVOA4 I_V Calibration.dat";
Vq4=EVOA_calib(direct,Ampq);
   
    
%% convert pixel grayscale to voltage output

OutV_1=convert2V(Pix_1,Ampq,Vq1);
OutV_2=convert2V(Pix_2,Ampq,Vq2);
OutV_3=convert2V(Pix_3,Ampq,Vq3);
OutV_4=convert2V(Pix_4,Ampq,Vq4);

%% generate output voltage file


dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\OutV_1.txt',OutV_1,'delimiter',' ');
dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\OutV_2.txt',OutV_2,'delimiter',' ');
dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\OutV_3.txt',OutV_3,'delimiter',' ');
dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\OutV_4.txt',OutV_4,'delimiter',' ');

%% generate output real level file
dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\Pix_1.txt',Pix_1,'delimiter',' ');
dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\Pix_2.txt',Pix_2,'delimiter',' ');
dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\Pix_3.txt',Pix_3,'delimiter',' ');
dlmwrite('G:\My Drive\Lab_Changming\data\02_17_20_Grt_Y\0C\Edge Detection\ImputGeneration_Test\Pix_4.txt',Pix_4,'delimiter',' ');

