clear all
display 'start....'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Download MNIST dataset from http://yann.lecun.com/exdb/mnist/
%% Store in a folder /directory named MNIST
%% correct the following path
currPath = pwd;
Datapath = strcat(currPath,'\AshuCNN\MNIST');
num1=0;num2=1;num3=2;

display 'start....'
display 'reading MNIST dataset...'
f=fopen(fullfile(Datapath, 'train-images.idx3-ubyte'),'r', 'b') ;
if f < 0
    error('please load MNIST dataset, store it in a folder and check the path and name of the file');
end
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
h=fread(f,1,'int32');
w=fread(f,1,'int32');
train_x1 = uint8(fread(f,h*w*num,'uchar')); %load train images
train_x1 = permute(reshape(train_x1, h, w,num), [2 1 3]);
train_x1 = double(train_x1)./255;
fclose(f) ;


f=fopen(fullfile(Datapath, 't10k-images.idx3-ubyte'),'r', 'b') ;
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
h=fread(f,1,'int32');
w=fread(f,1,'int32');
test_x1 = uint8(fread(f,h*w*num,'uchar')); %load train images
test_x1 = permute(reshape(test_x1, h, w,num), [2 1 3]);
test_x1 = double(test_x1)./255;
fclose(f) ;


f=fopen(fullfile(Datapath, 'train-labels.idx1-ubyte'),'r', 'b') ;
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
y1 = double(fread(f,num,'uint8'));   %load train labels
y1 = (y1)'; %.


train_y = zeros([3 length(find(y1==num1))+length(find(y1==num2))+length(find(y1==num3))]); % there are 10 labels in MNIST lables

count=0;
for i=1:num
    if y1(i)==num1
        count=count+1;
        train_x(:,:,count)=train_x1(:,:,i);
        y(count)=y1(i);
        train_y(1,count)=1;
    end
    if y1(i)==num2
        count=count+1;
        train_x(:,:,count)=train_x1(:,:,i);
        y(count)=y1(i);
        train_y(2,count)=1;
    end
    if y1(i)==num3
        count=count+1;
        train_x(:,:,count)=train_x1(:,:,i);
        y(count)=y1(i);
        train_y(3,count)=1;
    end
end
fclose(f) ;

clear y;

f=fopen(fullfile(Datapath, 't10k-labels.idx1-ubyte'),'r', 'b') ;
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
y2 = double(fread(f,num,'uint8')); %load test labels
y2 = (y2)' ;

test_y = zeros([3 length(find(y2==num1))+length(find(y2==num2))+length(find(y2==num3))]); % there are 10 labels in MNIST lables
count=0;
for i=1:num
    if y2(i)==num1
        count=count+1;
        test_x(:,:,count)=test_x1(:,:,i);
        y(count)=y2(i);
        test_y(1,count)=1;
    end
    if y2(i)==num2
        count=count+1;
        test_x(:,:,count)=test_x1(:,:,i);
        y(count)=y2(i);
        test_y(2,count)=1;
    end
    if y2(i)==num3
        count=count+1;
        test_x(:,:,count)=test_x1(:,:,i);
        y(count)=y2(i);
        test_y(3,count)=1;
    end
end
fclose(f) ;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% We create our arbitary CNN and train it with MNIST dataset
%%%%% The architecture of CNN is arbitrarily choosen for experimental purpose
%%%%% The architecture may be revised for better result.

%   cnnAddConvLayer - Add convolution layer
%   cnn, no_of_feature_maps, sizeof(kernels), activation function -'sigm' 
%   for sigmoid, 'tanh' for tanh, 'rect' for ReLu, 'soft' for softmax, 
%  'none' for none, 'plus' for softplus.

% cnnAddPoolLayer - Add Pool layer
% cnn, subsampling factor, subsampling type. Presently only 'mean'
% subsampling is implemented.

%cnnAddFCLayer - Add fully connected neural network layer
% cnn, no of NN nodes, activation function.


%% xx = xx - mean(xx(:));

% initialize cnn
cnn.namaste=3; % just intiationg cnn object

cnn=initcnn(cnn,[h w]);

% construct cnn
cnn=cnnAddConvLayer(cnn, 2, [3 3], 'rect');
cnn=cnnAddPoolLayer(cnn, 13, 'mean');
cnn=cnnAddFCLayer(cnn,3, 'sigm' ); %add fully connected layer % last layer no of nodes = no of lables

%%
%%%more parameters
%cnn.loss_func = 'cros';

%cnn.loss_func = 'quad'; 
no_of_epochs = 200;
batch_size=40;
display 'training started...Wait for ~200 seconds...'
tic
cnn=traincnn(cnn,train_x(:,:,1:11000),train_y(:,1:11000), no_of_epochs,batch_size);
toc
display '...training finished.'
display 'testing started....'
tic
err=testcnn(cnn, test_x, test_y);
toc
display '... testing finished. To get minimum error, increase no of epochs while training.'

%%
%experimental

