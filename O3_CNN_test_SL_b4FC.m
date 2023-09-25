%clc;clear;
%%
img_list=[2	4	7	9	14	18	20	22	23	39	57	62	65	66	67	69	70	83	84	95	101	107	109	115	117	118	134	135	136	138	162	165	170	175	179	188	192	196	201	203	206	211	217	223	226	229	230	231	234	253	258	266	270	272	277	281	284	288	308	314	315	319	322	323	328	333	335	337	338	340	343	346	369	388	391	396	397	398	404	412	414	417	440	447	466	470	471	480	483	500	523	526	528	529	531	541	545	547	550	551];
Datapath2 ='D:\SeokhyeongLee\BP_project\BP_CTT\ctt_v12_IDT_20210703\MNIST_Test_0709';
%X =imread('D:\SeokhyeongLee\BP_project\BP_CTT\ctt_v12_IDT_20210703\MNIST_Test_0705');
ni=28;
for q=1:100
    for p=36:80
        filename = strcat('test_',num2str(img_list(q)),'_',num2str(p),'00nA_1.csv');
%filename = strcat('alex_d1_p3p4_test_gate_3100mVb.csv');
     sig_data(:,:,p,q) = load(fullfile(Datapath2,filename));
     sig_data2(:,:,p,q) = load(fullfile(Datapath2,filename));

    end   
end
for q=1:100
    for p=36:80
        temp(:,:,p,q)=reshape(sig_data2(:,2,p,q),[28,28]);
        temp2(:,:,p,q)=transpose(temp(:,:,p,q));
    
        for i=1:28
            for j=1:28
                sig_data(i+(j-1)*28,2,p,q)=temp2(i,j,p,q);
    
        
            end
        end
    end 
end
%% cov layer
K_test1=[1.9539    0.6355   -0.9577;    2.0035    0.4924   -0.8589;    1.6262    0.1575   -0.7946];
K_test2=[3.4770    1.5833   -0.7730;    3.7587    1.6116   -1.0910;    4.7789    1.7567   -1.7929];

t_X =imread('D:\SeokhyeongLee\BP_project\BP_CTT_Paper\CNN\MNIST\mnistasjpg\testSet\testSet\img_480.jpg');
imag1=double(t_X)./255;

test_sig(:,:)=sig_data(:,:,80);
test_sig(:,2)=sig_data(:,2,80)./(8e-6);

for i=1:ni-2
    for j=1:ni-2
        Patch11=[imag1(i,j),imag1(i,j+1),imag1(i,j+2);imag1(i+1,j),imag1(i+1,j+1),imag1(i+1,j+2);imag1(i+2,j),imag1(i+2,j+1),imag1(i+2,j+2)];
        test_conv1(i,j)=sum(sum(Patch11.*K_test1));
        test_conv2(i,j)=sum(sum(Patch11.*K_test2));
    end
end




K1=[78	25	-38;80	20	-34;65	6	-32];

%F12=[3.5320 3.5640;3.3379 3.5785];
K2=[58	26	-13; 63 27	-18;80	29	-30];
%b1=cnn.layers{1, 2}.b;
b1=[[-1.460391990243832e-05,-0.004625446564373]];
K11=[78	75	1; 80	60	40; 65	46	40];
K12=[1	50	38; 1	40	74; 1	40	72];
K21=[58	76	40; 63	77	50; 80	79	 40];
K22=[1	50	53; 1	50	68; 1	50	70];

test_S11=max(0,test_conv1+b1(1));
test_S12=max(0,test_conv2+b1(2));




nk=size(K1);
ni=sqrt((length(sig_data(:,1,1,1))));
for q=1:100
for i=1:ni-2
    for j=1:ni-2
        conv1(i,j,q)=(sig_data(ni*(i-1)+j,2,K11(1,1),q)-sig_data(ni*(i-1)+j,2,K12(1,1),q))+(sig_data(j+1+ni*(i-1),2,K11(1,2),q)-sig_data(j+1+ni*(i-1),2,K12(1,2),q))+(sig_data(j+2+ni*(i-1),2,K11(1,3),q)-sig_data(j+2+ni*(i-1),2,K12(1,3),q))+(sig_data(j+ni*(i),2,K11(2,1),q)-sig_data(j+ni*(i),2,K12(2,1),q))+(sig_data(j+1+ni*(i),2,K11(2,2),q)-sig_data(j+1+ni*(i),2,K12(2,2),q))+(sig_data(j+2+ni*(i),2,K11(2,3),q)-sig_data(j+2+ni*(i),2,K12(2,3),q))+(sig_data(j+ni*(i+1),2,K11(3,1),q)-sig_data(j+ni*(i+1),2,K12(3,1),q))+(sig_data(j+1+ni*(i+1),2,K11(3,2),q)-sig_data(j+1+ni*(i+1),2,K12(3,2),q))+(sig_data(j+2+ni*(i+1),2,K11(3,3),q)-sig_data(j+2+ni*(i+1),2,K12(3,3),q));
        conv2(i,j,q)=(sig_data(j+ni*(i-1),2,K21(1,1),q)-sig_data(j+ni*(i-1),2,K22(1,1),q))+(sig_data(j+1+ni*(i-1),2,K21(1,2),q)-sig_data(j+1+ni*(i-1),2,K22(1,2),q))+(sig_data(j+2+ni*(i-1),2,K21(1,3),q)-sig_data(j+2+ni*(i-1),2,K22(1,3),q))+(sig_data(j+ni*(i),2,K21(2,1),q)-sig_data(j+ni*(i),2,K22(2,1),q))+(sig_data(j+1+ni*(i),2,K21(2,2),q)-sig_data(j+1+ni*(i),2,K22(2,2),q))+(sig_data(j+2+ni*(i),2,K21(2,3),q)-sig_data(j+2+ni*(i),2,K22(2,3),q))+(sig_data(j+ni*(i+1),2,K21(3,1),q)-sig_data(j+ni*(i+1),2,K22(3,1),q))+(sig_data(j+1+ni*(i+1),2,K21(3,2),q)-sig_data(j+1+ni*(i+1),2,K22(3,2),q))+(sig_data(j+2+ni*(i+1),2,K21(3,3),q)-sig_data(j+2+ni*(i+1),2,K22(3,3),q));
        
    end
end

%relu
S11(:,:,q)=10*0.025/(1e-6)*conv1(:,:,q)+b1(1);
S12(:,:,q)=10*0.0597/(1e-6)*conv2(:,:,q)+b1(2);
S31=S11;
S32=S12;
end
%%
figure('Name','imag');
imagesc(reshape(sig_data(:,2,80,1),[28,28]));
axis equal;
colormap gray
axis off;

figure('Name','S11');
imagesc(transpose(S11(:,:,1)));
axis equal;
colormap gray
axis off;


figure('Name','S12');
imagesc(transpose(S12(:,:,1)));
axis equal;
colormap gray
axis off;



figure('Name','timag');
imagesc(imag1);
axis equal;
colormap gray
%axis off;
grid on;
%figure('Name','tS11');
%imagesc(test_S11);
%axis equal;
%colormap gray
%axis off;
figure();
plot(1:26*26,reshape(S11(:,:,1).',1,[]))
hold on;
plot(1:26*26,reshape(S12(:,:,1).',1,[]))
out_raw_convolution(:,1)=1:26*26;
out_raw_convolution(:,2)=reshape(S11(:,:,1).',1,[]);
out_raw_convolution(:,3)=reshape(S12(:,:,1).',1,[]);
%figure('Name','tS12');
%imagesc(test_S12);
%axis equal;
%colormap gray
%axis off;

%%  average Pooling
for q=1:100
S11(:,:,q)=max(0,S11(:,:,q));
S12(:,:,q)=max(0,S12(:,:,q));

S211(q)=mean(mean(S11(1:13,1:13,q)));
S212(q)=mean(mean(S11(14:26,1:13,q)));
S213(q)=mean(mean(S11(1:13,14:26,q)));
S214(q)=mean(mean(S11(14:26,14:26,q)));
S221(q)=mean(mean(S12(1:13,1:13,q)));
S222(q)=mean(mean(S12(14:26,1:13,q)));
S223(q)=mean(mean(S12(1:13,14:26,q)));
S224(q)=mean(mean(S12(14:26,14:26,q)));


S2(:,q)=transpose([S211(q) S212(q) S213(q) S214(q) S221(q) S222(q) S223(q) S224(q)]);
end

test_S211=mean(mean(test_S11(1:13,1:13)));
test_S212=mean(mean(test_S11(14:26,1:13)));
test_S213=mean(mean(test_S11(1:13,14:26)));
test_S214=mean(mean(test_S11(14:26,14:26)));
test_S221=mean(mean(test_S12(1:13,1:13)));
test_S222=mean(mean(test_S12(14:26,1:13)));
test_S223=mean(mean(test_S12(1:13,14:26)));
test_S224=mean(mean(test_S12(14:26,14:26)));


test_S2=transpose([test_S211 test_S212 test_S213 test_S214 test_S221 test_S222 test_S223 test_S224]);


%%  FC layer

F3_simul=[1.83368428108911,1.74517773560521,-0.322721379236346,-0.407609964048763,4.59671112779894,2.84412572321349,-1.35058761284583,-0.168664114525886;-1.49050984180109,-1.87551726042018,0.734682748793879,0.682611302757359,-4.61535284312297,-2.86077463656027,1.27893659370491,-0.0223055057307822];
F3=[1.788888889	1.661111111	-0.383333333	-0.383333333	4.6	2.811111111	-1.405555556	-0.127777778;-1.405555556	-1.788888889	0.766666667	0.638888889	-4.6	-2.811111111	1.277777778	0];
F1=[14	13	-3	-3	36	22	-11	-1;-11	-14	6	5	-36	-22	10	0];
F11=[74,73,40,40,76,62,40,40;40,40,66,65,40,40,50,40];
F12=[60,60,43,43,40,40,51,41;51,54,60,60,76,62,60,40];
F_fac=0.127777778;


%b3=cnn.layers{1, 4}.W(:,1);
b3=[-6.05981112761357;6.06107829407114];
S3=F3*S2(:,1)+b3
Output=exp(S3)./(1+exp(S3))

test_S3=F3_simul*test_S2+b3
test_Output=exp(test_S3)./(1+exp(test_S3))
