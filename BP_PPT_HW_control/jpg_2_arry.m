%for p =1:length(array01)
   % readname= strcat('img_',num2str(array01(p)),'.jpg');
    readname= strcat('huskyimage.png');
    A = rgb2gray(imread(readname));
    Arry = [];
    for q=1:length(A(:,1))
       % for q=1:28
                   Arry=[Arry,A(q,:)];
      %  end
    end
    savefile=strcat('test_img_husky.csv');
    csvwrite(savefile,transpose(Arry))

%end