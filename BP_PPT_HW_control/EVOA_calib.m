function Vq=EVOA_calib(direct,Ampq)
    %In=importdata('C:\Users\admin\Desktop\Seokhyeong\bP_CTT\ctt_1550_EVOA\test3.dat');
    In=importdata(direct);
    V=In(:,1); Amp=normalize(In(:,2),'range');
    %Amp=In(:,2)/In(1,2);
    Amp=smooth(Amp,0.1,'rloess');
    
    Vq = interp1(Amp,V,Ampq);
    %Vq=V
    figure;
    %plot(V,Amp,Vq,Ampq,"linewidth",2);
    plot(V,Amp,Vq,Ampq,"*");

end