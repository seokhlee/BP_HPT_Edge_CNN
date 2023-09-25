function [Vc]=convert2V(pix,Ampq,Vq)
    for i=1:length(pix)
        
        k=find(abs(Ampq-pix(i))<0.005);
        k=k(1);
        while isnan(Vq(k))
            if k>=965  
                k=k-1;
            end
            Vc(i,1)=Vq(k);
            
            if k<=2
               k=k+1; 
            end
     
        end 
        Vc(i,1)=Vq(k); 
           
    end
end