function deci=DecodeChromosome(Population,point,length,j)
    % ����Ⱦɫ��
    deci=0;
    % code_2=Population(j,point+1:point+length);
    for i=1:length
        deci=deci+Population(j,point+i)*2^(length-i);
    end
end