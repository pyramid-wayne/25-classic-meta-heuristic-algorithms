function ObjVal=Rosenbrock(Chrom)
% Rosenbrock function
% Chrom is a vector of two variables
% ObjVal is the value of the function
    Dim=size(Chrom,2);
    [~,Nvar]=size(Chrom);
    x1=Chrom(:,1:Nvar-1);
    x2=Chrom(:,2:Nvar);
    if Dim==2
        ObjVal=100*(x2-x1.^2).^2+(1-x1).^2;     % 两个变量
    else
        ObjVal=sum((100*(x2-x1.^2).^2+(1-x1).^2)')';        % 多个变量
    end

end