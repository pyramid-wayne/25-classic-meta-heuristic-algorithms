function results = goldstein_priceFunc(xx,pop)
    % Goldstein-Price function
    % Input: xx - vector of variables[x1,x2];pop 种群数量
    % Output: results - function value；min=3，x1=0.5，x2=0.25
    results=zeros(1,pop);
    for i=1:pop
        x1bar=4*xx(i,1)-2;
        x2bar=4*xx(i,2)-2;
        fact1a=(x1bar+x2bar+1)^2;
        fact1b=19-14*x1bar+3*x1bar^2-14*x2bar+6*x1bar*x2bar+3*x2bar^2;
        fact1=fact1a*fact1b+1;
        fact2a=(2*x1bar-3*x2bar)^2;
        fact2b=18-32*x1bar+12*x1bar^2+48*x2bar-36*x1bar*x2bar+27*x2bar^2;
        fact2=fact2a*fact2b+30;
        prod=fact1*fact2;
        results(i)=prod;
    end
end