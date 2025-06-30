function [currentBestV,currentBestSol]=local_search(Xdata,city_tour)
    % ¾Ö²¿ËÑË÷
    currentBestSol=city_tour;           % ³õÊ¼»¯
    best_route=city_tour;
    currentBestV=distance_calc(Xdata,best_route);
    matrixSize=length(best_route)-1;
    for i=1:matrixSize-2            % Ñ­»·ÁÚÓò½»»»ËÑË÷
        for j=1:matrixSize-1
            best_route(1,i:j+1)=flip(best_route(1,i:j+1));
            best_route(1,end)=best_route(1,1);
            calResult=distance_calc(Xdata,best_route);
            if calResult<currentBestV
                currentBestV=calResult; %
                currentBestSol=best_route;
            end
            best_route=city_tour;       % »Ö¸´³õÊ¼×´Ì¬
        end
    end
end