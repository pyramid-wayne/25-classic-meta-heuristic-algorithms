function [currentBestV,currentBestSol]=local_search(Xdata,city_tour)
    % �ֲ�����
    currentBestSol=city_tour;           % ��ʼ��
    best_route=city_tour;
    currentBestV=distance_calc(Xdata,best_route);
    matrixSize=length(best_route)-1;
    for i=1:matrixSize-2            % ѭ�����򽻻�����
        for j=1:matrixSize-1
            best_route(1,i:j+1)=flip(best_route(1,i:j+1));
            best_route(1,end)=best_route(1,1);
            calResult=distance_calc(Xdata,best_route);
            if calResult<currentBestV
                currentBestV=calResult; %
                currentBestSol=best_route;
            end
            best_route=city_tour;       % �ָ���ʼ״̬
        end
    end
end