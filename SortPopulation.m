function [pop,SortOrder]= SortPopulation(pop)
    % Ŀ��ֵ��������
    Costs=[pop.Cost];
    [~,SortOrder]=sort(Costs);
    pop=pop(SortOrder);
end