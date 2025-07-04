function [pop,SortOrder]= SortPopulation(pop)
    % Ä¿±êÖµÉıĞòÅÅĞò
    Costs=[pop.Cost];
    [~,SortOrder]=sort(Costs);
    pop=pop(SortOrder);
end