function NB=Neiborhood(cityNum,neiborNum)
    % ����һ��2-opt ���򽻻�λ��
    ik=1;
    NB=zeros(neiborNum,2);  % ��������
    while ik<=neiborNum     % ������������ֱ������neiborNum������Ϊֹ
        M=ceil(cityNum*rand(1,2));  % ��������������ظ�λ��
        if M(1)~=M(2)
            NB(ik,1)=min(M);
            NB(ik,2)=max(M);
            if ik==1
                isdel=0;        % ��һ�����򽻻�����Ҫ���
            else
                for jk=1:ik-1   % �������򽻻���Ҫ����Ƿ��ظ�
                    if NB(ik,1)==NB(jk,1) && NB(ik,2)==NB(jk,2)
                        isdel=1;    % ����ظ�����ɾ��,�ظ����
                        break;
                    else
                        isdel=0;    % ������ظ��������
                    end
                end
            end
            if ~isdel
                ik=ik+1;       % ������ظ��������
            else
                continue;        % ����ظ����������������
            end
        else
            continue;        % ����������λ����ͬ���������������
        end
    end
end