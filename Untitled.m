p=[-1,1;-1,1]	 % ����������������������������ȡֵ��Χ��Ϊ-1~1
 t=1;	 
net=newp(p,t);             % ����
 P=[0,0,1,1;0,1,0,1]        % ѵ������
 T=[0,1,1,1]	           % Ŀ�����  �� �߼�
 net=train(net,P,T); 
 newP=[0,0.9];             % �µ�����
 newT=sim(net,newP)    % ����
 newP=[0.9,0.9]';	 
 newT=sim(net,newP) 
  newT=sim(net,P) 