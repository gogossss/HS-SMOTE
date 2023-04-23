clc
clear all

dataname =  {'poker-8-9_vs_5'};

for i = 1:length(dataname)
    Enew = [];
    name = ['data\',dataname{i},'\',dataname{i},'.txt'];
    ndata=load(name); 
    N1=size(ndata,1);
    N2=size(ndata,2);
    data=zscore(ndata(:,1:N2-1));
    [coeff,score,latent,tsquared,explained,mu]=pca(data);
    res=score(:, 1:2);
    res(:,3)=ndata(:,N2);

    rng(42)
    p1=1; %域影响因子  
    p2=1; %边界权重因子
    p3=0; %点数量权重因子
    x=res(:,1); 
    y=res(:,2);
    z=res(:,3); %分类标签
    [r,c]=meshgrid(1:N1); 
    a=sqrt((x(r)-x(c)).^2+(y(r)-y(c)).^2);%生成距离矩阵
    res1=find(res(:,3)==1);
    min_res=find(res(:,3)==1);
    min_x=res(min_res,1);
    min_y=res(min_res,2);
    [R,C]=meshgrid(1:length(min_res)); 
    min_a=sqrt((min_x(R)-min_x(C)).^2+(min_y(R)-min_y(C)).^2);%生成距离矩阵
    [m,n]=size(a); 

    ccsb=mean(min_a(:))/mean(a(:))

    a(a==0)=inf;
    min_a(min_a==0)=inf;
    [minv ind]=min(a,[],2);
    [minv2 ind]=min(min_a,[],2);
    ccsa=mean(minv2(:))/mean(minv(:))
    
    %p1=ccsa %域影响因子
    p1=ccsb
    %p1 = ccsb+(ccsa-ccsb)*0.75
    %p1 = ccsb+(ccsa-ccsb)*0.5
    %p1 = ccsb+(ccsa-ccsb)*0.25
    rc=mean(minv(:))*p1; %应该根据少数类样本点的平均距离来改进
    dy=2*rc;dx=rc*sqrt(3);%正六边形的大小
    AI=pi/3*[1:7];%圆周六等分
    cow=(max(y)-min(y));
    rol=(max(x)-min(x));
    num=1;
    TIP={};DATA={};ban3={};ban4={};
    for yk=[-cow:dy:cow]
        yfun=inline(['sqrt(3)*x/3+',num2str(yk)]);
        for xk=[-rol:dx:rol]
            xp=xk;
            yp=yfun(xp);
            if -rol<xp && xp<rol && -cow<yp && yp<cow
                T=[xp+1i*yp]+rc*exp(1i*AI)*2/sqrt(3);
                %plot(T);
                hold on
                Tx=real(T);
                Ty=imag(T);
                ban3=[ban3,{Tx}];
                ban4=[ban4,{Ty}];
                %text(xp,yp,num2str(num)); %标号
                ban1=[];ban2=[];ban5=[];ban6=[];
                for I=1:1:m-numel(i)
                    in=inpolygon(x(I),y(I),Tx,Ty); 
                    if in==1
                       ban1=[ban1,I];
                       ban2=[ban2,res(I,:)'];
                       if res(I,3)'==1
                           ban5=[ban5,I];
                       elseif res(I,3)'==0
                           ban6=[ban6,I];
                       end
                    end
                    TIP{num}=ban1; %储存每个域内点的标签
                    DATA{num}=ban2; %储存每个域内点的数据
                    TIP1{num}=ban5;
                    TIP2{num}=ban6;
                end
                num=num+1;
            end
        end
    end

    axis square 

    p=find(res(:,3)==0);
    q=find(res(:,3)==1);
    plot(x(q),y(q),'r*') 
    hold on
    plot(x(p),y(p),'b*')
    hold on


    YU0=[];
    YU1=[];
    YU2=[];
    YUdis=[];
    CENx=[];CENy=[];
    for NUM=1:1:num-1
     d=DATA{1,NUM};
     if isempty(d)
         YU0=[YU0,NUM];
     elseif d(3,:)==1
         YU1=[YU1,NUM];
     elseif d(3,:)==0
         YU2=[YU2,NUM];
     else
         YUdis=[YUdis,NUM];
     end
     CENx(NUM)=mean(cell2mat(ban3(NUM))); %域中心点横坐标
     CENy(NUM)=mean(cell2mat(ban4(NUM))); %域中心点纵坐标

    end
    [r2,c2]=meshgrid(1:num-1);
    a2=sqrt((CENx(r2)-CENx(c2)).^2+(CENy(r2)-CENy(c2)).^2); %域中心点距离矩阵（域距离矩阵）


    % 区别少数类与多数类，并确定应该过采样的数量，
    Num=0;
    s1=size(p,1);
    s2=size(q,1);
    if s1-s2<0
        Num=s2-s1;
        fprintf('0为少数类标签')
        min1=p;
        maj1=q;
        YUmin=YU2;
        YUmax=YU1;
        ms=0;
        md=1;
        TIP0=TIP2;
    elseif s1-s2>0
        Num=s1-s2;
        fprintf('1为少数类标签')
        min1=q;
        maj1=p;
        YUmin=YU1;
        YUmax=YU2;
        ms=1;
        md=0;
        TIP0=TIP1;
    elseif s1-s2==0
        error('数据已达平衡')
    end

    Num=Num+1500; %调整生成点的个数

    a3=a2;
    a3(find(a3==0))=inf;
    for NUM2=1:1:num-1
        B=a3(NUM2,:);
        [B11,B12]=sort(B,'ascend');
        ind2=B12(1:6);
        if ismember(ind2,YU1)==1 & ismember(NUM2,YU1)==0 %周围全是1域但自己不是1域时，转化为1域
              YU2(find(YU2==NUM2))=[];
              YUdis(find(YUdis==NUM2))=[];
              YU1=[YU1,NUM2];
              %d1=DATA{NUM2}; %该域中点的信息
              %find(d1(3,:)==-1)=[] %寻找其中非1域的点进行删除
              %DATA{NUM2}=d1;
        elseif ismember(ind2,YU2)==1 & ismember(NUM2,YU2)==0%周围全是-1域但自己不是-1域时，转化为-1域
              YU1(find(YU1==NUM2))=[];
              YUdis(find(YUdis==NUM2))=[];
              YU2=[YU2,NUM2];
              %d2=DATA{NUM2}; %该域中点的信息
              %find(d2(3,:)==1)=[] %寻找其中非-1域的点进行删除
              %DATA{NUM2}=d2;
        elseif ismember(NUM2,YU0)==1 %对于一个空域
            if ismember(ind2,YU1)==1 %周围全是1域时转化为空的1域
               YU0(find(YU0==NUM2))=[];
               YU1=[YU1,NUM2];
            elseif ismember(ind2,YU2)==1 %周围全是-1域时转化为空的1域
               YU0(find(YU0==NUM2))=[];
               YU2=[YU2,NUM2];
            elseif ismember(ind2,YU2)==1 %周围全是争议域时转化为空的争议域
               YU0(find(YU0==NUM2))=[];
               YUdis=[YUdis,NUM2];
            end
        end
    end

    P=[];prob=[];
    for NUM3=1:1:num-1 %打分规则需要考虑
        B2=a3(NUM3,:);
        [B21,B22]=sort(B2,'ascend');
        ind3=B22(1:6);
        P1=[];P2=[];
        if ismember(NUM3,YUmin)==1 % 选到少数类域
           for i3=1:1:6
               if ismember(ind3(i3),YUmin)==1
                   P1=[P1,0.25];
               elseif ismember(ind3(i3),YUdis)==1
                   P1=[P1,0.1];
               else
                   P1=[P1,0];
               end
           end
           P(NUM3)=1+sum(P1);
        elseif ismember(NUM3,YUdis)==1 %选到争议域
           for i4=1:1:6
               if ismember(ind3(i4),YUmin)==1
                   P2=[P2,0.25];
               elseif ismember(ind3(i4),YUdis)==1
                   P2=[P2,0.1];
               else
                   P2=[P2,0];
               end
           end
           P(NUM3)=0.5+sum(P2);
        else
           P(NUM3)=0;
        end
    end

    P2=[];
    for NUM6=1:1:num-1
        for NUM12=1:1:6
            B4=a3(NUM6,:);
            [B41,B42]=sort(B4,'ascend');
            ind5=B42(1:6);
            if isempty(TIP{NUM6}) || ismember(NUM6,YUmax)==1 || ismember(NUM6,YU0)==1 %考虑域内没有点的情况，比如空少数类域影响选取概率但不参与选取
               P2(NUM6)=0;
            %elseif ismember(NUM6,YUmin)==1 & isempty(TIP{ind5(NUM12)})==1 %孤立的不参与合成
            %length(TIP{NUM6})==1 %1
               %P2(NUM6)=0;   
            else
               P2(NUM6)=P(NUM6)*(p2.^P(NUM6)); %这里做了改动要注意数据
            end
        end
    end
    Psum1=sum(P2);
    prob1=P2/(Psum1);%归一化得概率 

    %如果少数类域周围

    for NUM5=1:1:Num+1 %要避免多次选到同一个域，每次迭代应该删除被选域的num
        d21=[];d22=[];
        PP=[1:num-1];
        S1=randsrc(1,1,[PP;prob1]); %依概率选取第一个域
        d21=TIP0{S1}; %第一个域中少数类点的标签
        K1(NUM5)=d21(int32(1+(length(d21)-1)*rand)); %随机选第一个点
        %在第一个域及其周围6个域中依概率选取第二个域
        B3=a3(S1,:);
        [B31,B32]=sort(B3,'ascend');
        ind4=B32(1:6);
        P3=[];K2=[];
        for NUM7=1:1:6
            if isempty(TIP{ind4(NUM7)}) || ismember(ind4(NUM7),YUmax)==1 || ismember(ind4(NUM7),YU0)==1
                S2=[];
            else
                S2=ind4(NUM7);
                d22=TIP0{S2};
                if isempty(d22)==0 & d22~=0
                   K2(NUM7)=d22(int32(1+(length(d22)-1)*rand));
                else
                   K2(NUM7)=[];
                end
            end
        end
        %smote合成新样本
        TIPT=[];
        for NUM11=1:1:length(K2)     
            if K2(NUM11)==0 
               TIPT=[TIPT,NUM11];
            end
        end
        K2(TIPT)=[];
        K=[K1(NUM5),K2];
        ecoli=ndata(:,1:N2-1);
        Enew(1,:)=ecoli(K(1),:);
        x2(1)=x(K(1));
        y2(1)=y(K(1));
        for NUM8=1:1:length(K)-1 
            p4=1;
            %p4=length(TIP0(NUM5))/length(DATA(NUM5));
            p5=floor(10*(p4^p3));
            x2(NUM8+1)=x2(NUM8)+rand(1)*(p5/10)*((x(K(NUM8+1)))-x2(NUM8)); 
            y2(NUM8+1)=y2(NUM8)+rand(1)*(p5/10)*((y(K(NUM8+1)))-y2(NUM8));
            Enew(NUM8+1,:)=Enew(NUM8,:)+rand(1)*(p5/10)*(ecoli(K(NUM8+1),:)-(Enew(NUM8,:))); 
        end
        x3(NUM5)=x2(length(K));
        y3(NUM5)=y2(length(K));
        Enew(NUM5,:)=Enew(length(K),:);
    end
    x3(1)=[];
    y3(1)=[];
    Enew(1,:)=[];

    %plot(x3,y3,'^') %合成新样本点
    hold on

    %如果生成点周围6个域都没有少数类点，那么删除这个生成的新样本点
    TIPN=[];
    for NUM9=1:1:Num 
        for NUM10=1:1:num-1
            in2=inpolygon(x3(NUM9),y3(NUM9),ban3{NUM10},ban4{NUM10});
            if in2==1 %判断新样本点在哪个域
               B4=a3(NUM10,:);
               [B41,B42]=sort(B4,'ascend');
               ind5=B42(1:6);
               D=DATA(ind5);
               D1=cell2mat(D);
               if isempty(cell2mat(D)) | D1(3,:)==md
                   if ismember(NUM9,TIPN)==0 
                      TIPN=[TIPN,NUM9];
                   end
               end
            end
        end
    end
    x3(TIPN)=[];
    y3(TIPN)=[];
    Enew(TIPN,:)=[];
    x3=x3';
    y3=y3';
    %合成样本图· 
    plot(x3,y3,'g>') %合成新样本点
    hold off
    Enew8=linspace(ms,ms,size(Enew,1))';
    Enewnew=[Enew,Enew8];
    datanew=[ndata',Enewnew']';%平衡数据集

    Data=datanew(:,1:N2-1);
    Tip=datanew(:,N2);

    datanew2 = datanew(1:(2*size(p,1)),:);
    name2 = ['D:\HS-SMOTE\实验数据\',dataname{i},'new0.xls'];
    s = xlswrite(name2, datanew2, 'sheet1');
end

%pathname='D:\python\数据\';
%filename='data.mat';
%save([pathname,filename],'datanew');
