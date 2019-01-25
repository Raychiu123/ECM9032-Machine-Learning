clear;
clc;
[num,txt,raw] = xlsread('D:\ML\[2017]ML_HW3\[2017]ML_HW3\Dataset\Problem2\Iris_data.xls');

%% The first two features
x1= num(:,1:2);
x2= cat(2,x1(:,1).^2,sqrt(2).*x1(:,1).*x1(:,2));
x1= cat(2,x2,x1(:,2).^2);

t1 = zeros(150,1); %setosa for 1 and -1
t2 = zeros(150,1); %vriginic for 1 and -1
t3 = zeros(150,1); %versicol for 1 and -1
t4 = zeros(150,1); %setosa for 1 and 0
t5 = zeros(150,1); %vriginic for 1 and 0
t6 = zeros(150,1); %versicol for 1 and 0

for i=1:150
    %setosa
    ans = (raw{i+1,5}=='O');
    if (ans(4)==1)
        t1(i)=1;
        t4(i)=1;
    else
        t1(i)=-1;
        t4(i)=0;
    end
    %vriginic
    ans = (raw{i+1,5}=='G');
    if (ans(4)==1)
        t2(i)=1;
        t5(i)=1;
    else
        t2(i)=-1;
        t5(i)=0;
    end
    %versicol
    ans = (raw{i+1,5}=='S');
    if (ans(4)==1)
        t3(i)=1;
        t6(i)=1;
    else
        t3(i)=-1;
        t6(i)=0;
    end
end

k1 = x1*(x1.');
[alpha1,basis1] = smo(k1,t1.',1000,0.001);
[alpha2,basis2] = smo(k1,t2.',1000,0.001);
[alpha3,basis3] = smo(k1,t3.',1000,0.001);

w1 = zeros(1,3);
w2 = zeros(1,3);
w3 = zeros(1,3);

for i=1:150
    w = alpha1(i)*t1(i)*x1(i,:);
    w1 = w1+w;
    w = alpha2(i)*t2(i)*x1(i,:);
    w2 = w2+w;
    w = alpha3(i)*t3(i)*x1(i,:);
    w3 = w3+w;
end

y1 = zeros(150,1);
y2 = zeros(150,1);
y3 = zeros(150,1);
for i=1:150
    for j=1:150
        y = alpha1(1,j)*t1(j)*(x1(i,:))*(x1(j,:).');
        y1(i,1) = y1(i,1)+y;
        y = alpha2(1,j)*t2(j)*(x1(i,:))*(x1(j,:).');
        y2(i,1) = y2(i,1)+y;
        y = alpha3(1,j)*t3(j)*(x1(i,:))*(x1(j,:).');
        y3(i,1) = y3(i,1)+y;
    end
    y1(i,1) = y1(i,1)+basis1;
    y2(i,1) = y2(i,1)+basis2;
    y3(i,1) = y3(i,1)+basis3;
end

test1 = zeros(150,1);
for i=1:150
    max = y1(i,1);
    test1(i,1) =1;
    if y2(i,1)>=max
        test1(i,1)=2;
        max = y2(i,1);
    end
    if y3(i,1)>=max
        test1(i,1)=3;
        max = y3(i,1);
    end
end

m1 = (test1==1);
m2 = (test1==2);
m3 = (test1==3);

c1 = zeros(1,401);
c2 = 1.5:0.01:5.5;
c = cat(1,c1,c2);
for i=1:401
    j=i;
    c3 = ones(1,401)*4 + (j/100);
    c4 = cat(1,c3,c2);
    c = cat(1,c,c4);
end
c1 = c;
c = reshape(c,2,804*401/2);
c = c.';
c2= cat(2,c(:,1).^2,sqrt(2).*c(:,1).*c(:,2));
c= cat(2,c2,c(:,2).^2);
c = c.';

y4 = zeros(401,1);
y5 = zeros(401,1);
y6 = zeros(401,1);
for i=1:401*804/2
    y4(i,1) = w1*(c(:,i))+basis1;
    y5(i,1) = w2*(c(:,i))+basis2;
    y6(i,1) = w3*(c(:,i))+basis3;
end
test2 = zeros(401,1);
for i=1:401*804/2
    max = y4(i,1);
    test2(i,1) =1;
    if y5(i,1)>=max
        test2(i,1)=2;
        max = y5(i,1);
    end
    if y6(i,1)>=max
        test2(i,1)=3;
        max = y6(i,1);
    end
end

c = reshape(c1,2,804*401/2);
m4 = (test2==1);
m5 = (test2==2);
m6 = (test2==3);
c = c.';
x1= num(:,1:2);
sup1 = (alpha1~=0); sup1 = cat(2,(sup1.').*x1(:,1),(sup1.').*x1(:,2));
sup2 = (alpha2~=0); sup2 = cat(2,(sup2.').*x1(:,1),(sup2.').*x1(:,2));
sup3 = (alpha3~=0); sup3 = cat(2,(sup3.').*x1(:,1),(sup3.').*x1(:,2));

%real data
r1 = x1.*t1;
r2 = x1.*t2;
r3 = x1.*t3;

% scatter(c(:,1).*m4);
hold on 
figure(1),
plot(c(:,1).*m4,c(:,2).*m4,'c.',c(:,1).*m5,c(:,2).*m5,'m.',c(:,1).*m6,c(:,2).*m6,'y.')
plot(x1(:,1).*m1,x1(:,2).*m1,'rx',x1(:,1).*m2,x1(:,2).*m2,'g+',x1(:,1).*m3,x1(:,2).*m3,'b*')
% plot(r1(:,1),r1(:,2),'ro',r2(:,1),r2(:,2),'go',r3(:,1),r3(:,2),'bo')
plot(sup1(:,1),sup1(:,2),'k^',sup2(:,1),sup2(:,2),'k^',sup3(:,1),sup3(:,2),'k^')
xlabel('sepal length'),ylabel('sepal width')
axis([4,8,1.5,5.5]);
legend('show')
hold off

%% LDA dimension reduction

m = mean(num);
m1 = sum(num.*t4)/sum(t4);
m2 = sum(num.*t5)/sum(t5);
m3 = sum(num.*t6)/sum(t6);
x1 = num.*t4;  x1(find(x1==0))=[];  x1 = reshape(x1,50,4);
x2 = num.*t5;  x2(find(x2==0))=[];  x2 = reshape(x2,50,4);
x3 = num.*t6;  x3(find(x3==0))=[];  x3 = reshape(x3,50,4);
sw = (x1-m1).'*(x1-m1)+(x2-m2).'*(x2-m2)+(x3-m3).'*(x3-m3);
sb = 50*((m1-m).')*((m1-m))+50*((m2-m).')*((m2-m))+50*((m3-m).')*((m3-m));
j = inv(sw)*sb; 
[V,D] = eig(j);
lamda = eig(j);
v1 = V(:,1); v2 = V(:,2);

dim_x1 = num*v1;
dim_x2 = num*v2;

x1= cat(2,dim_x1,dim_x2);
x3 =x1;
x2= cat(2,x1(:,1).^2,sqrt(2).*x1(:,1).*x1(:,2));
x1= cat(2,x2,x1(:,2).^2);


k1 = x1*(x1.');
[alpha1,basis1] = smo(k1,t1.',1000,0.001);
[alpha2,basis2] = smo(k1,t2.',1000,0.001);
[alpha3,basis3] = smo(k1,t3.',1000,0.001);

w1 = zeros(1,3);
w2 = zeros(1,3);
w3 = zeros(1,3);

for i=1:150
    w = alpha1(i)*t1(i)*x1(i,:);
    w1 = w1+w;
    w = alpha2(i)*t2(i)*x1(i,:);
    w2 = w2+w;
    w = alpha3(i)*t3(i)*x1(i,:);
    w3 = w3+w;
end

y1 = zeros(150,1);
y2 = zeros(150,1);
y3 = zeros(150,1);
for i=1:150
    for j=1:150
        y = alpha1(1,j)*t1(j)*(x1(i,:))*(x1(j,:).');
        y1(i,1) = y1(i,1)+y;
        y = alpha2(1,j)*t2(j)*(x1(i,:))*(x1(j,:).');
        y2(i,1) = y2(i,1)+y;
        y = alpha3(1,j)*t3(j)*(x1(i,:))*(x1(j,:).');
        y3(i,1) = y3(i,1)+y;
    end
    y1(i,1) = y1(i,1)+basis1;
    y2(i,1) = y2(i,1)+basis2;
    y3(i,1) = y3(i,1)+basis3;
end

test1 = zeros(150,1);
for i=1:150
    max = y1(i,1);
    test1(i,1) =1;
    if y2(i,1)>=max
        test1(i,1)=2;
        max = y2(i,1);
    end
    if y3(i,1)>=max
        test1(i,1)=3;
        max = y3(i,1);
    end
end

m1 = (test1==1);
m2 = (test1==2);
m3 = (test1==3);

c1 = zeros(1,501);
c2 = -5:0.01:0;
c = cat(1,c1,c2);
for i=1:501
    j=i;
    c3 = ones(1,501)*-2 + (j/100);
    c4 = cat(1,c3,c2);
    c = cat(1,c,c4);
end

c1 = c;
c = reshape(c,2,1004*501/2);
c = c.';
c2= cat(2,c(:,1).^2,sqrt(2).*c(:,1).*c(:,2));
c= cat(2,c2,c(:,2).^2);
c = c.';

y4 = zeros(501,1);
y5 = zeros(501,1);
y6 = zeros(501,1);
for i=1:501*1004/2
    y4(i,1) = w1*(c(:,i))+basis1;
    y5(i,1) = w2*(c(:,i))+basis2;
    y6(i,1) = w3*(c(:,i))+basis3;
end
test2 = zeros(501,1);
for i=1:501*1004/2
    max = y4(i,1);
    test2(i,1) =1;
    if y5(i,1)>=max
        test2(i,1)=2;
        max = y5(i,1);
    end
    if y6(i,1)>=max
        test2(i,1)=3;
        max = y6(i,1);
    end
end

c = reshape(c1,2,1004*501/2);
m4 = (test2==1);
m5 = (test2==2);
m6 = (test2==3);
c = c.';
sup1 = (alpha1~=0); sup1 = cat(2,(sup1.').*x3(:,1),(sup1.').*x3(:,2));
sup2 = (alpha2~=0); sup2 = cat(2,(sup2.').*x3(:,1),(sup2.').*x3(:,2));
sup3 = (alpha3~=0); sup3 = cat(2,(sup3.').*x3(:,1),(sup3.').*x3(:,2));

%real data
r1 = x3.*t4;
r2 = x3.*t5;
r3 = x3.*t6;

hold on 
figure(2),
plot(c(:,1).*m4,c(:,2).*m4,'c.',c(:,1).*m5,c(:,2).*m5,'m.',c(:,1).*m6,c(:,2).*m6,'y.')
plot(x3(:,1).*m1,x3(:,2).*m1,'rx',x3(:,1).*m2,x3(:,2).*m2,'g+',x3(:,1).*m3,x3(:,2).*m3,'b*')
% plot(r1(:,1),r1(:,2),'ro',r2(:,1),r2(:,2),'go',r3(:,1),r3(:,2),'bo')
plot(sup1(:,1),sup1(:,2),'k^',sup2(:,1),sup2(:,2),'k^',sup3(:,1),sup3(:,2),'k^')
xlabel('sepal length'),ylabel('sepal width')
axis([-2,3,-3.5,-0.75]);
% legend('show')
hold off
