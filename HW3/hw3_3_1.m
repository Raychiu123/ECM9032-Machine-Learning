clear;
clc;
he = imread('D:\ML\[2017]ML_HW3\[2017]ML_HW3\Dataset\Problem3\hw3_img.jpg');
he = imresize(he,0.1);
copy = he;
he = im2double(he);

nrows = 72;
ncols = 128;
indx = zeros(nrows*ncols,1);
a = double(reshape(he,nrows*ncols,3));

k=20;
mu = zeros(k,3);
for i=1:k
    mu(i,1) = randperm(255,1);
    mu(i,2) = randperm(255,1);
    mu(i,3) = randperm(255,1);
end
mu = normr(mu);

x=1;
min3 = zeros(100,1);
while x<20
    for i=1:nrows*ncols
        min = abs(mu(1,1)-a(i,1))+abs(mu(1,2)-a(i,2))+abs(mu(1,3)-a(i,3));
        indx(i)=1;
        for j=2:k
            if (abs(mu(j,1)-a(i,1))+abs(mu(j,2)-a(i,2))+abs(mu(j,3)-a(i,3)))<min
                indx(i)=j;
                min = (abs(mu(j,1)-a(i,1))+abs(mu(j,2)-a(i,2))+abs(mu(j,3)-a(i,3)));
            end
        end
    end
              
    indx1 = indx;
    for i=1:k
        m= (indx1 == i);
        if m~=0
            mu(i,:)=mean(double(m).*a);
        end
    end
    
    min2=0;
    for j=1:k
        m= (indx1 == j);
        m= double(m);
        m1 = sum(m.*(a-mu(j)).^2);
        min1 = m1(1)+m1(2)+m1(3);
        min2 = min1+min2;
    end
    
    min3(x)=min2;
    x=x+1;
end
mu1 = mu;

%% EM algorithm
covr = ones(3,3,k);
for i=1:k
    m= (indx1 == i);
    m= double(m);
    covr(:,:,i)=cov(m.*a)+0.01*eye(3,3);
end
pi = ones(k,1)/k;
%initial state
gama = ones(nrows*ncols,1,k)/k;
n = zeros(1,k);
log_like = zeros(100,1);

x=1;

while x<101
    for i=1:nrows*ncols
        y=0;
        for j=1:k
            for z = 1:k
                y1 = pi(z,1)*mvnpdf(a(i,:),mu(z,:),covr(:,:,z));
                y=y1+y;
            end
            gama(i,:,j)= pi(j,1)*mvnpdf(a(i,:),mu(j,:),covr(:,:,j))/(y);
        end
    end
    
    for i=1:k
        n(:,i)=sum(gama(:,1,i));
    end
    
    covr = zeros(3,3,k); 
    for j=1:k
        co = 0;
        mu(j,:) = sum(gama(:,1,j).*a)/n(1,j);
        for i=1:nrows*ncols
            co = gama(i,:,j)*(a(i,:)-mu(j,:)).'*(a(i,:)-mu(j,:));
            covr(:,:,j) = co + covr(:,:,j);
        end
        covr(:,:,j) = covr(:,:,j)/n(:,j)+eye(3,3)*0.1;
        pi(j,1) = n(:,j)/(nrows*ncols);
    end
    
    for j=1:nrows*ncols
        l=0;
        for i=1:k
        l1 = log(sum(pi(i,1).*mvnpdf(a(j,:),mu(i,:),covr(:,:,i)))) ;
        l = l1+l;
        end
        log_like(x,1) = l+log_like(x,1);
    end
    x= x+1;
end

x=(1:1:100);
figure(1),plot(x,log_like);

ax1 = zeros(9216,3);

for i=1:k
    x1 = (indx==i);
    ax=ones(9216,3);
    ax=ax.*x1.*mu1(i,:);
    ax1 = ax+ax1;
    
end
ax1_1 = ax1(:,1);
ax1_2 = ax1(:,2);
ax1_3 = ax1(:,3);
ax1_1 = reshape(ax1_1,72,128);
ax1_2 = reshape(ax1_2,72,128);
ax1_3 = reshape(ax1_3,72,128);
ax1 = cat(3,ax1_1,ax1_2,ax1_3);
ax1 = im2uint8(ax1);
figure(2), imshow(ax1)
        
        