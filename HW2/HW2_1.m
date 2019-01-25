a=load('C:\Users\gaexp\OneDrive\Documents\Machine Learning\Homework\[2017]ML_HW2\Data\1_data.mat');
r = a.r2;
x1 = datasample(r,10);
x2 = datasample(r,100);
x3 = datasample(r,500);
m1=(x1-[1,-1])'*(x1-[1,-1])/10
m2=(x2-[1,-1])'*(x2-[1,-1])/100
m3=(x3-[1,-1])'*(x3-[1,-1])/500
pre1 = inv(m1+inv(eye(2))/(1+10-2+1))
pre2 = inv(m2+inv(eye(2))/(1+100-2+1))
pre3 = inv(m3+inv(eye(2))/(1+500-2+1))

r1 = inv((r-[1,-1])'*(r-[1,-1])/1000)