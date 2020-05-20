clear all
%clc
load('D:\Codes\CSCNN\CSCNN_test\phi_suhas\phi_0_10_1089.mat');
all_white_meas = zeros(109,1);
for j = 1:383
    for i = 1:10
        load(['D:\Datasets\CSCNN\UA_data\Day_2\Gaussian\cr_0_10\measures_identifier_',num2str(i-1),'.mat']);
        all_white_meas = all_white_meas + M(j,:)'; 
    end
end
all_white_meas = all_white_meas/3830;
true_meas = sum(phi,2);
for k = 2:109
a(k-1) = (true_meas(k) - true_meas(1))/(all_white_meas(k) - all_white_meas(1));
b(k-1) = true_meas(k) - a(k-1)*all_white_meas(1);
end
save('a_b_cr_0_10.mat','a','b')