clear 
close all;

addpath(genpath('D:\Others_codes\Deep_learning\caffe-windows-master\matlab'))
addpath('D:\Codes\CSCNN\NLR_codes\NLR_CS');
addpath('D:\Codes\CSCNN\D-AMP_Toolbox');
rate = 0.01;
load('D:\Codes\CSCNN\CSCNN_test\phi_suhas\phi_0_10_1089.mat');
folder_model = 'D:\Codes\CSCNN\CSCNN_test\caffe_model\';
folder_prototxt = 'D:\Codes\CSCNN\CSCNN_test\Deploy_prototxt\';

dest_folder = 'D:\Datasets\CSCNN\Real_data_results_cr_0_01\';
folder_nlr_cs = [dest_folder 'nlrcs\'];
folder_d_amp = [dest_folder 'damp\'];
folder_cscnn = [dest_folder 'cscnn\'];
folder_alinet = [dest_folder 'alinet\'];
folder_tval3 = [dest_folder 'tval3\'];

load D:\Datasets\CSCNN\UA_data_results\a_b_cr_0_10.mat

addpath(genpath('D:\Codes\CSCNN\NLR_codes\NLR_CS'));
addpath(genpath('D:\Codes\CSCNN\D-AMP_Toolbox'));
addpath(genpath('D:\Codes\CSCNN\TV_AL3\TVAL3_beta2.4'));

opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;

% try
% caffe.reset_all();
% catch
% caffe.reset_all();
% end
% caffe.set_mode_gpu();

% model_cscnn = [folder_prototxt 'SRCNN_net_m_to_op_cr_0_10.prototxt'];
%  model_cscnn = [folder_prototxt 'SRCNN_net_m_to_op_cr_0_04.prototxt'];
% model_cscnn = [folder_prototxt 'SRCNN_net_m_to_op_cr_0_01.prototxt'];

% weights_cscnn =[folder_model 'CSCNN_0_10_11_7_m_to_op_phi_init_iter_632000.caffemodel'];
%  weights_cscnn =[folder_model 'CSCNN_0_04_11_7_m_to_op_for_real_iter_492000.caffemodel'];
%  weights_cscnn =[folder_model 'CSCNN_0_01_11_7_m_to_op_for_real_iter_453000.caffemodel'];

% net_cscnn = caffe.Net(model_cscnn,weights_cscnn,'test');

% model_alinet = [folder_prototxt 'SRCNN_net_baranuik_cr_0_10.prototxt'];
% model_alinet = [folder_prototxt 'SRCNN_net_baranuik_cr_0_04.prototxt'];
% model_alinet = [folder_prototxt 'SRCNN_net_baranuik_cr_0_01.prototxt'];
 
% weights_alinet =[folder_model 'CSCNN_0_10_baranuik_iter_12632000.caffemodel'];
% weights_alinet =[folder_model 'CSCNN_0_04_baranuik_iter_15000000.caffemodel'];
% weights_alinet =[folder_model 'CSCNN_0_01_baranuik_iter_24144000.caffemodel'];

% net_alinet = caffe.Net(model_alinet,weights_alinet,'test');

impath = 'D:\Datasets\CSCNN\UA_data\Day_2\Gaussian\cr_0_10\';
calibration_path = 'D:\Datasets\CSCNN\UA_data\Day_2\Gaussian\';
load([calibration_path,'calibration.mat']);

num_blks = length(calibration.blocks);
 for i = 1:num_blks
    x_blk_ind(i) = calibration.blocks(i).xBlock;
    y_blk_ind(i) = calibration.blocks(i).yBlock;
 end
num_blk_x = max(x_blk_ind) - min(x_blk_ind) + 1;
num_blk_y = max(y_blk_ind) - min(y_blk_ind) + 1;
x_blk_ind = x_blk_ind - min(x_blk_ind)+1;
y_blk_ind = y_blk_ind - min(y_blk_ind)+1;

sum_M  = zeros(num_blks,109);

image_names = {'barbara','boats','cameraman','foreman','house'};

for image_number = 1:5
    for set = 0:9
        load([impath, image_names{image_number}, '_',num2str(set),'.mat']);
       sum_M = sum_M + M; 
  
    end
    M = sum_M/10;
    M = a*M+b;
    
    %tic
    for count = 1:size(M,1)
%         temp = net_cscnn.forward({single(M(count,1:10))});
%         im_comp((num_blk_y-y_blk_ind(count))*33+1:(num_blk_y-y_blk_ind(count)+1)*33,(x_blk_ind(count)-1)*33+1:x_blk_ind(count)*33) = temp{1}; 
        
        measurements = 255*M(count,1:10);
        image_trad_cs = phi(1:10,:)'*measurements(:);
        image_trad_cs = reshape(image_trad_cs, [33 33]);
%         temp_nlr_cs = Main_NLR_CS(phi,rate,image_trad_cs, 0);
%         im_comp_nlr_cs((num_blk_y-y_blk_ind(count))*33+1:(num_blk_y-y_blk_ind(count)+1)*33,(x_blk_ind(count)-1)*33+1:x_blk_ind(count)*33) = temp_nlr_cs; 
%        
%disp('kuldeep')
        start_time = tic;
        temp_d_amp = CS_Imaging_Demo(phi(1:10,:), image_trad_cs, 0);
        im_comp_d_amp((num_blk_y-y_blk_ind(count))*33+1:(num_blk_y-y_blk_ind(count)+1)*33,(x_blk_ind(count)-1)*33+1:x_blk_ind(count)*33) = temp_d_amp;
%         toc(start_time);
        
        start_time = tic;
        temp_tval3 =  TVAL3(phi(1:10,:),measurements(:),33,33,opts);
        im_comp_tval3((num_blk_y-y_blk_ind(count))*33+1:(num_blk_y-y_blk_ind(count)+1)*33,(x_blk_ind(count)-1)*33+1:x_blk_ind(count)*33) = temp_tval3;
%         toc(start_time);
    end
   
%     save(strcat(folder_nlr_cs,image_names{image_number} ,'.mat'),'im_comp_nlr_cs');
%     imwrite(im_comp_nlr_cs,strcat(folder_cscnn,image_names{image_number}, '.png'));
    
    save(strcat(folder_d_amp,image_names{image_number} ,'.mat'),'im_comp_d_amp');
    imwrite(uint8(im_comp_d_amp),strcat(folder_d_amp,image_names{image_number}, '.png'));
    
    save(strcat(folder_tval3,image_names{image_number} ,'.mat'),'im_comp_tval3');
    imwrite(uint8(im_comp_tval3),strcat(folder_tval3,image_names{image_number}, '.png'));    
    
    sum_M  = zeros(num_blks,109);
image_number
% pause;
end