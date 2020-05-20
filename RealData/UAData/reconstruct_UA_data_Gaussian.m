clear
close all
clc

% addpath('/media/slohit/DATAPART1/Codes/CSCNN/NLR_codes/NLR_CS');
% addpath(genpath('/media/slohit/DATAPART1/Codes/CSCNN/D-AMP_Toolbox'));
addpath(genpath('..\TV_AL3\TVAL3_beta2.4'));
load experimentalData_GaussianMeasurements.mat

% Load phi
phi_0_10 = load('..\phi_suhas\phi_0_10_1089.mat');
phi_0_10 = phi_0_10.phi;
phi_0_04 = phi_0_10(1:43,:);
phi_0_01 = phi_0_10(1:10,:);

% Parameters for TVAL3
opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;

num_runs = 10; % Number of runs per measurement

image_names = {'Parrots', 'barbara', 'boats','house','foreman', 'cameraman', 'lena256'};
% image_names = {'Parrots'};
% image_names = {'Monarch'};

% Output folders 
output_folder_0_10_tval3 = 'reconstructed_images_Gaussian_no_calibration\mr_0_10\tval3\';
% output_folder_0_10_damp = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images_Gaussian/mr_0_10/damp/';
% output_folder_0_10_reconnet = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images_Gaussian/mr_0_10/reconnet/';

output_folder_0_04_tval3 = 'reconstructed_images_Gaussian_no_calibration\mr_0_04\tval3\';
% output_folder_0_04_damp = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images_Gaussian/mr_0_04/damp/';
% output_folder_0_04_reconnet = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images_Gaussian/mr_0_04/reconnet/';

output_folder_0_01_tval3 = 'reconstructed_images_Gaussian_no_calibration\mr_0_01\tval3\';
% output_folder_0_01_damp = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images_Gaussian/mr_0_01/damp/';
% output_folder_0_01_reconnet = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images_Gaussian/mr_0_01/reconnet/';

% addpath(genpath('/home/slohit/caffe-master/matlab'))
% try
%     caffe.reset_all();
% catch
%     caffe.reset_all();
% end
% caffe.set_mode_gpu();

% Point to prototxt files
% prototxt_0_10 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/prototxt/SRCNN_net_m_to_op_cr_0_10.prototxt';
% prototxt_0_04 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/prototxt/SRCNN_net_m_to_op_cr_0_04.prototxt';
% prototxt_0_01 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/prototxt/SRCNN_net_m_to_op_cr_0_01.prototxt';

% Point to caffemodel files
% caffemodel_0_10 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/caffemodel/CSCNN_0_10_11_7_m_to_op_phi_init_iter_632000.caffemodel';
% caffemodel_0_04 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/caffemodel/CSCNN_0_04_11_7_m_to_op_for_real_iter_492000.caffemodel';
% caffemodel_0_01 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/caffemodel/CSCNN_0_01_11_7_m_to_op_for_real_iter_453000.caffemodel';

% Create networks
% net_0_10 = caffe.Net(prototxt_0_10, caffemodel_0_10, 'test');
% net_0_04 = caffe.Net(prototxt_0_04, caffemodel_0_04, 'test');
% net_0_01 = caffe.Net(prototxt_0_01, caffemodel_0_01, 'test');


for image = 1:length(image_names)
    cs_meas = zeros(404,109);
    all_cs_meas = getfield(results{1,1}.measures,image_names{image});
        
    for runs = 1:num_runs
        cs_meas = cs_meas + all_cs_meas{1,runs};
    end
    
%     0.10
    min_op_0_10 = -2.5764;
    max_op_0_10 = 2.3096;
    
%     0.04
    min_op_0_04 = -2.2938;
    max_op_0_04 = 2.7070;

%     0.01
    min_op_0_01 = -2.2144;
    max_op_0_01 = 1.1551;
    
    
    calibration = results{1,1}.calibration;
    num_blks = length(calibration.blocks);
    for i = 1:num_blks
        x_blk_ind(i) = calibration.blocks(i).xBlock;
        y_blk_ind(i) = calibration.blocks(i).yBlock;
    end
    num_blk_x = max(x_blk_ind) - min(x_blk_ind) + 1;
    num_blk_y = max(y_blk_ind) - min(y_blk_ind) + 1;
    x_blk_ind = x_blk_ind - min(x_blk_ind)+1;
    y_blk_ind = y_blk_ind - min(y_blk_ind)+1;
    
%     save('/media/slohit/DATAPART1/Codes/CSGAN/mine/UAData_Gaussian/x_blk_ind.mat', 'x_blk_ind')
%     save('/media/slohit/DATAPART1/Codes/CSGAN/mine/UAData_Gaussian/y_blk_ind.mat', 'y_blk_ind')
    
    
    cs_meas_0_10 = cs_meas;
%     save(['/media/slohit/DATAPART1/Codes/CSGAN/mine/UAData_Gaussian/', image_names{image}, '_cs_meas_0_10.mat'],'cs_meas_0_10');
%     cs_meas_0_10 = 10 *cs_meas_0_10;
    
%     cs_meas_0_10 = min_op_0_10 + (max_op_0_10 - min_op_0_10)*(cs_meas_0_10 - min(cs_meas_0_10(:)))/(max(cs_meas_0_10(:)) - min(cs_meas_0_10(:)));
    cs_meas_0_04 = cs_meas(:,1:43);
%     save(['/media/slohit/DATAPART1/Codes/CSGAN/mine/UAData_Gaussian/', image_names{image}, '_cs_meas_0_04.mat'],'cs_meas_0_04');
%     cs_meas_0_04 = 10*cs_meas_0_04;
    
%     cs_meas_0_04 = min_op_0_04 + (max_op_0_04 - min_op_0_04)*(cs_meas_0_04 - min(cs_meas_0_04(:)))/(max(cs_meas_0_04(:)) - min(cs_meas_0_04(:)));
    cs_meas_0_01 = cs_meas(:,1:10);
%     save(['/media/slohit/DATAPART1/Codes/CSGAN/mine/UAData_Gaussian/', image_names{image}, '_cs_meas_0_01.mat'],'cs_meas_0_01');
%     cs_meas_0_01 = 10*cs_meas_0_01;
    
%     cs_meas_0_01 = min_op_0_01 + (max_op_0_01 - min_op_0_01)*(cs_meas_0_01 - min(cs_meas_0_01(:)))/(max(cs_meas_0_01(:)) - min(cs_meas_0_01(:)));
    
    
    %% Reconstruct images
    % MR = 0.10
    for block = 1:size(cs_meas_0_10,1)
%         temp = net_0_10.forward({single(cs_meas_0_10(block,:))});
%         im_recon_0_10((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp{1};
        
%         image_trad_cs = 255*phi_0_10'*transpose(cs_meas_0_10(block,:));
%         image_trad_cs = reshape(image_trad_cs, [33 33]);
%         temp_d_amp = CS_Imaging_Demo(phi_0_10, image_trad_cs, 0);
%         im_recon_0_10_damp((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp_d_amp;
        
        temp_tval3 =  TVAL3(phi_0_10, transpose(cs_meas_0_10(block,:)),33,33,opts);
        im_recon_0_10_tval3((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp_tval3;
    end
%     im_recon_0_10 = imrotate(im_recon_0_10, 90);
%     im_recon_0_10_crop = im_recon_0_10(199:462 ,298:561);
%     im_recon_0_10_damp = imrotate(im_recon_0_10_damp, 90);
%     im_recon_0_10_damp_crop = im_recon_0_10_damp(199:462 ,298:561);
    im_recon_0_10_tval3 = imrotate(im_recon_0_10_tval3, 90);
    im_recon_0_10_tval3_crop = im_recon_0_10_tval3(199:462 ,298:561);
    
    
    
    % MR = 0.04
    for block = 1:size(cs_meas_0_04,1) 
%         temp = net_0_04.forward({single(cs_meas_0_04(block,:))});
%         im_recon_0_04((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp{1};
%         
%         image_trad_cs = 255*phi_0_04'*transpose(cs_meas_0_04(block,:));
%         image_trad_cs = reshape(image_trad_cs, [33 33]);
%         temp_d_amp = CS_Imaging_Demo(phi_0_04, image_trad_cs, 0);
%         im_recon_0_04_damp((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp_d_amp;
        
        temp_tval3 =  TVAL3(phi_0_04, transpose(cs_meas_0_04(block,:)),33,33,opts);
        im_recon_0_04_tval3((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp_tval3;
    end
%     im_recon_0_04 = imrotate(im_recon_0_04, 90);
%     im_recon_0_04_crop = im_recon_0_04(199:462 ,298:561);
%     im_recon_0_04_damp = imrotate(im_recon_0_04_damp, 90);
%     im_recon_0_04_damp_crop = im_recon_0_04_damp(199:462 ,298:561);
    im_recon_0_04_tval3 = imrotate(im_recon_0_04_tval3, 90);
    im_recon_0_04_tval3_crop = im_recon_0_04_tval3(199:462 ,298:561);
    
    
    % MR = 0.01
    for block = 1:size(cs_meas_0_01,1)
%         temp = net_0_01.forward({single(cs_meas_0_01(block,:))});
%         im_recon_0_01((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp{1};
%         
%         image_trad_cs = 255*phi_0_01'*transpose(cs_meas_0_01(block,:));
%         image_trad_cs = reshape(image_trad_cs, [33 33]);
%         temp_d_amp = CS_Imaging_Demo(phi_0_01, image_trad_cs, 0);
%         im_recon_0_01_damp((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp_d_amp;
        
        temp_tval3 =  TVAL3(phi_0_01, transpose(cs_meas_0_01(block,:)),33,33,opts);
        im_recon_0_01_tval3((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp_tval3;
    end
%     im_recon_0_01 = imrotate(im_recon_0_01, 90);
%     im_recon_0_01_crop = im_recon_0_01(199:462 ,298:561);
%     im_recon_0_01_damp = imrotate(im_recon_0_01_damp, 90);
%     im_recon_0_01_damp_crop = im_recon_0_01_damp(199:462 ,298:561);
    im_recon_0_01_tval3 = imrotate(im_recon_0_01_tval3, 90);
    im_recon_0_01_tval3_crop = im_recon_0_01_tval3(199:462 ,298:561);
    

    % Save images
%     figure(1)
%     imshow(im_recon_0_10);
%     imwrite(im_recon_0_10_crop, [output_folder_0_10_reconnet, image_names{image}, '.png']);
    imwrite((im_recon_0_10_tval3_crop), [output_folder_0_10_tval3, image_names{image}, '.png']);
    save([output_folder_0_10_tval3, image_names{image}, '.mat'], 'im_recon_0_10_tval3_crop');
    
%     imwrite(uint8(im_recon_0_10_damp_crop), [output_folder_0_10_damp, image_names{image}, '.png']);
%     figure(2)
%     imshow(im_recon_0_04);
%     imwrite(im_recon_0_04_crop, [output_folder_0_04_reconnet, image_names{image}, '.png']);
    imwrite((im_recon_0_04_tval3_crop), [output_folder_0_04_tval3, image_names{image}, '.png']);
    save([output_folder_0_04_tval3, image_names{image}, '.mat'], 'im_recon_0_04_tval3_crop');
    
%     imwrite(uint8(im_recon_0_04_damp_crop), [output_folder_0_04_damp, image_names{image}, '.png']);
%     figure(3)
%     imshow(im_recon_0_01);
%     imwrite(im_recon_0_01_crop, [output_folder_0_01_reconnet, image_names{image}, '.png']);
    imwrite((im_recon_0_01_tval3_crop), [output_folder_0_01_tval3, image_names{image}, '.png']);
    save([output_folder_0_01_tval3, image_names{image}, '.mat'], 'im_recon_0_01_tval3_crop');
%     imwrite(uint8(im_recon_0_01_damp_crop), [output_folder_0_01_damp, image_names{image}, '.png']);

    image_names{image}
    clear im_recon_0_10 
    clear im_recon_0_04
    clear im_recon_0_01
    clear im_recon_0_10_tval3
    clear im_recon_0_04_tval3
    clear im_recon_0_01_tval3
    clear im_recon_0_10_damp
    clear im_recon_0_04_damp
    clear im_recon_0_01_damp
    
%     pause
end