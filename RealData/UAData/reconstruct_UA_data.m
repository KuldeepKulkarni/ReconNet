clear
close all
clc

load experimentalData.mat
num_mats = 1; % Number of measurement matrices
num_runs = 10; % Number of runs per measurement
num_images = 8;
image_names = {'Parrots', 'barbara', 'boats','house','foreman', 'cameraman', 'lena256'};
% image_names = {'Parrots'};
% image_names = {'Monarch'};
output_folder_0_10 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images/mr_0_10/';
output_folder_0_04 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images/mr_0_04/';
output_folder_0_01 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/reconstructed_images/mr_0_01/';

addpath(genpath('/home/slohit/caffe-master/matlab'))
try
    caffe.reset_all();
catch
    caffe.reset_all();
end
caffe.set_mode_gpu();

%% Normalize the measurements based on the respective calibration files -- find a and b for each run
% a = zeros(1,num_mats);
% b = zeros(1,num_mats);
% for mat = 1:num_mats
%     if mat < num_mats
%         white_meas = zeros(1,120);
%         black_meas = zeros(1,120);
%     else
%         white_meas = zeros(1,74);
%         black_meas = zeros(1,74);
%     end
%     
%     for runs = 1:num_runs
%         white_meas = white_meas + sum(experimentalData{1,4}.measures.white{1,run},1); %Doing everything only for the last mat
%         black_meas = black_meas + sum(experimentalData{1,4}.measures.black{1,run},1);
%     end
%     white_meas = white_meas/4020;
%     white_meas = white_meas';
%     black_meas = black_meas/4020;
%     black_meas = black_meas';
%     
%     %How to find a and b from this??
%     
% end

% Point to prototxt files
prototxt_0_10 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/prototxt/ReconNet_learn_mm_0_10.prototxt';
prototxt_0_04 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/prototxt/ReconNet_learn_mm_0_04.prototxt';
prototxt_0_01 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/prototxt/ReconNet_learn_mm_0_01.prototxt';

% Point to caffemodel files
caffemodel_0_10 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/caffemodel/ReconNet_learn_mm_0_10_iter_375000.caffemodel';
caffemodel_0_04 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/caffemodel/ReconNet_learn_mm_0_04_iter_160000.caffemodel';
caffemodel_0_01 = '/media/slohit/DATAPART1/Codes/PAMI_ReconNet/learn_mm/UAData/caffemodel/ReconNet_learn_mm_0_01_iter_160000.caffemodel';

% Create networks
net_0_10 = caffe.Net(prototxt_0_10, caffemodel_0_10, 'test');
net_0_04 = caffe.Net(prototxt_0_04, caffemodel_0_04, 'test');
net_0_01 = caffe.Net(prototxt_0_01, caffemodel_0_01, 'test');

for image = 1:length(image_names)
    cs_meas_3 = zeros(401,120);
    cs_meas_4 = zeros(403,74);
    %% Normalize and group measurements of each image based on the measurement rate
    all_cs_meas_3 = getfield(experimentalData{1,3}.measures,image_names{image});
    for runs = 1:num_runs
        cs_meas_3 = cs_meas_3 + all_cs_meas_3{1,runs};
    end
    cs_meas_3 = cs_meas_3/num_runs;
    
    all_cs_meas_4 = getfield(experimentalData{1,4}.measures,image_names{image});
    for runs = 1:num_runs
        cs_meas_4 = cs_meas_4 + all_cs_meas_4{1,runs};
    end
    cs_meas_4 = cs_meas_4/num_runs;
%     a = 255;
%     b = 0;
%     cs_meas = a*cs_meas + b;
    
    %     0.10
    min_op_0_10 = -6.9963;
    max_op_0_10 = 6.8763;
    
%     0.04
    min_op_0_04 = -6.3624;
    max_op_0_04 = 6.0307;

%     0.01
    min_op_0_01 = -5.4070;
    max_op_0_01 = 5.4789;
    
    cs_meas_0_10_part_1 = cs_meas_3(:,33:120);
%     cs_meas_0_10_part_1 = min_op_0_10 + (max_op_0_10 - min_op_0_10)*(cs_meas_0_10_part_1 - min(cs_meas_0_10_part_1(:)))/(max(cs_meas_0_10_part_1(:)) - min(cs_meas_0_10_part_1(:)));
    
    cs_meas_0_10_part_2 = cs_meas_4(:,1:21);
%     cs_meas_0_10_part_2 = min_op_0_10 + (max_op_0_10 - min_op_0_10)*(cs_meas_0_10_part_2 - min(cs_meas_0_10_part_2(:)))/(max(cs_meas_0_10_part_2(:)) - min(cs_meas_0_10_part_2(:)));
%     
%     cs_meas_0_10 = [cs_meas_0_10_part_1, cs_meas_0_10_part_2];
    
    cs_meas_0_04 = cs_meas_4(:,22:64);
%     cs_meas_0_04 = min_op_0_04 + (max_op_0_04 - min_op_0_04)*(cs_meas_0_04 - min(cs_meas_0_04(:)))/(max(cs_meas_0_04(:)) - min(cs_meas_0_04(:)));
    cs_meas_0_04 = 175*cs_meas_0_04;

    cs_meas_0_01 = cs_meas_4(:,65:74); 
%     cs_meas_0_01 = min_op_0_01 + (max_op_0_01 - min_op_0_01)*(cs_meas_0_01 - min(cs_meas_0_01(:)))/(max(cs_meas_0_01(:)) - min(cs_meas_0_01(:)));
    cs_meas_0_01 = 75*cs_meas_0_01;
    
    calibration = experimentalData{1,4}.calibration;
    num_blks = length(calibration.blocks);
    for i = 1:num_blks
        x_blk_ind(i) = calibration.blocks(i).xBlock;
        y_blk_ind(i) = calibration.blocks(i).yBlock;
    end
    num_blk_x = max(x_blk_ind) - min(x_blk_ind) + 1;
    num_blk_y = max(y_blk_ind) - min(y_blk_ind) + 1;
    x_blk_ind = x_blk_ind - min(x_blk_ind)+1;
    y_blk_ind = y_blk_ind - min(y_blk_ind)+1;
    
    
    
    calibration2 = experimentalData{1,3}.calibration;
    num_blks2 = length(calibration2.blocks);
    for i = 1:num_blks2
        x_blk_ind2(i) = calibration2.blocks(i).xBlock;
        y_blk_ind2(i) = calibration2.blocks(i).yBlock;
    end
    num_blk_x2 = max(x_blk_ind2) - min(x_blk_ind2) + 1;
    num_blk_y2 = max(y_blk_ind2) - min(y_blk_ind2) + 1;
    x_blk_ind2 = x_blk_ind2 - min(x_blk_ind2)+1;
    y_blk_ind2 = y_blk_ind2 - min(y_blk_ind2)+1;
    
    % Combining measurements for 0.10
    block_num = 1;
    for i = 1:num_blks
        indx = find(x_blk_ind2 == x_blk_ind(i));
        indy = find(y_blk_ind2 == y_blk_ind(i));
        if length(indx) > 0 && length(indy) > 0
            intersection = intersect(indx, indy);
            if length(intersection) > 0
                cs_meas_0_10(block_num, :) = [cs_meas_0_10_part_1(intersection,:) , cs_meas_0_10_part_2(i,:)];
                
                block_location_x(block_num) = calibration.blocks(i).xBlock;
                block_location_y(block_num) = calibration.blocks(i).yBlock;
                block_num = block_num + 1;
            end
        end
    end
    size(cs_meas_0_10)
    
    
%     cs_meas_0_10 = min_op_0_10 + (max_op_0_10 - min_op_0_10)*(cs_meas_0_10 - min(cs_meas_0_10(:)))/(max(cs_meas_0_10(:)) - min(cs_meas_0_10(:)));
    cs_meas_0_10 = 100*cs_meas_0_10;
    block_location_x = block_location_x - min(block_location_x) + 1;
    block_location_y = block_location_y - min(block_location_y) + 1;

    %% Reconstruct images
    
    % For most images: use 199:462, 298:561
    % For Monarch: use
    
    for block = 1:size(cs_meas_0_10,1)
        temp = net_0_10.forward({single(cs_meas_0_10(block,:))});
        im_recon_0_10((num_blk_y2-block_location_y(block))*33+1:(num_blk_y2-block_location_y(block)+1)*33,(block_location_x(block)-1)*33+1:block_location_x(block)*33) = temp{1};
    end
    im_recon_0_10 = imrotate(im_recon_0_10, 90);
    im_recon_0_10_crop = im_recon_0_10(199:462 ,298:561);
    
    for block = 1:size(cs_meas_0_04,1) 
        temp = net_0_04.forward({single(cs_meas_0_04(block,:))});
        im_recon_0_04((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp{1};
    end
    im_recon_0_04 = imrotate(im_recon_0_04, 90);
    im_recon_0_04_crop = im_recon_0_04(199:462 ,298:561);
    
    for block = 1:size(cs_meas_0_01,1)
        temp = net_0_01.forward({single(cs_meas_0_01(block,:))});
        im_recon_0_01((num_blk_y-y_blk_ind(block))*33+1:(num_blk_y-y_blk_ind(block)+1)*33,(x_blk_ind(block)-1)*33+1:x_blk_ind(block)*33) = temp{1};
    end
    im_recon_0_01 = imrotate(im_recon_0_01, 90);
    im_recon_0_01_crop = im_recon_0_01(199:462 ,298:561);
    
    % Save images
    figure(1)
    imshow(im_recon_0_10);
    imwrite(im_recon_0_10_crop, [output_folder_0_10, image_names{image}, '.png'])
    figure(2)
    imshow(im_recon_0_04);
    imwrite(im_recon_0_04_crop, [output_folder_0_04, image_names{image}, '.png'])
    figure(3)
    imshow(im_recon_0_01);
    imwrite(im_recon_0_01_crop, [output_folder_0_01, image_names{image}, '.png'])
    
    image_names{image}
    clear im_recon_0_10 
    clear im_recon_0_04
    clear im_recon_0_01
    pause
end