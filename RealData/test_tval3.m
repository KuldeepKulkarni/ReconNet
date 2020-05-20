clear
close all
rate = 0.25;
sigma = [0 10 20 30];
load('D:\Codes\CSCNN\CSCNN_test\phi_suhas\phi_0_25_1089.mat');
addpath(genpath('D:\Codes\CSCNN\BM3D'));
addpath(genpath('D:\Codes\CSCNN\TV_AL3\TVAL3_beta2.4'));

%Dataset for testing
test_images_dir = 'D:\Datasets\CSCNN\Images_for_CS_recon';
test_images = dir(test_images_dir);

psnr_tval3 = zeros(length(sigma),11);
time_tval3 = zeros(1,11);

opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;

for noise = 1:length(sigma)
    dest_folder = strcat('D:\Datasets\CSCNN\Results_cr_0_25\Gaussian_sigma_', num2str(sigma(noise)), '\');
    folder_tval3 = [dest_folder 'tval3\'];
    for image_number = 1:11
        image_name = test_images(image_number+2).name;
        input_im = double(imread(fullfile(test_images_dir,image_name)));

        input_im_nn = im2double(imread(fullfile(test_images_dir,image_name))); %Input for the CSCNN and Alinet

        block_size = 33;
        [row, col] = size(input_im);
        row_pad = block_size-mod(row,block_size);
        col_pad = block_size-mod(col,block_size);
        im_pad = [input_im, zeros(row,col_pad)];
        im_pad = [im_pad; zeros(row_pad,col+col_pad)];

        im_pad_nn = [input_im_nn, zeros(row,col_pad)];
        im_pad_nn = [im_pad_nn; zeros(row_pad,col+col_pad)];


        for i = 1:size(im_pad,1)/33
            for j = 1:size(im_pad,2)/33
                ori_im = im_pad((i-1)*33+1:i*33,(j-1)*33+1:j*33);
                ori_im_nn = im_pad_nn((i-1)*33+1:i*33,(j-1)*33+1:j*33);
                
                %TVAL3
                start_time = tic;   
                randn('seed',0);
                y = phi*ori_im(:);
                z = y + (sigma(noise))*randn(size(y));
                temp_tval3 =  TVAL3(phi,z,33,33,opts);
                end_time = toc;
                time_block = end_time - start_time;
                time_tval3(image_number) = time_tval3(image_number) + time_block;
                im_comp_tval3((i-1)*33+1:i*33,(j-1)*33+1:j*33) = temp_tval3;
                                
            end
        end

        image_name = strtok(image_name, '.');
        image_name = [image_name, '.mat'];

        rec_im_tval3 = im_comp_tval3(1:row, 1:col,:);
        save([folder_tval3 image_name],'rec_im_tval3');

        diff_tval3 = input_im - rec_im_tval3;
        sig_tval3 = sqrt(mean(input_im(:).^2));
        diff_tval3 = diff_tval3(:);
        rmse_tval3 = sqrt(mean(diff_tval3(:).^2));
        psnr_tval3(noise, image_number) = 20*log10(255/rmse_tval3);
        
        
        clear im_comp_nlr_cs im_comp_d_amp im_comp_cscnn im_comp_alinet im_comp_tval3

        disp(image_number);
        save(strcat('cr_0_10_sigma_',num2str(sigma(noise)),'_tval3_variables.mat'));
    end
end