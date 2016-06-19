clear
close all
clc

%% Select measurement rate (MR): 0.25, 0.10, 0.04 or 0.01
mr = '0.01';
mr_str = mr(3:end);

fprintf('Measurement Rate = %f \n', mr)

%% Initializations

% Dataset for testing
test_images_dir = './test_images';
test_images = dir(test_images_dir);
test_images = test_images(3:end);

output_dir = ['./reconstruction_results/mr_0_', mr_str, '/'];

% Initialize Caffe
addpath(genpath('../matlab'))
try
    caffe.reset_all();
catch
    caffe.reset_all();
end
caffe.set_mode_gpu();

% Prototxt file for the selected MR
prototxt_file = ['./deploy_prototxt_files/reconnet_0_', mr_str, '.prototxt'];

% Caffemodel for the selected MR
caffemodel = ['./caffemodels/reconnet_0_', mr_str, '.caffemodel'];

% Load the measurement matrix for the selected MR
load(['../phi/phi_0_', mr_str, '_1089.mat']);

psnr = zeros(11,1);
time_complexity = zeros(11,1);

%%
for image_number = 1:length(test_images)
    
    image_name = test_images(image_number).name;
    input_im_nn = im2double(imread(fullfile(test_images_dir,image_name))); %Input for the ReconNet
    block_size = 33;
    num_blocks = ceil(size(input_im_nn,1)/block_size)*ceil(size(input_im_nn,2)/block_size);
    
    modify_prototxt(prototxt_file, num_blocks);
    
    net = caffe.Net(prototxt_file, caffemodel, 'test');
        
    % Determine the size of zero pad
    [row, col] = size(input_im_nn);
    row_pad = block_size-mod(row,block_size);
    col_pad = block_size-mod(col,block_size);
   
    % Do zero padding
    im_pad_nn = [input_im_nn, zeros(row,col_pad)];
    im_pad_nn = [im_pad_nn; zeros(row_pad,col+col_pad)];
    
    count = 0;
    for i = 1:size(im_pad_nn,1)/block_size
        for j = 1:size(im_pad_nn,2)/block_size
            % Access the (i,j)th block of image 
            ori_im_nn = im_pad_nn((i-1)*block_size+1:i*block_size,(j-1)*block_size+1:j*block_size);
            count = count + 1;
            %CSCNN - Take the compressed measurements of the block
            y = phi*ori_im_nn(:);
            input_deep(count,1,:,1) = y;
        end
    end
    start_time = tic;
    
    % input_deep contains the set of CS measurements of all block,
    % net.forward compute reconstructions of all blocks parallelly
    temp = net.forward({permute(input_deep,[4 3 2 1])});
    
    %Rearrange the reconstructions to form the final image im_comp_cscnn
    count = 0;
    for i = 1:size(im_pad_nn,1)/block_size
        for j = 1:size(im_pad_nn,2)/block_size
            count = count + 1;
            im_comp((i-1)*block_size+1:i*block_size,(j-1)*block_size+1:j*block_size) = temp{1}(:,:,1,count);
        end
    end
    time_complexity(image_number) = toc(start_time);

    rec_im = im_comp(1:row, 1:col,:);
    [~,name,~] = fileparts(image_name);
    imwrite(rec_im, [output_dir name '.png']);
    
    %imshow(rec_im)
    %pause;
    
    diff = input_im_nn - rec_im;
    sig = sqrt(mean(input_im_nn(:).^2));
    diff = diff(:);
    rmse = sqrt(mean(diff(:).^2));
    psnr(image_number) = 20*log10(1/rmse);
    
    clear im_comp temp input_deep
    
    fprintf('\n %15s: PSNR = %f dB, Time = %f  seconds\n', image_name, psnr(image_number), time_complexity(image_number))    
end

fprintf(['\n All reconstruction results are saved in ', output_dir, '\n'])