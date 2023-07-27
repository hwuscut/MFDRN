% Author��HaotianYan
% Date��2020/10/26

% 13 ���޲ο�ͼ�������������ָ��
%  �����excel��ÿһ�зֱ�Ϊͬһ������ָ�꣬˳������Ϊ��
%       1. EOG (devided by E+06)
%       2. Roberts (devided by E+06)
%       3. Tenengrad (devided by E+08)
%       4. Brenner (devided by E+07)
%       5. Variance (devided by E+08)
%       6. Laplace (devided by E+06)
%       7. SMD (devided by E+05)
%       8. SMD2 (devided by E+05)
%       9. DFT (devided by E+06)
%     10. DCT (devided by E+03)
%     11. Range���� (devided by E+05)
%     12. Vollaths���� (devided by E+07)
%     13. Entropy ��(times 100)


% ֱ�����и�evaluation.m�ļ����ɣ�13������ָ����λ�����ɵ�excel�ļ��С�
% Ϊ�˿���������ֱ�ۣ�����ָ�궼����һ��ֵ��������Ҫ�����Լ�������

% ��Ҫ���õ�4��������
% 1�����ݼ����ڵ��ļ���
file_path = 'HXY/614/3/';
% 2���Ƿ�Ϊ��ͨ��RGBͼ --> false / true
RGB_FLAG = true;
% 3��ͼƬ����
files = dir(fullfile(file_path,'*.jpg'));
% 4������ļ�����
outputName = 'bicubic-result_2x_0614_3_2loss.xlsx';


lengthFiles = length(files);
A = zeros(13,lengthFiles);  % �洢ÿһ��ͼ�������������ֵ
for i =1:lengthFiles
    B{i} = files(i,1).name;     % ��ͷ
end


% 1. EOG��2. Roberts (both devided by 1000000)
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    I = double(I);
    [M, N] = size(I);
    FEOG = 0;
    FRobert = 0;
for x = 1 : M-1
    for y = 1 : N-1
         FEOG = FEOG + (I(x+1,y)-I(x,y))*(I(x+1,y)-I(x,y)) + (I(x,y+1)-I(x,y))*(I(x,y+1)-I(x,y));
         FRobert = FRobert + (abs(I(x,y)-I(x+1,y+1)) + abs(I(x+1,y)-I(x,y+1)));
    end
end

A(1, L) = FEOG / 1000000;
A(2, L) = FRobert / 1000000;
end
time = toc;
disp(['1. & 2. Time of EOG & Roberts = ', num2str(time), ' s.'])


% 3. Tenengrad (devided by 100000000)
tic
for L = 1 : lengthFiles
I = imread(strcat(file_path,files(L).name));
I = double(I);
[M, N] = size(I);
% ����sobel����gx,gy��ͼ�����������ȡͼ��ˮƽ����ʹ�ֱ������ݶ�ֵ
GX = 0;   % ͼ��ˮƽ�����ݶ�ֵ
GY = 0;   % ͼ��ֱ�����ݶ�ֵ
FTenengrad = 0;   % ��������ʱ�洢ͼ��������ֵ
T = 0;   % ���õ���ֵ
for x = 2 : M-1
    for y = 2 : N-1
        GX = I(x-1,y+1) + 2*I(x,y+1) + I(x+1,y+1) - I(x-1,y-1) - 2*I(x,y-1) - I(x+1,y-1);
        GY = I(x+1,y-1) + 2*I(x+1,y) + I(x+1,y+1) - I(x-1,y-1) - 2*I(x-1,y) - I(x-1,y+1);
        SXY = sqrt(GX*GX + GY*GY); % ĳһ����ݶ�ֵ
        % ĳһ���ص��ݶ�ֵ�����趨����ֵ���������ص㿼�ǣ���������Ӱ��
        if SXY > T
            FTenengrad = FTenengrad + SXY*SXY;    % Tenengradֵ����
        end
    end
end

A(3,L) = FTenengrad / 100000000;
end
time = toc;
disp(['3. Time of Tenengrad = ', num2str(time), ' s.'])


% 4. Brenner (devided by 10000000)
tic
for L = 1 : lengthFiles
I = imread(strcat(file_path,files(L).name));
I = double(I);
[M, N] = size(I);
FBrenner = 0;        % ͼ���Brennerֵ
for x = 1 : M-2      % Brenner����ԭ�������������λ�õ����ص�ĻҶ�ֵ
    for y = 1 : N
        FBrenner = FBrenner + (I(x+2,y)-I(x,y))*(I(x+2,y)-I(x,y));
    end
end

A(4,L) = FBrenner / 10000000;
end
time = toc;
disp(['4. Time of Brenner = ', num2str(time), ' s.'])


% 5. Variance (devided by 100000000)
tic
for L = 1 : lengthFiles
I = imread(strcat(file_path,files(L).name));
I = double(I);
[M, N] = size(I);
gama = 0;   %gamaͼ��ƽ���Ҷ�ֵ
%��gama
for x = 1 : M
    for y = 1 : N
        gama = gama + I(x,y);
    end
end
gama = gama/(M*N);

FVariance = 0;
for x = 1 : M
    for y = 1 : N
        FVariance = FVariance + (I(x,y)-gama)*(I(x,y)-gama);
    end
end
A(5,L) = FVariance / 100000000;
end
time = toc;
disp(['5. Time of Variance = ', num2str(time), ' s.'])
 

% 6. Laplace (devided by 1000000)
tic
for L = 1 : lengthFiles
I = imread(strcat(file_path,files(L).name));
I = double(I);
[M, N] = size(I);
FLaplace = 0;
for x = 2 : M-1
    for y = 2 : N-1
        IXXIYY = -4*I(x,y) + I(x,y+1) + I(x,y-1) + I(x+1,y) + I(x-1,y); 
        FLaplace = FLaplace + IXXIYY*IXXIYY;        %ȡ�����ص��ݶȵ�ƽ������Ϊ������ֵ    
    end
end
A(6,L) = FLaplace / 1000000;
end
time = toc;
disp(['6. Time of Laplace = ', num2str(time), ' s.'])


% 7. SMD (�Ҷȷ���)������8. SMD2 (�Ҷȷ���˻�)���� (both devided by 100000)
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    I = double(I);
    [M, N] = size(I);
    FSMD = 0;
    FSMD2 = 0;
    for x = 1 : M-1
        for y = 2 : N-1
            % x�����y������������ػҶ�ֵ֮��ĵ�ƽ������Ϊ������ֵ
            FSMD = FSMD + abs(I(x,y)-I(x,y-1)) + abs(I(x,y)-I(x+1,y));
            FSMD2 = FSMD2 + (I(x,y)-I(x+1,y))*(I(x,y)-I(x,y+1));
        end
    end

A(7,L) = FSMD / 100000;
A(8,L) = FSMD2 / 100000;
end
time = toc;
disp(['7. & 8. Time of SMD & SMD2 = ', num2str(time), ' s.'])


% 9. DFT(��ά��ɢ����Ҷ�任) (devided by 1000000)
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    if RGB_FLAG
        I = rgb2gray(I);
    end
    I = double(I);
    [M, N] = size(I);       
    fftI = fft2(I);   % ���ж�ά��ɢ����Ҷ�任
    sfftI = fftshift(fftI);   % ��λ��ֱ�������Ƶ�ͼ������
    magnitude = abs(sfftI);      % ȡģֵ
    FDFT = 0;
for u = 1 : M
    for v = 1 : N
        FDFT = FDFT + sqrt(u*u+v*v)*magnitude(u,v);      % ������ɢ����Ҷ�任�����������ۺ���
    end
end
A(9,L) = FDFT / (M*N*1000000);
end
time = toc;
disp(['9. Time of DFT = ', num2str(time), ' s.'])


% 10. DCT(��ɢ���ұ任) (devided by 1000)
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    if RGB_FLAG
        I = rgb2gray(I);
    end
    I = double(I) + 10*randn(size(I));
    [M, N] = size(I);
    dctI = dct2(I);   % ���ж�ά��ɢ���ұ任
    magnitude = abs(dctI);      % ȡģֵ
    FDCT = 0;
for u = 1 : M
    for v = 1 : N
        FDCT = FDCT + (u+v)*magnitude(u,v);      % ������ɢ���ұ任�����������ۺ���
    end
end
A(10,L) = FDCT / (M*N*1000);
end
time = toc;
disp(['10. Time of DCT = ', num2str(time), ' s.'])



% 11. Range����(����ͳ��ѧ) (devided by 100000)
gray_level = 32; % �Ҷ�ֱ��ͼ�л��ֵĻҶȵȼ�
temp = zeros(1,gray_level);
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    if RGB_FLAG
        I = rgb2gray(I);
    end
    I = double(I); 
    % imhist()���������Ҷȷֲ�ֱ��ͼ
    %   count����ʾĳһ�Ҷ���������ظ���
    %   K����ʾ�Ҷ�����ȡֵ
    [count,K] = imhist(I,gray_level);
for y = 1 : gray_level
    temp(1,y) = count(y) * K(y);
end
A(11,L) = (max(temp)-min(temp)) / 100000;
end
time = toc;
disp(['11. Time of Range = ', num2str(time), ' s.'])


% 12. Vollaths ����(����ͳ��ѧ) (devided by 10000000)
tic
for L = 1 : lengthFiles
I = imread(strcat(file_path,files(L).name));
I = double(I);
[M, N] = size(I);

FVollaths = 0;
for x = 1 : M-2
    for y = 1 : N
        FVollaths = FVollaths + I(x,y)*abs(I(x+1,y)-I(x+2,y));
    end
end
A(12,L) = FVollaths / 10000000;
end
time = toc;
disp(['12. Time of Vollaths = ', num2str(time), ' s.'])


% 13. Entropy ��(times 100)
tic
for L=1: lengthFiles
	I=imread(strcat(file_path,files(L).name));
    if RGB_FLAG
        I = rgb2gray(I);
    end
	I=double(I);
	[M, N]=size(I);
    ImgSize = M * N;
    
    FEntropy=0;
    level=256;	% ͼ��ĻҶȼ�0-255
    count = zeros(level, 1);    % �洢ͼ��Ҷȳ��ִ���
    for x = 1 : M
        for y = 1 : N
            ImgLevel = I(x, y) + 1;     % ��ȡͼ��ĻҶȼ�
            count(ImgLevel) = count(ImgLevel) + 1;	% ͳ��ÿ���Ҷȼ����صĳ��ֵĴ���
        end
    end
for k = 1: level
    Ps(k)=count(k) / ImgSize;	% ����ÿһ���Ҷȼ����ص���ռ�ĸ���
end

if Ps(k)~=0     % ȥ������Ϊ0�����ص�
    FEntropy=-Ps(k)*log2(Ps(k))+FEntropy;
end
 
 A(13,L)=FEntropy * 100;
end
time=toc;
disp(['13. Time of Entropy = ', num2str(time), ' s.'])


xlswrite(outputName, B, 'Sheet1', 'A1');
xlswrite(outputName, A, 'Sheet1', 'A2');
