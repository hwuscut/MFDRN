% Author：HaotianYan
% Date：2020/10/26

% 13 种无参考图像的清晰度评价指标
%  输出的excel中每一行分别为同一种评价指标，顺序依次为：
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
%     11. Range函数 (devided by E+05)
%     12. Vollaths函数 (devided by E+07)
%     13. Entropy 熵(times 100)


% 直接运行该evaluation.m文件即可，13个评价指标结果位于生成的excel文件中。
% 为了看起来更加直观，所有指标都除了一个值，如有需要可以自己调整。

% 【要设置的4个参数】
% 1、数据集所在的文件夹
file_path = 'HXY/614/3/';
% 2、是否为三通道RGB图 --> false / true
RGB_FLAG = true;
% 3、图片类型
files = dir(fullfile(file_path,'*.jpg'));
% 4、输出文件名称
outputName = 'bicubic-result_2x_0614_3_2loss.xlsx';


lengthFiles = length(files);
A = zeros(13,lengthFiles);  % 存储每一幅图像的清晰度评价值
for i =1:lengthFiles
    B{i} = files(i,1).name;     % 表头
end


% 1. EOG、2. Roberts (both devided by 1000000)
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
% 利用sobel算子gx,gy与图像做卷积，提取图像水平方向和垂直方向的梯度值
GX = 0;   % 图像水平方向梯度值
GY = 0;   % 图像垂直方向梯度值
FTenengrad = 0;   % 变量，暂时存储图像清晰度值
T = 0;   % 设置的阈值
for x = 2 : M-1
    for y = 2 : N-1
        GX = I(x-1,y+1) + 2*I(x,y+1) + I(x+1,y+1) - I(x-1,y-1) - 2*I(x,y-1) - I(x+1,y-1);
        GY = I(x+1,y-1) + 2*I(x+1,y) + I(x+1,y+1) - I(x-1,y-1) - 2*I(x-1,y) - I(x-1,y+1);
        SXY = sqrt(GX*GX + GY*GY); % 某一点的梯度值
        % 某一像素点梯度值大于设定的阈值，将该像素点考虑，消除噪声影响
        if SXY > T
            FTenengrad = FTenengrad + SXY*SXY;    % Tenengrad值定义
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
FBrenner = 0;        % 图像的Brenner值
for x = 1 : M-2      % Brenner函数原理，计算相差两个位置的像素点的灰度值
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
gama = 0;   %gama图像平均灰度值
%求gama
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
        FLaplace = FLaplace + IXXIYY*IXXIYY;        %取各像素点梯度的平方和作为清晰度值    
    end
end
A(6,L) = FLaplace / 1000000;
end
time = toc;
disp(['6. Time of Laplace = ', num2str(time), ' s.'])


% 7. SMD (灰度方差)函数、8. SMD2 (灰度方差乘积)函数 (both devided by 100000)
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    I = double(I);
    [M, N] = size(I);
    FSMD = 0;
    FSMD2 = 0;
    for x = 1 : M-1
        for y = 2 : N-1
            % x方向和y方向的相邻像素灰度值之差的的平方和作为清晰度值
            FSMD = FSMD + abs(I(x,y)-I(x,y-1)) + abs(I(x,y)-I(x+1,y));
            FSMD2 = FSMD2 + (I(x,y)-I(x+1,y))*(I(x,y)-I(x,y+1));
        end
    end

A(7,L) = FSMD / 100000;
A(8,L) = FSMD2 / 100000;
end
time = toc;
disp(['7. & 8. Time of SMD & SMD2 = ', num2str(time), ' s.'])


% 9. DFT(二维离散傅里叶变换) (devided by 1000000)
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    if RGB_FLAG
        I = rgb2gray(I);
    end
    I = double(I);
    [M, N] = size(I);       
    fftI = fft2(I);   % 进行二维离散傅里叶变换
    sfftI = fftshift(fftI);   % 移位，直流分量移到图像中心
    magnitude = abs(sfftI);      % 取模值
    FDFT = 0;
for u = 1 : M
    for v = 1 : N
        FDFT = FDFT + sqrt(u*u+v*v)*magnitude(u,v);      % 基于离散傅里叶变换的清晰度评价函数
    end
end
A(9,L) = FDFT / (M*N*1000000);
end
time = toc;
disp(['9. Time of DFT = ', num2str(time), ' s.'])


% 10. DCT(离散余弦变换) (devided by 1000)
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    if RGB_FLAG
        I = rgb2gray(I);
    end
    I = double(I) + 10*randn(size(I));
    [M, N] = size(I);
    dctI = dct2(I);   % 进行二维离散余弦变换
    magnitude = abs(dctI);      % 取模值
    FDCT = 0;
for u = 1 : M
    for v = 1 : N
        FDCT = FDCT + (u+v)*magnitude(u,v);      % 基于离散余弦变换的清晰度评价函数
    end
end
A(10,L) = FDCT / (M*N*1000);
end
time = toc;
disp(['10. Time of DCT = ', num2str(time), ' s.'])



% 11. Range函数(基于统计学) (devided by 100000)
gray_level = 32; % 灰度直方图中划分的灰度等级
temp = zeros(1,gray_level);
tic
for L = 1 : lengthFiles
    I = imread(strcat(file_path,files(L).name));
    if RGB_FLAG
        I = rgb2gray(I);
    end
    I = double(I); 
    % imhist()：用来画灰度分布直方图
    %   count：表示某一灰度区间的像素个数
    %   K：表示灰度区间取值
    [count,K] = imhist(I,gray_level);
for y = 1 : gray_level
    temp(1,y) = count(y) * K(y);
end
A(11,L) = (max(temp)-min(temp)) / 100000;
end
time = toc;
disp(['11. Time of Range = ', num2str(time), ' s.'])


% 12. Vollaths 函数(基于统计学) (devided by 10000000)
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


% 13. Entropy 熵(times 100)
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
    level=256;	% 图像的灰度级0-255
    count = zeros(level, 1);    % 存储图像灰度出现次数
    for x = 1 : M
        for y = 1 : N
            ImgLevel = I(x, y) + 1;     % 获取图像的灰度级
            count(ImgLevel) = count(ImgLevel) + 1;	% 统计每个灰度级像素的出现的次数
        end
    end
for k = 1: level
    Ps(k)=count(k) / ImgSize;	% 计算每一个灰度级像素点所占的概率
end

if Ps(k)~=0     % 去掉概率为0的像素点
    FEntropy=-Ps(k)*log2(Ps(k))+FEntropy;
end
 
 A(13,L)=FEntropy * 100;
end
time=toc;
disp(['13. Time of Entropy = ', num2str(time), ' s.'])


xlswrite(outputName, B, 'Sheet1', 'A1');
xlswrite(outputName, A, 'Sheet1', 'A2');
