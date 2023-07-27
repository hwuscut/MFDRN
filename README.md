<<<<<<< HEAD
# IMDN
Deep learning-based image super-resolution restoration for mobile infrared imaging system

# Requirements
python 3.6;
pytorch==0.4.0;
pillow;
h5py;
matplotlib;
numpy.

# Usage
First, download the training dataset and place it in IR_ In the data folder;

Second, train the network  python train_MFDRN_2.py;

Third, do testing  python test_MFDRN_2.py  --test_hr_folder ./Test_Datasets/HR/ --test_lr_folder ./Test_Datasets/LR_2/ --output_folder ./results/MFDRN_duibi_x2 --checkpoint ./checkpoint_MFDRN_duibi_x2/epoch_100.pth --upscale 2.

