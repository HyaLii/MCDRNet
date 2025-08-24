# MCDRNet: A Multi-Granularity Approach for All-in-One Image Restoration via Contrast-Guided Degradation Reconstruction!

PyTorch implementation for A Multi-Granularity Approach for All-in-One Image Restoration via Contrast-Guided Degradation Reconstruction (MCDRNet) 

## Dependencies

* Python == 3.8.11
* Pytorch == 1.7.0 
* mmcv-full == 1.3.11 

## Dataset

You could find the dataset we used in the paper at following:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://ece.uwaterloo.ca/~k29ma/exploration/), 

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

## Demo

You could download the pre-trained model from [Quark](https://pan.quark.cn/s/6cb9a260af4b) (password: 1P6k). Remember to put the pre-trained model into ckpt/

If you only need the visual results, you could put the test images into test/demo/ and use the following command to restore the test image:

```bash
python demo.py --mode 3
```

where mode == 3 means we use the checkpoint trained on all-in-one setting. (0 for denoising, 1 for deraining and 2 for dehazing)

## Training

If you want to re-train our model, you need to first put the training set into the data/, and use the following command:

```bash
python train.py
```

ps. To train with different combinations of corruptions, you could modify the "de_type" in option.py.

## Testing

If you want to test our model and get the psnr and ssim, you need to put the testing set into the test/, where several examples are given. Then, you could use the following command:

```bash
python test.py --mode 3
```

where mode == 3 means we use the checkpoint trained on all-in-one setting. (0 for denoising, 1 for deraining and 2 for dehazing)

