# Learning-based models to obtain amplitude and phase reconstructions of defocused holograms
The proposed hybrid model aims to deliver both aberration-free in-focus amplitude and phase reconstructions, while accurately predicting in-focus distances, from out-of-focus holograms. The tasks were handled independently with the aim of later merging them.

The development began with a 7-category ResNet model for hologram processing, which includes a Fourier spectra amplitude layer. This was expanded to a 21-category  regression model, followed by transfer learning to create a final regression model with added dense layers.

For the image-to-image (hologram-to-reconstruction) task, a U-Net architecture was developed, leveraging the CNN from the regression model.


## Downloads
Trained models are available at: https://www.kaggle.com/models/mariareyb/autofocusing-model-for-dhm-holograms. 

## Reference
For more details on these models, please refer to the following publications. These are also the recommended citations if you use this tool in your work. (Pending Publication) 

## License & Copyright
Copyright 2024 Universidad EAFIT
Licensed under the MIT License; you may not use this file except in compliance with the License.

## Contact
Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)
- Maria Paula Rey (mpreyb@eafit.edu.co)
- Raul Castañeda (racastaneq@eafit.edu.co)


