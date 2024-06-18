# AgriSenAI
AgriSenAI is a Python-based desktop application developed to automate the processing and analysis of UAV thermal and multispectral images for precision agriculture. This tool supports agricultural professionals by providing critical insights for data-driven irrigation and crop management decisions.

## Features
Automated Image Processing: Extracts canopy temperature and vegetation indices from UAV thermal and multispectral images.
Geospatial Analysis: Accurate field and plot extraction, plant detection, and noise removal.
User-Friendly Interface: Developed with PyQt5, providing an intuitive and accessible interface.
Data Export: Outputs processed data into CSV files for easy analysis and decision-making.
## Motivation
Unmanned Aerial Vehicles (UAVs) have become essential in agriculture for monitoring crops with high spatial and temporal resolution. AgriSenAI addresses the challenges of processing and analyzing large-scale UAV imagery by automating these tasks, thus enhancing the efficiency and accuracy of precision agriculture practices.


## Import Data: 
Load thermal and RGB raster files.

![1](https://github.com/boaztulu/AgriSenAI--UAV-Image-Processing-for-Precision-Agriculture/assets/151384871/65a51a72-a446-4fa9-8b78-f14069953e2b)

## Field Extraction: 
Automatically delineate field boundaries within imported images.

![2](https://github.com/boaztulu/AgriSenAI--UAV-Image-Processing-for-Precision-Agriculture/assets/151384871/586937c7-082b-45c0-a29f-a8492c319bd2)

## Plot Extraction: 
Identify and extract individual experimental plots.

![3](https://github.com/boaztulu/AgriSenAI--UAV-Image-Processing-for-Precision-Agriculture/assets/151384871/39cb766f-2a88-4cea-8728-077ba00ee3d0)

## Plant Detection:
Detect plant areas using the Red-Green Ratio Index (RGRI) and Otsu's thresholding method.

![4](https://github.com/boaztulu/AgriSenAI--UAV-Image-Processing-for-Precision-Agriculture/assets/151384871/403253c5-b686-46c6-b351-207069266a68)

## Noise Reduction: 
Refine thermal data by reducing noise.

![6](https://github.com/boaztulu/AgriSenAI--UAV-Image-Processing-for-Precision-Agriculture/assets/151384871/d8e6c8a6-72a0-40f3-bd89-686af8be977a)

## Data Export: 
Export processed data to CSV for further analysis.

![image](https://github.com/boaztulu/AgriSenAI--UAV-Image-Processing-for-Precision-Agriculture/assets/151384871/35b623e7-10b2-4fb1-9e16-0ed57f170a32)


## Example

A comprehensive experiment was conducted at the University of Florida's Tropical Research and Education Center (TREC) to validate AgriSenAI. The app successfully processed datasets collected over three years from 24 plots under green beans and sweet corn, demonstrating its capability to handle large-scale remote sensing data.

## Contributing

We welcome contributions to enhance AgriSenAI. Please fork the repository and submit a pull request with your improvements.

## License

AgriSenAI is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

## Acknowledgments
The authors thank the Water Resource Research Group lab at the Tropical Research and Education Center, Institute of Food and Agricultural Sciences, University of Florida, for their invaluable assistance.
